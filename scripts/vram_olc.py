"""4 GiB VRAM'de hangi yigin boyutu GERCEKTEN siginiyor - olcer, tahmin etmez.

Referans ayar MINIBATCH_SIZE=16 @ 480x640. Bu kartta ne oldugunu gorelim.
Egitim adiminin tamamini kosar: ileri gecis + kayip + geri yayilim + optimiser.

TUZAK - "OOM vermedi" ile "siginiyor" ayni sey DEGIL
  Windows WDDM surucusu VRAM tasinca sessizce sistem RAM'ine tasirir. Kod
  cokmez, sadece 10-50 kat yavaslar. Olcumde 4 GiB kartta 11 GiB tepe tahsis
  gorulmesi bundandir. Bu yuzden yalnizca OOM'a bakmak yetmez; sureyi de
  izlemek gerekir.

  Tasma nasil ayirt edilir: yigin buyudukce ornek basina sure normalde DUSER
  ya da sabitlenir (sabit ek yuk daha cok ornege dagilir). Sure YUKSELMEYE
  baslarsa neden bellek tasmasidir. Betik bunu iki isaretten yakalar:
    1) tepe tahsis kart kapasitesinin %85'ini gecti, ya da
    2) ornek basina sure bir onceki (daha kucuk) yigina gore %50'den fazla
       artti - ve o yigindan sonraki her yigin da tasmis sayilir.
  Kucuk yigini "en iyiyle" karsilastirmak yanlistir: orada sure zaten sabit
  ek yuk yuzunden yuksektir, tasma degil.
"""

import argparse
import sys
from pathlib import Path

import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "reference" / "TUS-REC2025-Challenge_baseline"))

from utils.funs import pair_samples, type_dim  # noqa: E402
from utils.network import build_model  # noqa: E402


def cikti_boyutu(kare, num_pred=1, pred_type="parameter", kose_nokta=4):
    """Agin cikti boyutu dizi uzunluguna gore degisir - sabit 6 DEGIL.

    NUM_SAMPLES kare icin (NUM_SAMPLES-1) cift olusuyor ve her cift icin
    ayri donusum tahmin ediliyor: pred_dim = 6 * cift sayisi.
      kare  2 -> 1 cift ->  6
      kare  5 -> 4 cift -> 24
      kare 10 -> 9 cift -> 54
    """
    ciftler = pair_samples(kare, num_pred, 0).shape[0]
    return type_dim(pred_type, kose_nokta, ciftler)


def dene(yigin, kare, amp, yukseklik=480, genislik=640):
    """Tek egitim adimini kosar. (basarili, tepe_GiB, sure_ms) dondurur."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pred_dim = cikti_boyutu(kare)
    model = build_model({"model_name": "efficientnet_b1"},
                        in_frames=kare, pred_dim=pred_dim).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    olcek = torch.cuda.amp.GradScaler(enabled=amp)
    try:
        x = torch.randn(yigin, kare, yukseklik, genislik, device="cuda")
        y = torch.randn(yigin, pred_dim, device="cuda")
        bas = torch.cuda.Event(enable_timing=True)
        son = torch.cuda.Event(enable_timing=True)
        bas.record()
        for _ in range(2):  # ilk adim ayirici isinmasi; ikincisi olculur
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                kayip = torch.nn.functional.mse_loss(model(x), y)
            olcek.scale(kayip).backward()
            olcek.step(opt)
            olcek.update()
        son.record()
        torch.cuda.synchronize()
        tepe = torch.cuda.max_memory_allocated() / 1024**3
        return True, tepe, bas.elapsed_time(son) / 2
    except torch.cuda.OutOfMemoryError:
        return False, torch.cuda.max_memory_allocated() / 1024**3, 0.0
    finally:
        del model, opt
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kare", type=int, nargs="+", default=[2],
                    help="NUM_SAMPLES degerleri (girdi kanal sayisi)")
    ap.add_argument("--yiginlar", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA yok")
    toplam = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{torch.cuda.get_device_name(0)}  {toplam:.1f} GiB")

    ozet = {}
    for kare in a.kare:
        print(f"\n{'=' * 62}")
        print(f"NUM_SAMPLES={kare}  (girdi 480x640x{kare}, "
              f"cikti {cikti_boyutu(kare)} boyut)  efficientnet_b1")
        print("=" * 62)
        ozet[kare] = tara(kare, a.yiginlar, toplam)

    if len(a.kare) > 1:
        print(f"\n{'=' * 62}")
        print("OZET - dizi uzunlugu buyudukce kullanilabilir yigin")
        print("=" * 62)
        print(f"{'NUM_SAMPLES':>12} {'AMP kapali':>12} {'AMP acik':>10}")
        for kare, (k_kapali, k_acik) in ozet.items():
            print(f"{kare:>12} {k_kapali or 'HICBIRI':>12} {k_acik or 'HICBIRI':>10}")


def tara(kare, yiginlar, toplam):
    """Bir dizi uzunlugu icin butun yiginlari olcer, kullanilabilir tavani dondurur."""
    print()
    olcumler = []  # (amp, yigin, tepe_GiB, adim_ms, ornek_basina_ms)
    for amp in (False, True):
        for y in yiginlar:
            ok, tepe, ms = dene(y, kare, amp)
            if not ok:
                print(f"  OOM: yigin {y}, AMP {'acik' if amp else 'kapali'}")
                break
            olcumler.append((amp, y, tepe, ms, ms / y))
    if not olcumler:
        print("  hicbir yigin kosmadi")
        return (0, 0)

    # TASMA TESPITI - docstring'deki iki isaret.
    # Her AMP ayari icin yiginlari artan sirada gezip ilk tasma noktasini bul;
    # o noktadan sonraki her yigin da tasmis sayilir.
    print(f"{'yigin':>6} {'AMP':>6} {'tepe VRAM':>11} {'adim':>9} {'ornek':>8}  karar")
    print("-" * 62)
    kullanilabilir = {False: 0, True: 0}
    for amp in (False, True):
        seri = sorted((m for m in olcumler if m[0] == amp), key=lambda m: m[1])
        tasma_basladi = False
        onceki_ob = None
        for _, y, tepe, ms, ob in seri:
            if not tasma_basladi:
                bellek_isareti = tepe > toplam * 0.85
                sure_isareti = onceki_ob is not None and ob > onceki_ob * 1.5
                tasma_basladi = bellek_isareti or sure_isareti
            if not tasma_basladi:
                kullanilabilir[amp] = max(kullanilabilir[amp], y)
            print(f"{y:>6} {'acik' if amp else 'kapali':>6} {tepe:>8.2f} GiB "
                  f"{ms:>7.0f}ms {ob:>6.0f}ms  "
                  f"{'TASTI - sistem RAMine' if tasma_basladi else 'kullanilabilir'}")
            onceki_ob = ob

    print(f"\nGERCEKTEN kullanilabilir en buyuk yigin:"
          f"  AMP kapali {kullanilabilir[False]},  AMP acik {kullanilabilir[True]}")
    hedef = 16
    if kullanilabilir[True] >= hedef:
        print(f"referans MINIBATCH_SIZE={hedef} dogrudan kosabilir.")
    elif kullanilabilir[True] > 0:
        kat = -(-hedef // kullanilabilir[True])  # yukari yuvarla
        print(f"referans MINIBATCH_SIZE={hedef} DOGRUDAN KOSAMAZ.")
        print(f"Esdeger yol: AMP acik + yigin {kullanilabilir[True]} + "
              f"{kat} adim gradyan biriktirme.")
    else:
        print(f"referans MINIBATCH_SIZE={hedef} KOSAMAZ - bu dizi uzunlugunda "
              f"yigin 1 bile tasiyor. Dizi uzunlugunu dusur ya da daha buyuk kart.")

    return (kullanilabilir[False], kullanilabilir[True])


if __name__ == "__main__":
    main()
