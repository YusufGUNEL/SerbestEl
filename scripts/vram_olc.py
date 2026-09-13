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

from utils.network import build_model  # noqa: E402


def dene(yigin, kare, amp, yukseklik=480, genislik=640):
    """Tek egitim adimini kosar. (basarili, tepe_GiB, sure_ms) dondurur."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model({"model_name": "efficientnet_b1"},
                        in_frames=kare, pred_dim=6).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    olcek = torch.cuda.amp.GradScaler(enabled=amp)
    try:
        x = torch.randn(yigin, kare, yukseklik, genislik, device="cuda")
        y = torch.randn(yigin, 6, device="cuda")
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
    ap.add_argument("--kare", type=int, default=2, help="NUM_SAMPLES (girdi kanali)")
    ap.add_argument("--yiginlar", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA yok")
    toplam = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{torch.cuda.get_device_name(0)}  {toplam:.1f} GiB")
    print(f"girdi 480x640, NUM_SAMPLES={a.kare}, efficientnet_b1\n")
    olcumler = []  # (amp, yigin, tepe_GiB, adim_ms, ornek_basina_ms)
    for amp in (False, True):
        for y in a.yiginlar:
            ok, tepe, ms = dene(y, a.kare, amp)
            if not ok:
                print(f"  OOM: yigin {y}, AMP {'acik' if amp else 'kapali'}")
                break
            olcumler.append((amp, y, tepe, ms, ms / y))

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
    else:
        kat = hedef // max(kullanilabilir[True], 1)
        print(f"referans MINIBATCH_SIZE={hedef} DOGRUDAN KOSAMAZ.")
        print(f"Esdeger yol: AMP acik + yigin {kullanilabilir[True]} + "
              f"{kat} adim gradyan biriktirme.")


if __name__ == "__main__":
    main()
