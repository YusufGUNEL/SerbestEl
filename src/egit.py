"""Egitim dongusu — referansin matematigi, calisabilir bir sarmal icinde.

NEDEN KENDI DONGUMUZ VAR
  Referans `train.py` Windows'ta kosamiyor: butun kod modul seviyesinde,
  `if __name__ == "__main__"` korumasi yok, ama DataLoader `num_workers=8`
  ile kuruluyor. Windows `spawn` kullandigi icin her isci surec betigi bastan
  calistirip yeni surec dogurmaya kalkiyor ve Python bunu hatayla durduruyor.
  (`src/referans_kos.py` num_workers=0 yapip bunu asiyor ama veri okuma tek
  surece dusuyor — gercek egitim icin fazla yavas.)

  Ikinci sebep: 4 GiB VRAM. Referansin `MINIBATCH_SIZE=16` ayari bu kartta
  dogrudan kosmuyor (olculdu: tepe 5,88 GiB, sistem RAM'ine tasiyor ve adim
  suresi 8 kat artiyor). Esdegeri: AMP acik + yigin 8 + 2 adim gradyan
  biriktirme.

NEYI DEGISTIRMIYORUZ
  Ag, kayip, etiket donusumleri, veri yukleyici ve ornekleme referanstan
  DOGRUDAN ithal ediliyor. Faz 2'nin amaci referansi iyilestirmek degil,
  referansi URETMEK. Matematige dokunulmadi; degisen yalnizca calistirma
  bicimi: koruma, AMP, gradyan biriktirme, denek bazli bolme, surdurulebilirlik.

  Gradyan biriktirme matematigi degistirmez: 8'lik iki yigindan gelen
  gradyanlar toplanip tek adimda uygulaniyor, 16'lik tek yigin gibi.
  Tek incelik kaybin biriktirme sayisina bolunmesi (yoksa gradyan iki kat olur).

KULLANIM
  python src/bolme.py --veri <veri>/frames_transfs        # bir kere
  python src/egit.py --veri <veri> --epok 20000
  python src/egit.py --devam                              # kaldigi yerden
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"
for p in (str(KOK), str(REFERANS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.bolme import yukle as bolme_yukle  # noqa: E402
from src.omurga import kur as omurga_kur  # noqa: E402
from src.temsil import (  # noqa: E402
    TahminDonusturucu,
    cikti_boyutu,
    donusume_matrise,
    son_katmani_birime_ayarla,
)
from src.veri import kume_kur, taramalari_bul  # noqa: E402
from utils.funs import pair_samples  # noqa: E402
from utils.loss import PointDistance  # noqa: E402
from utils.plot_functions import read_calib_matrices, reference_image_points  # noqa: E402
from utils.transform import LabelTransform, PointTransform  # noqa: E402

_DURDUR = False


def _sinyal(signum, cerceve):
    """Ctrl+C gelince epok ortasinda kesme; adim bitsin, checkpoint yazilsin."""
    global _DURDUR
    if _DURDUR:                      # ikinci Ctrl+C: hemen cik
        raise KeyboardInterrupt
    _DURDUR = True
    print("\n[durdurma istendi — bu epok bitince kaydedip cikilacak; "
          "tekrar Ctrl+C hemen keser]", flush=True)


def alt_kume(kok, denekler: list[str], ad: str, num_samples: int,
             sample_range: int):
    """Denek KIMLIKLERINDEN alt veri kumesi kurar (klasor sirasindan degil).

    Referans Dataset denekleri klasor sirasina gore indeksliyor ve denek
    basina TAM IKI tarama sart kosuyor. Ikisi de bizim icin calismiyor:
    bolmemiz kimlik tabanli, TUS-REC2024'te ise denek basina 24 tarama var.
    Bu yuzden `src/veri.py` icindeki kendi yukleyicimiz kullaniliyor.
    """
    alt = kume_kur(kok, denekler, num_samples=num_samples,
                   sample_range=sample_range, min_kare=sample_range)
    print(f"  {ad:<10} {len(denekler):>3} denek, {len(alt):>5} tarama")
    return alt


def olcum_dosyasi(yol: Path, satir: dict) -> None:
    yeni = not yol.exists()
    with open(yol, "a", newline="", encoding="utf-8") as f:
        y = csv.DictWriter(f, fieldnames=list(satir))
        if yeni:
            y.writeheader()
        y.writerow(satir)


def tutarlilik_kaybi(model, kareler, cikti, a, num_pairs, kalib, nokta_mm):
    """Ileri ve geri tahminin birbirini goturmesi kisiti (yol haritasi D hatti).

    Kare sirasi ters cevrilip ag ikinci kez kosturulur. Ilk gecis
    T(1->0), ikincisi T(0->1) tahmin eder. Ikisi gercekse bilesimleri
    BIRIM matris olmali: T(1->0) . T(0->1) = I.

    Kisit ek etiket istemiyor — gercek donusumler hic kullanilmiyor, yalnizca
    modelin kendi iki tahmini karsilastiriliyor. Bu yuzden dogrudan hata
    BIRIKMESINI hedefler: birikme, ard arda carpilan donusumlerin birbiriyle
    tutarsizligindan doguyor.

    Hata yine MM cinsinden olculuyor: bilesim referans kose noktalarina
    uygulanip noktalarin ne kadar kaydigina bakiliyor. Boylece kisit ana
    kayipla AYNI birimde olur ve agirlik katsayisi yorumlanabilir kalir
    (1.0 = bir milimetrelik tutarsizlik, bir milimetrelik konum hatasi kadar
    cezalandirilir).

    BEDELI DURUSTCE: ikinci ileri gecis adim suresini yaklasik iki katina
    cikarir. Sabit sure butcesinde bu, gorulen pencere sayisinin yariya
    inmesi demektir; deneyin kaydinda bu bedel de yaziyor.
    """
    ters = torch.flip(kareler, dims=[1])
    cikti_ters = model(ters / 255)
    T_ileri = donusume_matrise(a.donme_temsili, cikti, num_pairs, **kalib)
    T_geri = donusume_matrise(a.donme_temsili, cikti_ters, num_pairs, **kalib)
    bilesim = torch.matmul(T_ileri.float(), T_geri.float())
    kaymis = torch.matmul(bilesim, nokta_mm)[:, :, 0:3, :]
    beklenen = nokta_mm[0:3, :].expand_as(kaymis)
    return torch.nn.functional.mse_loss(kaymis, beklenen)


def main() -> int:
    global _DURDUR          # sure siniri ve Ctrl+C ayni bayragi kullaniyor
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--veri", type=Path, default=KOK / "data",
                    help="frames_transfs/ ve calib_matrix.csv iceren dizin")
    ap.add_argument("--bolme", type=Path, default=KOK / "configs" / "bolme.json")
    ap.add_argument("--kayit", type=Path, default=KOK / "results" / "faz2_referans")
    # --- referans degerleri: DEGISTIRME, Faz 2 referansi uretmek icin ---
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--sample-range", type=int, default=2)
    ap.add_argument("--num-pred", type=int, default=1)
    # --- Faz 4 degiskenleri: varsayilanlari Faz 2 kosumuyla BIREBIR ayni ---
    ap.add_argument("--donme-temsili", default="euler",
                    choices=["euler", "6b", "kuaterniyon", "matris"],
                    help="agin donmeyi hangi sayilarla tahmin edecegi (yol haritasi B hatti)")
    ap.add_argument("--kayip-uzayi", default="nokta", choices=["nokta", "parametre"],
                    help="kayip mm cinsinden nokta mesafesi mi, ham parametre mi (C hatti)")
    ap.add_argument("--lr-plan", default="sabit", choices=["sabit", "plato"],
                    help="plato: dogrulama duzelmeyince lr yariya iner")
    ap.add_argument("--omurga-onegitimli", action="store_true",
                    help="omurgayi ImageNet agirligiyla baslat (sizinti tasimaz)")
    ap.add_argument("--tutarlilik", type=float, default=0.0, metavar="AGIRLIK",
                    help="ileri/geri tutarlilik kisiti agirligi, 0 = kapali (D hatti)")
    # dest 'model_name' olmali: referansin build_model'i bu adi ariyor
    ap.add_argument("--model", dest="model_name", default="efficientnet_b1")
    ap.add_argument("--lr", type=float, default=1e-4)
    # --- 4 GiB karta uyarlama: matematigi degistirmez ---
    ap.add_argument("--yigin", type=int, default=8, help="olculen guvenli tavan")
    ap.add_argument("--biriktirme", type=int, default=2,
                    help="etkin yigin = yigin x biriktirme (referans 16)")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--amp-kapali", dest="amp", action="store_false")
    ap.add_argument("--isci", type=int, default=4, help="DataLoader num_workers")
    # --- kosum ---
    ap.add_argument("--epok", type=int, default=20000)
    ap.add_argument("--sure-siniri", type=float, default=0.0, metavar="DAKIKA",
                    help="bu sureden sonra epok bitince durdurup kaydet "
                         "(0 = sinirsiz). Referansin 20000 epoklik butcesi "
                         "bu donanimda gunler surer; deneyin butcesini epok "
                         "degil SURE olarak sabitlemek tekrar edilebilir kilar")
    ap.add_argument("--dogrulama-sikligi", type=int, default=25)
    ap.add_argument("--kayit-sikligi", type=int, default=100)
    ap.add_argument("--ara-kayit", type=int, default=0, metavar="EPOK",
                    help="her EPOK'ta agirliklari AYRI bir dosyaya yaz "
                         "(0 = kapali). Uzun kosumda bagdasimin egitim "
                         "olcegiyle nasil degistigini olcmek icin")
    ap.add_argument("--devam", action="store_true", help="son checkpoint'ten devam")
    ap.add_argument("--on-egitimli", type=Path,
                    default=REFERANS / "TUS-REC2024_model" / "model_weights",
                    help="referansin TUS-REC2024 agirliklari; 'yok' ile sifirdan")
    ap.add_argument("--tohum", type=int, default=20260915)
    a = ap.parse_args()

    a.label_type = {"nokta": "point", "parametre": "parameter"}[a.kayip_uzayi]
    torch.manual_seed(a.tohum)
    np.random.seed(a.tohum)
    a.kayit.mkdir(parents=True, exist_ok=True)
    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"aygit: {aygit}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB)"
             if aygit.type == "cuda" else ""))
    if a.amp and aygit.type != "cuda":
        a.amp = False
        print("AMP yalnizca CUDA'da anlamli, kapatildi")

    # ---------------- veri ----------------
    duzen, tum_taramalar = taramalari_bul(a.veri)
    denek_sayisi = len({t.denek for t in tum_taramalar})
    print(f"veri: {duzen} duzen, {denek_sayisi} denek, {len(tum_taramalar)} tarama "
          f"(denek basina {len(tum_taramalar)/denek_sayisi:.1f})")

    bolme = bolme_yukle(a.bolme)
    print(f"bolme: {a.bolme.name}  (tohum {bolme.tohum})")
    ok = dict(num_samples=a.num_samples, sample_range=a.sample_range)
    dset_egitim = alt_kume(a.veri, bolme.egitim, "egitim", **ok)
    dset_dogrulama = alt_kume(a.veri, bolme.dogrulama, "dogrulama", **ok)
    alt_kume(a.veri, bolme.test, "test", **ok)   # yalnizca var oldugunu dogrulamak icin
    print("  test kumesine EGITIM SIRASINDA DOKUNULMAZ")

    ortak = dict(num_workers=a.isci, pin_memory=(aygit.type == "cuda"),
                 persistent_workers=a.isci > 0)
    egitim_yukleyici = torch.utils.data.DataLoader(
        dset_egitim, batch_size=a.yigin, shuffle=True, drop_last=False, **ortak)
    dogrulama_yukleyici = torch.utils.data.DataLoader(
        dset_dogrulama, batch_size=1, shuffle=False, **ortak)

    # ---------------- donusumler (referanstan, degistirilmedi) ----------------
    ciftler = pair_samples(a.num_samples, a.num_pred, 0).to(aygit)
    olcek, rijit, tam = (t.to(aygit) for t in
                         read_calib_matrices(str(a.veri / "calib_matrix.csv")))
    ornek_kare = dset_egitim[0][0]
    nokta = reference_image_points(ornek_kare.shape[1:], 2).to(aygit)
    pred_dim = cikti_boyutu(a.donme_temsili, ciftler.shape[0])
    kalib = dict(image_points=nokta, tform_image_to_tool=tam,
                 tform_image_mm_to_tool=rijit, tform_image_pixel_to_mm=olcek)

    etiket_donusturucu = LabelTransform(a.label_type, pairs=ciftler, **kalib)
    tahmin_donusturucu = TahminDonusturucu(
        a.donme_temsili, a.kayip_uzayi, ciftler.shape[0], **kalib)
    noktaya = PointTransform(label_type=a.label_type, **kalib)
    # tutarlilik kisiti icin: mm cinsinden referans kose noktalari
    nokta_mm = torch.matmul(olcek, nokta)

    # ---------------- model ----------------
    model = omurga_kur(a.model_name, in_frames=a.num_samples, pred_dim=pred_dim,
                       onegitimli=a.omurga_onegitimli)
    # butun temsiller BIRIM donusumden basliyor — bkz. src/temsil.py aciklamasi
    son_katmani_birime_ayarla(model, a.donme_temsili, ciftler.shape[0])
    model = model.to(aygit)
    print(f"temsil {a.donme_temsili} -> cikti {pred_dim} sayi   "
          f"kayip uzayi {a.kayip_uzayi}   omurga "
          f"{'ImageNet' if a.omurga_onegitimli else 'rastgele'}"
          + (f"   tutarlilik {a.tutarlilik}" if a.tutarlilik else ""))
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    planlayici = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, threshold=1e-3, min_lr=1e-6)
        if a.lr_plan == "plato" else None)
    olcekleyici = torch.cuda.amp.GradScaler(enabled=a.amp)
    kayip_fn = torch.nn.MSELoss()
    mesafe_fn = PointDistance()

    baslangic, en_iyi = 0, float("inf")
    ckpt_yolu = a.kayit / "son_checkpoint.pt"
    if a.devam and ckpt_yolu.exists():
        ck = torch.load(ckpt_yolu, map_location=aygit)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        olcekleyici.load_state_dict(ck["olcekleyici"])
        if planlayici is not None and ck.get("planlayici"):
            planlayici.load_state_dict(ck["planlayici"])
        baslangic, en_iyi = ck["epok"] + 1, ck["en_iyi"]
        print(f"devam: epok {baslangic}, en iyi dogrulama mesafesi {en_iyi:.4f} mm")
    elif str(a.on_egitimli).lower() != "yok" and Path(a.on_egitimli).exists():
        model.load_state_dict(torch.load(a.on_egitimli, map_location=aygit))
        print(f"on egitimli agirlik yuklendi: {Path(a.on_egitimli).name}")
    else:
        print("sifirdan egitim (on egitimli agirlik yok)")

    etkin = a.yigin * a.biriktirme
    print(f"\nyigin {a.yigin} x {a.biriktirme} biriktirme = etkin {etkin}"
          f"  (referans 16)   AMP {'acik' if a.amp else 'kapali'}   isci {a.isci}")
    (a.kayit / "ayarlar.json").write_text(
        json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2), encoding="utf-8")

    signal.signal(signal.SIGINT, _sinyal)
    olcum = a.kayit / "olcumler.csv"
    print(f"\nepok {baslangic} -> {a.epok}\n")
    bas_zaman = time.time()

    for epok in range(baslangic, a.epok):
        model.train()
        top_kayip = top_mesafe = 0.0
        adim = 0
        optimizer.zero_grad(set_to_none=True)

        for i, (kareler, tforms, _, _) in enumerate(egitim_yukleyici):
            kareler = kareler.to(aygit, non_blocking=True)
            tforms = tforms.to(aygit, non_blocking=True)
            etiket = etiket_donusturucu(tforms, torch.linalg.inv(tforms))

            with torch.cuda.amp.autocast(enabled=a.amp):
                cikti = model(kareler / 255)
                tahmin = tahmin_donusturucu(cikti)
                ham = kayip_fn(tahmin, etiket)
                if a.tutarlilik:
                    ham = ham + a.tutarlilik * tutarlilik_kaybi(
                        model, kareler, cikti, a, ciftler.shape[0], kalib, nokta_mm)
                # biriktirme sayisina BOL: yoksa gradyan biriktirme kati kadar buyur
                kayip = ham / a.biriktirme

            olcekleyici.scale(kayip).backward()
            if (i + 1) % a.biriktirme == 0 or (i + 1) == len(egitim_yukleyici):
                olcekleyici.step(optimizer)
                olcekleyici.update()
                optimizer.zero_grad(set_to_none=True)

            top_kayip += kayip.item() * a.biriktirme
            top_mesafe += mesafe_fn(noktaya(tahmin.detach().float()),
                                    noktaya(etiket)).mean().item()
            adim += 1

        e_kayip, e_mesafe = top_kayip / adim, top_mesafe / adim

        # Sure sinirina epok SONUNDA bakiliyor ve Ctrl+C ile ayni yola
        # giriliyor: boylece dogrulama kosuyor, en iyi model ve checkpoint
        # yaziliyor, sonra cikiliyor. Yarim epokta kesmek hem olcumu bozar
        # hem de checkpoint'i tutarsiz birakirdi.
        if a.sure_siniri and (time.time() - bas_zaman) / 60 >= a.sure_siniri:
            _DURDUR = True
            print(f"\nsure siniri doldu ({a.sure_siniri:.0f} dk)", flush=True)

        dogrulama = {}
        if epok % a.dogrulama_sikligi == 0 or epok == a.epok - 1 or _DURDUR:
            model.eval()
            d_kayip = d_mesafe = 0.0
            with torch.no_grad():
                for kareler, tforms, _, _ in dogrulama_yukleyici:
                    kareler = kareler.to(aygit); tforms = tforms.to(aygit)
                    etiket = etiket_donusturucu(tforms, torch.linalg.inv(tforms))
                    with torch.cuda.amp.autocast(enabled=a.amp):
                        tahmin = tahmin_donusturucu(model(kareler / 255))
                    d_kayip += kayip_fn(tahmin.float(), etiket).item()
                    d_mesafe += mesafe_fn(noktaya(tahmin.float()),
                                          noktaya(etiket)).mean().item()
            n = len(dogrulama_yukleyici)
            d_kayip, d_mesafe = d_kayip / n, d_mesafe / n
            dogrulama = {"dogrulama_kayip": d_kayip, "dogrulama_mesafe": d_mesafe}

            if d_mesafe < en_iyi:
                en_iyi = d_mesafe
                torch.save(model.state_dict(), a.kayit / "en_iyi_model.pt")
            if planlayici is not None:
                planlayici.step(d_mesafe)
            simdiki_lr = optimizer.param_groups[0]["lr"]
            dogrulama["lr"] = simdiki_lr
            gecen = (time.time() - bas_zaman) / 60
            print(f"epok {epok:>6}  egitim {e_kayip:.4f} / {e_mesafe:.3f} mm   "
                  f"dogrulama {d_kayip:.4f} / {d_mesafe:.3f} mm"
                  f"{'  <- en iyi' if d_mesafe == en_iyi else ''}"
                  + (f"   lr {simdiki_lr:.2e}" if a.lr_plan != "sabit" else "")
                  + f"   [{gecen:.0f} dk]", flush=True)

        olcum_dosyasi(olcum, {"epok": epok, "egitim_kayip": e_kayip,
                              "egitim_mesafe": e_mesafe, **dogrulama})

        if epok % a.kayit_sikligi == 0 or epok == a.epok - 1 or _DURDUR:
            torch.save({"epok": epok, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "olcekleyici": olcekleyici.state_dict(),
                        "planlayici": (planlayici.state_dict()
                                       if planlayici is not None else None),
                        "en_iyi": en_iyi}, ckpt_yolu)

        # ARA ANLIK GORUNTU — uzun kosumun asil ciktisi
        #
        # Faz 4'un dort deneyi de tek bir soruyu cevapsiz birakti: cokus
        # BUTCEYE mi bagli? Bunu tek bir bitmis modele bakarak soylemek
        # imkansiz; gereken sey bagdasimin egitim olcegiyle nasil degistigi.
        # Bu yuzden uzun kosumda belirli epoklarda agirliklar ayri dosyalara
        # yaziliyor ve sonradan her biri icin r olculuyor. Duz bir cizgi
        # "cokus butceye bagli degil" der, yukselen bir egri "bagli ve su
        # hizla" der. Ikisi de yazilabilir bir sonuc; hangisi oldugunu
        # bilmemek en kotusu.
        if a.ara_kayit and epok % a.ara_kayit == 0:
            torch.save(model.state_dict(), a.kayit / f"ara_epok{epok:06d}.pt")
        if _DURDUR:
            print(f"durduruldu (epok {epok}). Devam: python src/egit.py --devam")
            break

    print(f"\nbitti. en iyi dogrulama mesafesi {en_iyi:.4f} mm")
    print(f"model: {a.kayit / 'en_iyi_model.pt'}")
    print(f"olcumler: {olcum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
