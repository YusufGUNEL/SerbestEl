"""Hacmin GERCEKTEN dogru kuruldugunu dogrular — Faz 1 kontrol adimi.

"Kod hata vermedi" ile "geometri dogru" ayni sey degil. Yanlis bir carpim
sirasi da hatasiz kosar, sadece hacmi bozar. Burada olculebilir bir kanit
ariyoruz.

YONTEM — capraz dogrulama
  Bir kareyi hacimden CIKAR, hacmi geri kalan karelerle kur, sonra o karenin
  piksel konumlarindan hacmi ORNEKLE ve gercek kareyle karsilastir.
  Geometri dogruysa okunan degerler gercek kareye benzemeli; cunku ayni fiziksel
  dokuyu komsu kareler de gormus.

KONTROL GRUBU NEDEN SART
  Yalnizca "benzerlik yuksek cikti" demek yetmez: ultrason goruntuleri puruzsuz
  oldugu icin rastgele iki kare bile bir miktar benzesir. Bu yuzden ayni
  olcumu KASITLI BOZULMUS donusumle de yapiyoruz (birkac mm kaydirma ve
  birkac derece dondurme). Dogru donusum bozulmustan belirgin iyi degilse
  test hicbir sey kanitlamaz.

Kosum:  python tests/test_hacim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))

from src.geometri import Kalibrasyon, kuresel_donusumler, piksel_noktalari  # noqa: E402
from src.hacim import hacim_kur, tarama_oku  # noqa: E402

VERI = KOK / "data"
sonuclar: list[tuple[str, bool, str]] = []


def kontrol(ad, fn):
    try:
        gecti, detay = fn()
    except Exception as exc:  # noqa: BLE001
        gecti, detay = False, f"{type(exc).__name__}: {exc}"
    sonuclar.append((ad, gecti, detay))
    print(f"[{'OK ' if gecti else 'HATA'}] {ad}: {detay}")


def hacimden_ornekle(hacim, donusum, kalib, h, w):
    """Bir karenin piksel konumlarindan hacmi okur. Cikti (H*W,) ve gecerli maske."""
    p = piksel_noktalari(h, w)
    mm = (donusum @ (kalib.olcek @ p))[0:3].numpy()              # (3, H*W)
    idx = ((mm - hacim.kok_mm[:, None]) / hacim.voksel_mm).astype(np.int64)
    nz, ny, nx = hacim.ortalama.shape
    gecerli = (
        (idx[0] >= 0) & (idx[0] < nx) & (idx[1] >= 0) & (idx[1] < ny)
        & (idx[2] >= 0) & (idx[2] < nz)
    )
    okunan = np.full(idx.shape[1], np.nan, dtype=np.float32)
    okunan[gecerli] = hacim.ortalama[idx[2][gecerli], idx[1][gecerli], idx[0][gecerli]]
    return okunan, gecerli & ~np.isnan(okunan)


def bozuk_donusum(T, kaydirma_mm=4.0, aci_derece=4.0):
    """Kasitli olarak yanlis bir donusum uretir — kontrol grubu icin."""
    import pytorch3d.transforms as t3

    a = np.deg2rad(aci_derece)
    R = t3.euler_angles_to_matrix(torch.tensor([a, a, a], dtype=T.dtype), "ZYX")
    B = torch.eye(4, dtype=T.dtype)
    B[0:3, 0:3] = R
    B[0:3, 3] = kaydirma_mm
    return B @ T


def _hazirla(kare_siniri=600, atla=2, voksel=0.6):
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    kareler, tforms = tarama_oku(VERI, "050", "LH_rotation", kare_siniri)
    donusumler = kuresel_donusumler(tforms, kalib, ilk_kare_dahil=True)

    # disarida birakilacak kareler: taramanin farkli yerlerinden
    n = kareler.shape[0]
    disarida = [n // 4, n // 2, 3 * n // 4]
    tut = np.ones(n, dtype=bool)
    for d in disarida:
        tut[max(d - 2, 0):d + 3] = False   # komsulari da cikar, sizinti olmasin

    hacim = hacim_kur(
        kareler[tut], None, kalib, voksel_mm=voksel, kare_atla=atla,
        blok=32, ilerleme=False, donusumler=donusumler[tut],
    )
    return kalib, kareler, donusumler, hacim, disarida


def t1_capraz_dogrulama():
    """Disarida birakilan kareler hacimden dogru okunuyor mu — ve bozuk
    donusumden belirgin iyi mi."""
    kalib, kareler, donusumler, hacim, disarida = _hazirla()
    h, w = kareler.shape[1:]

    dogru_r, bozuk_r, kapsama = [], [], []
    for d in disarida:
        gercek = kareler[d].reshape(-1).astype(np.float32)

        okunan, maske = hacimden_ornekle(hacim, donusumler[d], kalib, h, w)
        if maske.sum() < 1000:
            return False, f"kare {d}: hacim kapsamasi yok ({maske.sum()} piksel)"
        dogru_r.append(np.corrcoef(gercek[maske], okunan[maske])[0, 1])
        kapsama.append(maske.mean())

        b_okunan, b_maske = hacimden_ornekle(
            hacim, bozuk_donusum(donusumler[d]), kalib, h, w)
        ortak = maske & b_maske
        bozuk_r.append(np.corrcoef(gercek[ortak], b_okunan[ortak])[0, 1])

    d_ort, b_ort = float(np.mean(dogru_r)), float(np.mean(bozuk_r))
    ok = d_ort > 0.75 and d_ort > b_ort + 0.15
    return ok, (f"{len(disarida)} kare: dogru donusum r={d_ort:.3f}, "
                f"bozuk donusum r={b_ort:.3f}, fark {d_ort - b_ort:+.3f}, "
                f"kapsama {np.mean(kapsama):.0%}")


def t2_ilk_kare_kendi_yerinde():
    """Ilk kare kimlik donusumuyle gidiyor: hacimde kendi mm yerinde olmali."""
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    kareler, tforms = tarama_oku(VERI, "050", "LH_rotation", 3)
    # yalnizca ilk kareyle hacim kur: okuma neredeyse birebir olmali
    T = torch.eye(4).unsqueeze(0)
    hacim = hacim_kur(kareler[:1], None, kalib, voksel_mm=0.25,
                      ilerleme=False, donusumler=T)
    h, w = kareler.shape[1:]
    okunan, maske = hacimden_ornekle(hacim, T[0], kalib, h, w)
    gercek = kareler[0].reshape(-1).astype(np.float32)
    hata = np.abs(gercek[maske] - okunan[maske]).mean()
    return (maske.mean() > 0.99 and hata < 1.0,
            f"kapsama {maske.mean():.1%}, ortalama parlaklik hatasi {hata:.3f}/255")


def t3_hacim_saglikli():
    """Hacim mm olceginde makul mu — birim ya da eksen hatasi yakalar."""
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    kareler, tforms = tarama_oku(VERI, "050", "LH_rotation")
    hacim = hacim_kur(kareler, tforms, kalib, voksel_mm=1.0,
                      kare_atla=4, ilerleme=False)
    nz, ny, nx = hacim.ortalama.shape
    boy_mm = np.array([nx, ny, nz]) * hacim.voksel_mm

    # kare kendisi 640x0.2294 = 146.8 mm genis, 480x0.2210 = 106.1 mm derin.
    # doner tarama bunu ustune hareket ekliyor; her eksen bu mertebede olmali.
    makul = bool(np.all(boy_mm > 50) and np.all(boy_mm < 600))
    dolu = hacim.sayim > 0
    # ortalama parlaklik gecerli aralikta olmali
    deger = hacim.ortalama[dolu]
    aralik_ok = bool(deger.min() >= 0 and deger.max() <= 255)
    return (makul and aralik_ok and 0.05 < hacim.doluluk < 0.95,
            f"{nx}x{ny}x{nz} voksel = {boy_mm[0]:.0f}x{boy_mm[1]:.0f}x{boy_mm[2]:.0f} mm, "
            f"doluluk {hacim.doluluk:.1%}, parlaklik {deger.min():.0f}-{deger.max():.0f}")


for ad, fn in [
    ("T1 capraz dogrulama (kontrol gruplu)", t1_capraz_dogrulama),
    ("T2 ilk kare kendi yerinde", t2_ilk_kare_kendi_yerinde),
    ("T3 hacim mm olceginde makul", t3_hacim_saglikli),
]:
    kontrol(ad, fn)

gecen = sum(1 for _, g, _ in sonuclar if g)
print(f"\n{gecen}/{len(sonuclar)} kontrol gecti")
sys.exit(0 if gecen == len(sonuclar) else 1)
