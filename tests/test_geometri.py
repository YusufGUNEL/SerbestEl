"""src/geometri.py dogrulamasi — Faz 1'in 'sessiz hata tuzagi' adimi.

Bu problemde matris carpim sirasini ya da bir tersi yanlis yazmak HATA VERMEZ,
sessizce yanlis sonuc verir. O yuzden iki bagimsiz kontrol yapiyoruz:

  A) IC TUTARLILIK — gidis-donus, ortogonallik, zincir kurallari.
     Referans olmasa bile tutmasi gereken ozellikler.
  B) REFERANSA KARSI — ayni girdiyle referans depo ne uretiyorsa
     bizim uygulamamiz da ayni sayiyi uretmeli. Sira/ters hatalarini
     yakalayan asil kontrol bu.

Kosum:  python tests/test_geometri.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "reference" / "TUS-REC2025-Challenge_baseline"))

from src.geometri import (  # noqa: E402
    Kalibrasyon,
    isaret_ddf,
    kuresel_donusumler,
    noktalari_tasi,
    piksel_noktalari,
    yer_degistirme,
    yerel_donusumler,
)

VERI = KOK / "data"
CALIB = VERI / "calib_matrix.csv"

sonuclar: list[tuple[str, bool, str]] = []


def kontrol(ad, fn):
    try:
        gecti, detay = fn()
    except Exception as exc:  # noqa: BLE001
        gecti, detay = False, f"{type(exc).__name__}: {exc}"
    sonuclar.append((ad, gecti, detay))
    print(f"[{'OK ' if gecti else 'HATA'}] {ad}: {detay}")


def ornek_tforms(n=40, tohum=0):
    """Gercek veriden tforms al; veri yoksa sentetik rijit donusum uret."""
    dosya = VERI / "transfs" / "050" / "LH_rotation.h5"
    if dosya.exists():
        import h5py

        with h5py.File(dosya, "r") as f:
            return torch.tensor(np.asarray(f["tforms"][:n])), "gercek veri"
    # sentetik yedek: rastgele ama gecerli rijit donusumler
    import pytorch3d.transforms as t3

    g = torch.Generator().manual_seed(tohum)
    aci = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([2.0, 1.0, 2.0])
    R = t3.euler_angles_to_matrix(aci, "ZYX")
    t = (torch.rand(n, 3, generator=g) - 0.5) * 100
    T = torch.eye(4).repeat(n, 1, 1)
    T[:, 0:3, 0:3], T[:, 0:3, 3] = R, t
    return T, "sentetik"


# ======================= A) IC TUTARLILIK =============================

def a1_kalibrasyon_okundu():
    k = Kalibrasyon.csvden(CALIB)
    # olcek kosegen olmali, z ve homojen 1
    kosegen_disi = (k.olcek - torch.diag(torch.diagonal(k.olcek))).abs().max().item()
    sx, sy = k.olcek[0, 0].item(), k.olcek[1, 1].item()
    # rijit matrisin donme blogu ortogonal olmali
    R = k.rijit[0:3, 0:3]
    ort = (R @ R.T - torch.eye(3)).abs().max().item()
    det = torch.linalg.det(R).item()
    ok = kosegen_disi < 1e-6 and ort < 1e-5 and abs(det - 1) < 1e-5 and 0.1 < sx < 1
    return ok, (f"olcek x={sx:.6f} y={sy:.6f} mm/px, kosegen disi {kosegen_disi:.1e}, "
                f"rijit ortogonallik {ort:.1e}, det={det:.6f}")


def a2_izgara_sirasi():
    """Duzlestirme sirasi satir-once olmali; DDF'lerin 307200 duzeni buna bagli."""
    p = piksel_noktalari(4, 5)
    if p.shape != (4, 20):
        return False, f"bicim {tuple(p.shape)}, (4,20) olmali"
    # indeks 0 -> (x=1,y=1); indeks 1 -> (x=2,y=1); indeks 5 -> (x=1,y=2)
    bekle = {0: (1, 1), 1: (2, 1), 4: (5, 1), 5: (1, 2), 19: (5, 4)}
    for i, (bx, by) in bekle.items():
        if (p[0, i].item(), p[1, i].item()) != (bx, by):
            return False, f"indeks {i}: ({p[0,i]},{p[1,i]}), beklenen ({bx},{by})"
    z_sifir = p[2].abs().max().item() == 0
    hepsi_bir = (p[3] == 1).all().item()
    # tam cozunurlukte DDF uzunlugu 307200 olmali
    n = piksel_noktalari(480, 640).shape[1]
    return (z_sifir and hepsi_bir and n == 307200,
            f"satir-once sira dogru, z=0, homojen=1, tam izgara {n} nokta")


def a3_gidis_donus():
    """Kareyi ileri gotur, tersiyle geri getir: baslangica donmeli.

    Yol haritasinin onerdigi kontrol tam bu. Carpim sirasi ya da bir ters
    yanlissa bu test kirilir.
    """
    tforms, kaynak = ornek_tforms()
    k = Kalibrasyon.csvden(CALIB)
    kur = kuresel_donusumler(tforms, k)          # kare i -> kare 0
    p = piksel_noktalari(480, 640, yogunluk=(7, 9))
    mm = k.olcek @ p

    ileri = noktalari_tasi(kur, p, k)             # (K,3,N) kare 0 uzayinda
    # geri: ayni donusumun tersiyle, homojenlestirip don
    ileri_h = torch.cat([ileri, torch.ones(ileri.shape[0], 1, ileri.shape[2])], dim=1)
    geri = (torch.linalg.inv(kur) @ ileri_h)[:, 0:3, :]
    hata = (geri - mm[None, 0:3, :]).abs().max().item()
    return hata < 1e-3, f"{kaynak}, en buyuk gidis-donus hatasi {hata:.2e} mm"


def a4_ortogonallik():
    """Donusumler ortogonal kalmali — yoksa Euler acisina cevrilemez."""
    tforms, kaynak = ornek_tforms()
    k = Kalibrasyon.csvden(CALIB)
    en_kotu = 0.0
    for ad, T in (("kuresel", kuresel_donusumler(tforms, k)),
                  ("yerel", yerel_donusumler(tforms, k))):
        R = T[:, 0:3, 0:3]
        h = (R @ R.transpose(-1, -2) - torch.eye(3)).abs().max().item()
        en_kotu = max(en_kotu, h)
        alt = T[:, 3, :] - torch.tensor([0.0, 0.0, 0.0, 1.0])
        if alt.abs().max().item() > 1e-5:
            return False, f"{ad}: alt satir [0,0,0,1] degil"
    return en_kotu < 1e-4, f"{kaynak}, en buyuk ortogonallik sapmasi {en_kotu:.2e}"


def a5_zincir_kurali():
    """Yerel donusumlerin carpimi kuresel donusumu vermeli.

    T_{0<-i} = T_{0<-1} . T_{1<-2} ... T_{i-1<-i}
    Suruklenmenin matematiksel sebebi tam bu zincir.
    """
    tforms, kaynak = ornek_tforms(n=30)
    k = Kalibrasyon.csvden(CALIB)
    kur = kuresel_donusumler(tforms, k)
    yer = yerel_donusumler(tforms, k)

    birikim = yer[0].clone()
    en_kotu = (birikim - kur[0]).abs().max().item()
    for i in range(1, yer.shape[0]):
        birikim = birikim @ yer[i]
        en_kotu = max(en_kotu, (birikim - kur[i]).abs().max().item())
    return en_kotu < 1e-2, f"{kaynak}, {yer.shape[0]} adim, en buyuk sapma {en_kotu:.2e}"


def a6_ilk_kare_sifir():
    """Ilk kareden ilk kareye yer degistirme sifir olmali (kimlik kontrolu).

    ESIK NEDEN 1e-3 VE DAHA SIKI DEGIL
      Veri float32 saklaniyor ve referans bastan sona float32 calisiyor.
      T_(0<-0) = R^-1 . (tforms[0]^-1 . tforms[0]) . R  matematikte tam kimlik,
      ama float32'de ~7e-5 sapiyor; oteleme terimleri 80-500 mm oldugu icin
      matris tersi bu buyuklukte gurultu uretiyor. Ayni hesap float64'te
      2.2e-13 veriyor — yani sapma uygulamadan degil, veri tipinden geliyor.
      Koordinatlar 147 mm'ye kadar cikiyor; 1e-4 mm bagil olarak 1e-6,
      float32 epsilonu mertebesinde. Daha siki esik uygulamayi degil
      float32'yi test etmis olurdu.
    """
    tforms, _ = ornek_tforms(n=5)
    k = Kalibrasyon.csvden(CALIB)
    from src.geometri import goruntu_donusumleri

    T = goruntu_donusumleri(tforms, k, torch.tensor([0]), torch.tensor([0]))
    p = piksel_noktalari(480, 640, yogunluk=(5, 5))
    d32 = yer_degistirme(T, p, k).abs().max().item()

    # ayni hesabi float64'te yap: sapma kaybolmali. Kaybolmuyorsa bu bir
    # yuvarlama sorunu degil, gercek uygulama hatasidir.
    k64 = Kalibrasyon(k.olcek.double(), k.rijit.double())
    T64 = goruntu_donusumleri(tforms.double(), k64, torch.tensor([0]), torch.tensor([0]))
    d64 = yer_degistirme(T64, p.double(), k64).abs().max().item()

    ok = d32 < 1e-3 and d64 < 1e-9
    return ok, f"float32 {d32:.2e} mm, float64 {d64:.2e} mm (sapma veri tipinden)"


def a7_sayisal_gurultu_birikmiyor():
    """Gercek referansin kendi sayisal gurultusu kare sayisiyla BUYUMEMELI.

    Bu, projenin merkezindeki ayrimi kilitliyor: kuresel donusum tforms[0] ve
    tforms[i]'den DOGRUDAN hesaplaniyor, zincirleme degil. Bu yuzden gercek
    referansta suruklenme YOK. Suruklenme yalnizca MODELIN kare basina hatasi
    zincirlendiginde dogar. Bu test bozulursa gercek referans hesabinda
    istemeden zincirlemeye gecmis oluruz.
    """
    tforms, kaynak = ornek_tforms(n=1570)
    if tforms.shape[0] < 500:
        return False, f"yeterli kare yok ({tforms.shape[0]}), gercek veri gerekli"
    k = Kalibrasyon.csvden(CALIB)
    k64 = Kalibrasyon(k.olcek.double(), k.rijit.double())
    p = piksel_noktalari(480, 640, yogunluk=(9, 9))

    pt32 = noktalari_tasi(kuresel_donusumler(tforms, k), p, k)
    pt64 = noktalari_tasi(kuresel_donusumler(tforms.double(), k64), p.double(), k64)
    sapma = (pt32.double() - pt64).norm(dim=1).mean(dim=1)   # kare basina, mm

    bas, son = sapma[9].item(), sapma[-1].item()
    buyume = son / max(bas, 1e-12)
    # gercek hareketle kiyas: gurultu sinyalin yaninda gorunmez olmali
    mm = (k64.olcek @ p.double())[0:3]
    gercek = (pt64[-1] - mm).norm(dim=0).mean().item()

    ok = buyume < 3 and son < 1e-2 and son / gercek < 1e-4
    return ok, (f"{kaynak}, {tforms.shape[0]} kare: kare 10'da {bas:.1e} mm, "
                f"son karede {son:.1e} mm ({buyume:.1f} kat). "
                f"Gercek hareket {gercek:.1f} mm, oran 1:{gercek/son:.0f}")


# ==================== B) REFERANSA KARSI ==============================

def b1_donusumler_referansla_ayni():
    """Referansin generate_ddf_from_label'i ile ayni donusumleri uretiyor muyuz."""
    from utils.generate_ddf_from_label import generate_ddf_from_label

    tforms, kaynak = ornek_tforms(n=25)
    k = Kalibrasyon.csvden(CALIB)

    ref = generate_ddf_from_label(str(CALIB), "cpu")
    r_kur, r_yer = ref.get_global_local_transformations(
        tforms[None], torch.linalg.inv(tforms[None])
    )
    b_kur = kuresel_donusumler(tforms, k)
    b_yer = yerel_donusumler(tforms, k)

    if r_kur.shape != b_kur.shape or r_yer.shape != b_yer.shape:
        return False, f"bicim farkli: ref {tuple(r_kur.shape)} biz {tuple(b_kur.shape)}"
    hk = (r_kur - b_kur).abs().max().item()
    hy = (r_yer - b_yer).abs().max().item()
    return max(hk, hy) < 1e-3, f"{kaynak}, kuresel fark {hk:.2e}, yerel fark {hy:.2e}"


def b2_kalibrasyon_referansla_ayni():
    from utils.plot_functions import read_calib_matrices

    r_olcek, r_rijit, r_tam = read_calib_matrices(str(CALIB))
    k = Kalibrasyon.csvden(CALIB)
    h = max((r_olcek - k.olcek).abs().max().item(),
            (r_rijit - k.rijit).abs().max().item(),
            (r_tam - k.tam).abs().max().item())
    return h < 1e-6, f"en buyuk fark {h:.2e} (olcek, rijit ve bilesim)"


def b3_izgara_referansla_ayni():
    from utils.plot_functions import reference_image_points

    r = reference_image_points([480, 640], [480, 640])
    b = piksel_noktalari(480, 640)
    if r.shape != b.shape:
        return False, f"bicim: ref {tuple(r.shape)} biz {tuple(b.shape)}"
    h = (r - b).abs().max().item()
    return h < 1e-4, f"{b.shape[1]} nokta, en buyuk fark {h:.2e}"


def b4_ddf_referansla_ayni():
    """Dort DDF'nin tamami: referansin urettigi sayilarla birebir ayni mi."""
    import h5py
    from utils.generate_ddf_from_label import generate_ddf_from_label

    kf = VERI / "frames" / "050" / "LH_rotation.h5"
    kt = VERI / "transfs" / "050" / "LH_rotation.h5"
    kl = VERI / "landmarks" / "landmark_050.h5"
    if not (kf.exists() and kt.exists() and kl.exists()):
        return False, "gercek veri yok, bu kontrol atlanamaz"

    n = 30  # tum tarama 1570 kare; 30 kare bellek icin yeterli kanit
    with h5py.File(kt, "r") as f:
        tforms = torch.tensor(np.asarray(f["tforms"][:n]))
    with h5py.File(kf, "r") as f:
        kareler = np.asarray(f["frames"][:n])
    with h5py.File(kl, "r") as f:
        isaretler = np.asarray(f["LH_rotation"])
    # isaret kare indeksleri bu dilime sigmali
    isaretler = isaretler[(isaretler[:, 0] >= 1) & (isaretler[:, 0] < n)]
    if len(isaretler) == 0:
        isaretler = np.array([[n // 2, 300, 200]], dtype=np.int64)

    ref = generate_ddf_from_label(str(CALIB), "cpu")
    r_gp, r_gl, r_lp, r_ll = ref.calculate_GT_DDF(kareler, tforms.numpy(), isaretler)

    k = Kalibrasyon.csvden(CALIB)
    p = piksel_noktalari(480, 640)
    b_gp = yer_degistirme(kuresel_donusumler(tforms, k), p, k)
    b_lp = yer_degistirme(yerel_donusumler(tforms, k), p, k)
    isaret_t = torch.from_numpy(isaretler)
    b_gl = isaret_ddf(b_gp, isaret_t)
    b_ll = isaret_ddf(b_lp, isaret_t)

    farklar = {
        "GP": np.abs(r_gp - b_gp.numpy()).max(),
        "GL": np.abs(r_gl - b_gl.numpy()).max(),
        "LP": np.abs(r_lp - b_lp.numpy()).max(),
        "LL": np.abs(r_ll - b_ll.numpy()).max(),
    }
    en_kotu = max(farklar.values())
    detay = ", ".join(f"{a}={v:.2e}" for a, v in farklar.items())
    return en_kotu < 1e-3, f"{len(isaretler)} isaret, {detay}"


for ad, fn in [
    ("A1 kalibrasyon okundu ve gecerli", a1_kalibrasyon_okundu),
    ("A2 izgara sirasi satir-once", a2_izgara_sirasi),
    ("A3 gidis-donus baslangica donuyor", a3_gidis_donus),
    ("A4 donusumler ortogonal", a4_ortogonallik),
    ("A5 yerel zincir = kuresel", a5_zincir_kurali),
    ("A6 ilk kare yer degistirmesi sifir", a6_ilk_kare_sifir),
    ("A7 sayisal gurultu birikmiyor", a7_sayisal_gurultu_birikmiyor),
    ("B1 donusumler referansla ayni", b1_donusumler_referansla_ayni),
    ("B2 kalibrasyon referansla ayni", b2_kalibrasyon_referansla_ayni),
    ("B3 izgara referansla ayni", b3_izgara_referansla_ayni),
    ("B4 dort DDF referansla ayni", b4_ddf_referansla_ayni),
]:
    kontrol(ad, fn)

gecen = sum(1 for _, g, _ in sonuclar if g)
print(f"\n{gecen}/{len(sonuclar)} kontrol gecti")
sys.exit(0 if gecen == len(sonuclar) else 1)
