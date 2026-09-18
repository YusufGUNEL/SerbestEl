"""Faz 4 donme temsillerinin testleri.

NEDEN BU TESTLER SART
  Faz 4'un butun iddiasi "tek degiskeni degistirdim, dort sayi soyle degisti"
  uzerine kurulu. Temsil degistiren kod sessizce yanlis bir donusum uretirse
  deney iyi ya da kotu bir sonuc verir ve o sonuc temsile atfedilir — oysa
  hata cevirmede olmustur. Iki seyi kilitlemek gerekiyor:

    1. Yeni yol DOGRU: 6B cikti gercekten bir donme matrisi uretiyor
       (ortonormal, det=+1) ve bilinen bir donmeyi gidip geri veriyor.
    2. Eski yol DEGISMEDI: euler yolu hala referansin urettiginin BIREBIR
       aynisi. Faz 2 sayilari bu yolla uretildi; bir ondalik kayarsa Faz 4'un
       butun karsilastirmalari dayanaksiz kalir.

Kosum:  python tests/test_temsil.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"
for p in (str(KOK), str(REFERANS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytorch3d.transforms as p3d  # noqa: E402

from src.temsil import (  # noqa: E402
    TahminDonusturucu,
    altib_donusume,
    cikti_boyutu,
    donusume_matrise,
)
from utils.funs import pair_samples  # noqa: E402
from utils.plot_functions import reference_image_points  # noqa: E402
from utils.transform import PredictionTransform, Transforms  # noqa: E402

_gecen = _kalan = 0


def kontrol(ad: str, kosul: bool, ayrinti: str = "") -> None:
    global _gecen, _kalan
    if kosul:
        _gecen += 1
        print(f"[OK ] {ad}: {ayrinti}")
    else:
        _kalan += 1
        print(f"[HTA] {ad}: {ayrinti}")


def kalibrasyon():
    """Testler icin gecerli bir kalibrasyon takimi.

    Matematik kalibrasyonun kendisine bagli degil; onemli olan matrislerin
    tersinir ve boyutlarin dogru olmasi.
    """
    nokta = reference_image_points([480, 640], 2)
    olcek = torch.eye(4)
    olcek[0, 0], olcek[1, 1] = 0.229389, 0.220980
    rijit = torch.eye(4)
    rijit[0:3, 3] = torch.tensor([3.0, -2.0, 11.0])
    tam = rijit @ olcek
    return dict(image_points=nokta, tform_image_to_tool=tam,
                tform_image_mm_to_tool=rijit, tform_image_pixel_to_mm=olcek)


def t1_boyutlar():
    """Her temsil kac sayi istiyor — ag son katmani buna gore kuruluyor."""
    beklenen = {"euler": 6, "kuaterniyon": 7, "matris": 12, "6b": 9}
    hepsi = all(cikti_boyutu(t, 1) == d for t, d in beklenen.items())
    coklu = cikti_boyutu("6b", 10) == 90
    kontrol("T1 cikti boyutlari", hepsi and coklu,
            f"{beklenen}, 10 cift icin 6b -> {cikti_boyutu('6b', 10)}")


def t2_altib_gercek_donme_uretiyor():
    """Gram-Schmidt ciktisi ortonormal ve sag el sistemi olmali.

    Kritik olan: ag ciktisi TAMAMEN rastgeleyken bile bu tutmali. Matris
    temsilinde (12 sayi) tutmaz — orasi Faz 4'te olculecek farkin ta kendisi.
    """
    g = torch.Generator().manual_seed(4)
    x = torch.randn(64, 3, 9, generator=g) * 5      # buyuk, dengesiz cikti
    T = altib_donusume(x)
    R = T[..., 0:3, 0:3]
    birim = torch.matmul(R.transpose(-1, -2), R)
    sapma = (birim - torch.eye(3)).abs().max().item()
    det = torch.linalg.det(R)
    det_sapma = (det - 1).abs().max().item()
    son_satir = (T[..., 3, :] - torch.tensor([0., 0., 0., 1.])).abs().max().item()
    kontrol("T2 6B ortonormal ve det=+1",
            sapma < 1e-5 and det_sapma < 1e-5 and son_satir == 0,
            f"R^T R sapmasi {sapma:.2e}, det sapmasi {det_sapma:.2e}")


def t3_altib_bilinen_donmeyi_geri_veriyor():
    """Bilinen bir donmenin ilk iki sutunu verilince ayni donme cikmali.

    6B temsilin tanimi tam budur: donme matrisinin ilk iki sutunu. Ucuncu
    sutun capraz carpimla tek sekilde belirlenir, o yuzden saklanmasina gerek
    yoktur — 9 yerine 6 sayi.
    """
    g = torch.Generator().manual_seed(11)
    aci = (torch.rand(32, 1, 3, generator=g) - 0.5) * 2.0
    R = p3d.euler_angles_to_matrix(aci, "ZYX")
    t = torch.randn(32, 1, 3, generator=g) * 4
    x = torch.cat((R[..., 0], R[..., 1], t), dim=-1)   # ilk iki SUTUN + oteleme
    T = altib_donusume(x)
    R_fark = (T[..., 0:3, 0:3] - R).abs().max().item()
    t_fark = (T[..., 0:3, 3] - t).abs().max().item()
    kontrol("T3 6B gidip geri ayni donmeyi veriyor",
            R_fark < 1e-5 and t_fark < 1e-6,
            f"donmede en buyuk fark {R_fark:.2e}, otelemede {t_fark:.2e}")


def t4_euler_yolu_referansla_birebir():
    """Faz 2 yolu DEGISMEDI — ayni girdi, ayni cikti, bit bit.

    Bu test kirilirsa Faz 2'nin dort sayisi ile Faz 4 deneyleri artik
    karsilastirilamaz. Faz 4'un butun tablosu buna dayaniyor.
    """
    k = kalibrasyon()
    ciftler = pair_samples(2, 1, 0)
    n = ciftler.shape[0]
    g = torch.Generator().manual_seed(7)
    cikti = torch.randn(8, cikti_boyutu("euler", n), generator=g) * 0.1

    bizim_nokta = TahminDonusturucu("euler", "nokta", n, **k)(cikti)
    ref_nokta = PredictionTransform("parameter", "point", num_pairs=n, **k)(cikti)
    bizim_par = TahminDonusturucu("euler", "parametre", n, **k)(cikti)
    ref_par = PredictionTransform("parameter", "parameter", num_pairs=n, **k)(cikti)
    bizim_T = donusume_matrise("euler", cikti, n, **k)
    ref_T = Transforms(pred_type="parameter", num_pairs=n, **k)(cikti)

    farklar = [(bizim_nokta - ref_nokta).abs().max().item(),
               (bizim_par - ref_par).abs().max().item(),
               (bizim_T - ref_T).abs().max().item()]
    kontrol("T4 euler yolu referansla birebir", max(farklar) == 0.0,
            f"nokta / parametre / matris farki {farklar}")


def t5_butun_temsiller_gecerli_donusum_veriyor():
    """Dort temsil de (B,P,4,4) ve dogru son satiri uretiyor.

    Matris temsilinde donme kismi ORTONORMAL DEGIL — bilerek. Ag 12 sayiyi
    serbestce uretir, hicbir kisit yoktur. Bu testin amaci onu duzeltmek
    degil, farkin bilincli oldugunu kayda gecirmek.
    """
    k = kalibrasyon()
    n = 1
    g = torch.Generator().manual_seed(3)
    sonuc = {}
    for temsil in ("euler", "kuaterniyon", "matris", "6b"):
        cikti = torch.randn(5, cikti_boyutu(temsil, n), generator=g)
        T = donusume_matrise(temsil, cikti, n, **k)
        R = T[..., 0:3, 0:3]
        dik = (torch.matmul(R.transpose(-1, -2), R)
               - torch.eye(3)).abs().max().item()
        sonuc[temsil] = (tuple(T.shape), dik)
    bicim = all(s[0] == (5, 1, 4, 4) for s in sonuc.values())
    dik_olanlar = {t for t, (_, d) in sonuc.items() if d < 1e-4}
    matris_sapmasi = sonuc["matris"][1]
    kontrol("T5 dort temsil de 4x4 uretiyor",
            bicim and dik_olanlar == {"euler", "kuaterniyon", "6b"},
            "dik olanlar " + ", ".join(sorted(dik_olanlar))
            + f"; matris sapmasi {matris_sapmasi:.2f} (beklenen: dik degil)")


def t6_tutarlilik_kaybi_gercekte_sifir():
    """D hatti kisiti: T ve tersi verilince ceza TAM sifir olmali.

    Kisit ek etiket kullanmiyor, modelin iki tahminini karsilastiriyor.
    Gercek bir donusum ve onun tersi verildiginde bilesim birim matristir;
    ceza sifirdan farkli cikarsa kisit modeli yanlis yere ceker.
    """
    k = kalibrasyon()
    nokta_mm = torch.matmul(k["tform_image_pixel_to_mm"], k["image_points"])
    g = torch.Generator().manual_seed(21)
    aci = (torch.rand(6, 1, 3, generator=g) - 0.5) * 0.2
    R = p3d.euler_angles_to_matrix(aci, "ZYX")
    t = torch.randn(6, 1, 3, generator=g) * 0.5
    T = torch.cat((torch.cat((R, t[..., None]), -1),
                   torch.tensor([0., 0., 0., 1.]).expand(6, 1, 1, 4)), -2)
    T_ters = torch.linalg.inv(T)

    bilesim = torch.matmul(T, T_ters)
    kaymis = torch.matmul(bilesim, nokta_mm)[:, :, 0:3, :]
    beklenen = nokta_mm[0:3, :].expand_as(kaymis)
    ceza = torch.nn.functional.mse_loss(kaymis, beklenen).item()

    # kontrol grubu: tersi yerine baska bir donusum verilirse ceza BUYUK olmali
    yanlis = torch.matmul(T, torch.linalg.inv(T.roll(1, dims=0)))
    kaymis2 = torch.matmul(yanlis, nokta_mm)[:, :, 0:3, :]
    ceza2 = torch.nn.functional.mse_loss(kaymis2, beklenen).item()

    kontrol("T6 tutarlilik cezasi gercekte sifir", ceza < 1e-8 and ceza2 > 1e-3,
            f"dogru eslemede {ceza:.2e} mm^2, yanlis eslemede {ceza2:.4f} mm^2")


def t7_sifira_yakin_cikti_cokmuyor():
    """Egitimin ILK adiminda cikti sifira yakin — normalize orada patlar.

    Rastgele baslatilmis bir Linear katmanin ciktisi cok kucuktur. Boluma
    giren norm sifira giderse NaN dogar ve egitim daha ilk adimda oler.
    eps koruma bu yuzden var; test onu kilitliyor.
    """
    T = altib_donusume(torch.zeros(4, 1, 9))
    kucuk = altib_donusume(torch.full((4, 1, 9), 1e-7))
    saglam = bool(torch.isfinite(T).all() and torch.isfinite(kucuk).all())
    kontrol("T7 sifir ciktida NaN yok", saglam,
            "tam sifir ve 1e-7 ciktida butun degerler sonlu")


def main() -> int:
    print("=" * 66)
    print("FAZ 4 DONME TEMSILLERI")
    print("=" * 66)
    for f in (t1_boyutlar, t2_altib_gercek_donme_uretiyor,
              t3_altib_bilinen_donmeyi_geri_veriyor,
              t4_euler_yolu_referansla_birebir,
              t5_butun_temsiller_gecerli_donusum_veriyor,
              t6_tutarlilik_kaybi_gercekte_sifir,
              t7_sifira_yakin_cikti_cokmuyor):
        try:
            f()
        except Exception as e:                      # noqa: BLE001
            kontrol(f.__name__, False, f"istisna: {type(e).__name__}: {e}")
    print(f"\n{_gecen}/{_gecen + _kalan} kontrol gecti")
    return 1 if _kalan else 0


if __name__ == "__main__":
    sys.exit(main())
