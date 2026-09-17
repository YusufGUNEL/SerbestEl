"""Faz 3 analiz matematiginin testleri.

NEDEN BU TESTLER SART
  analiz.py'nin cikardigi sayilar projenin bilimsel iddiasi olacak: "hata su
  turden, su kadari su sebepten". Bir isaret hatasi ya da ters cevrilmis bir
  carpim bu iddiayi sessizce yanlis yapar — kod cokmez, grafik guzel cikar,
  sonuc yalan olur.

  Bu yuzden her olcum, cevabin ONCEDEN BILINDIGI kurulmus bir durumda
  sinaniyor: saf yanlilik verildiginde giderme tam sifir hata birakmali,
  bilinen bir olcek verildiginde uydurma o olcegi bulmali.

Kosum:  python tests/test_analiz.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from src.analiz import (  # noqa: E402
    alti_vektor,
    alti_vektordan,
    alti_vektordan_yigin,
    dogrusal_uydur,
    hata_donusumu,
    kare_basi_mesafe,
    olcegi_duzelt,
    sicrama_bul,
    yanliligi_gider,
)
from src.olcum import kuresel_biriktir  # noqa: E402

DER = 180.0 / math.pi
_gecen = _kalan = 0


def kontrol(ad: str, kosul: bool, ayrinti: str = "") -> None:
    global _gecen, _kalan
    if kosul:
        _gecen += 1
        print(f"[OK ] {ad}: {ayrinti}")
    else:
        _kalan += 1
        print(f"[HTA] {ad}: {ayrinti}")


def rastgele_rijit(k: int, tohum: int, olcek: float = 1.0) -> torch.Tensor:
    """k tane kucuk rijit donusum. Kare basina hareket gercekciligine yakin."""
    g = torch.Generator().manual_seed(tohum)
    v = torch.zeros(k, 6, dtype=torch.float64)
    v[:, 0:3] = (torch.rand(k, 3, generator=g, dtype=torch.float64) - 0.5) * olcek
    v[:, 3:6] = (torch.rand(k, 3, generator=g, dtype=torch.float64) - 0.5) * olcek
    return alti_vektordan_yigin(v)


def _mm_izgara(n: int = 64, tohum: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(tohum)
    p = torch.zeros(4, n)
    p[0:2] = (torch.rand(2, n, generator=g) - 0.5) * 100
    p[3] = 1.0
    return p


# ------------------------------------------------------------------ A1..A7

def a1_alti_vektor_gidis_donus() -> None:
    T = rastgele_rijit(200, tohum=1, olcek=2.0)
    geri = alti_vektordan_yigin(alti_vektor(T))
    fark = (T - geri).abs().max().item()
    kontrol("A1 alti vektor gidis-donus", fark < 1e-10,
            f"200 donusum, en buyuk fark {fark:.2e}")


def a2_tekil_ve_yiginli_ayni() -> None:
    V = alti_vektor(rastgele_rijit(50, tohum=2, olcek=3.0))
    yigin = alti_vektordan_yigin(V)
    tek = torch.stack([alti_vektordan(v) for v in V])
    fark = (yigin - tek).abs().max().item()
    kontrol("A2 yiginli surum tekille ayni", fark < 1e-12,
            f"50 donusum, en buyuk fark {fark:.2e}")


def a3_saf_yanlilik_tam_gideriliyor() -> None:
    """Hata her karede AYNI ise giderme gercegi tam geri vermeli.

    Bu testin gecmesi, yanlilik giderme deneyinin olctugu seyin gercekten
    'sabit yanlilik' oldugunu kanitliyor. Gecmezse deneyin sonucu yorumlanamaz.
    """
    gt_yerel = rastgele_rijit(300, tohum=3, olcek=1.5)
    E = alti_vektordan(torch.tensor([0.05, -0.02, 0.01, 0.03, -0.01, 0.02]))
    t_yerel = (gt_yerel @ E[None]).float()

    v = alti_vektor(hata_donusumu(gt_yerel, t_yerel))
    yanlilik = v.mean(dim=0)
    duzeltilmis = yanliligi_gider(t_yerel, yanlilik)

    mm = _mm_izgara()
    once = kare_basi_mesafe(kuresel_biriktir(gt_yerel.float()),
                            kuresel_biriktir(t_yerel), mm).mean()
    sonra = kare_basi_mesafe(kuresel_biriktir(gt_yerel.float()),
                             kuresel_biriktir(duzeltilmis), mm).mean()
    kontrol("A3 saf yanlilik tam gideriliyor", sonra < 1e-3 and once > 1.0,
            f"kuresel hata {once:.3f} mm -> {sonra:.2e} mm")


def a4_yansiz_gurultude_giderme_ise_yaramaz() -> None:
    """Kontrol grubu: hata yansiz gurultuyse yanlilik gidermenin isi yok.

    Bu test A3 kadar onemli. A3 tek basina 'giderme her seyi duzeltir'
    izlenimi verirdi; burada duzeltmemesi gerektigi durumda duzeltmedigi
    gosteriliyor. Ayirt edici olmayan bir olcum hicbir sey kanitlamaz.
    """
    g = torch.Generator().manual_seed(4)
    gt_yerel = rastgele_rijit(300, tohum=4, olcek=1.5)
    gurultu = torch.zeros(300, 6, dtype=torch.float64)
    gurultu[:, :] = (torch.rand(300, 6, generator=g, dtype=torch.float64) - 0.5) * 0.1
    t_yerel = (gt_yerel @ alti_vektordan_yigin(gurultu)).float()

    v = alti_vektor(hata_donusumu(gt_yerel, t_yerel))
    duzeltilmis = yanliligi_gider(t_yerel, v.mean(dim=0))

    mm = _mm_izgara()
    once = kare_basi_mesafe(kuresel_biriktir(gt_yerel.float()),
                            kuresel_biriktir(t_yerel), mm).mean()
    sonra = kare_basi_mesafe(kuresel_biriktir(gt_yerel.float()),
                             kuresel_biriktir(duzeltilmis), mm).mean()
    kazanc = 1 - sonra / once
    kontrol("A4 yansiz gurultude giderme ise yaramiyor", kazanc < 0.5,
            f"kuresel hata {once:.3f} -> {sonra:.3f} mm, kazanc %{100*kazanc:.0f} "
            f"(A3'te %100'e yakindi)")


def a5_olcek_bilinen_degeri_buluyor() -> None:
    g = torch.Generator().manual_seed(5)
    gercek = (torch.rand(4000, 6, generator=g, dtype=torch.float64) - 0.5) * 2
    a_ger = torch.tensor([0.70, 0.60, 0.15, 1.00, 0.05, 0.90], dtype=torch.float64)
    b_ger = torch.tensor([0.01, -0.02, 0.00, 0.03, 0.00, -0.01], dtype=torch.float64)
    gurultu = (torch.rand(4000, 6, generator=g, dtype=torch.float64) - 0.5) * 0.02
    tahmin = gercek * a_ger + b_ger + gurultu

    a, b, r = dogrusal_uydur(gercek, tahmin)
    da = (a - a_ger).abs().max().item()
    db = (b - b_ger).abs().max().item()
    kontrol("A5 olcek uydurmasi bilinen degeri buluyor", da < 0.01 and db < 0.01,
            f"a hatasi {da:.2e}, b hatasi {db:.2e}")


def a6_guvenilirlik_kapisi_calisiyor() -> None:
    """Bagdasimi dusuk bilesen DOKUNULMADAN birakilmali.

    Kapi olmadan a ~ 0 olan bilesende bolme gurultuyu buyutuyor: gercek
    kosumda GP 38 mm'den 128 mm'e cikmisti.
    """
    v = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    #          bilesen:   0     1      2      3      4      5
    a = torch.tensor([0.5, 0.02, 0.0001, -0.8, 9.0, 0.5], dtype=torch.float64)
    b = torch.zeros(6, dtype=torch.float64)
    r = torch.tensor([0.9, 0.04, 0.60, 0.90, 0.90, 0.9], dtype=torch.float64)

    cikti = olcegi_duzelt(v, a, b, r, r_esik=0.5)
    durum = {
        "0 gecerli (a=0,5 r=0,9) olceklendi":
            torch.allclose(cikti[:, 0], v[:, 0].double() / 0.5),
        "1 dusuk r dokunulmadi": torch.allclose(cikti[:, 1], v[:, 1].double()),
        # r kapisini geciyor ama a ~ 0: eski surumde tam burasi patliyordu
        "2 a~0 ama r yuksek dokunulmadi":
            torch.allclose(cikti[:, 2], v[:, 2].double()),
        "3 a negatif dokunulmadi": torch.allclose(cikti[:, 3], v[:, 3].double()),
        "4 a cok buyuk dokunulmadi": torch.allclose(cikti[:, 4], v[:, 4].double()),
    }
    kontrol("A6 guvenilirlik kapisi calisiyor", all(durum.values()),
            ", ".join(k for k, v_ in durum.items() if v_)
            + ("" if all(durum.values())
               else "  BASARISIZ: " + ", ".join(k for k, v_ in durum.items()
                                                if not v_)))


def a7_sicrama_bulunuyor() -> None:
    g = torch.Generator().manual_seed(7)
    egri = (torch.rand(500, generator=g).numpy() * 0.02) + 0.20
    for i in (100, 250, 400):
        egri[i] = 2.0
    bulunan = set(sicrama_bul(egri))
    kontrol("A7 sicramalar bulunuyor", {100, 250, 400} <= bulunan
            and len(bulunan) <= 6,
            f"uc sicrama gomuldu, {len(bulunan)} bulundu: {sorted(bulunan)}")


def a8_zincir_gercegi_geri_veriyor() -> None:
    """Gercek yerel donusumler zincirlenince gercek kuresel donusum cikmali.

    Bu, butun analizin dayandigi varsayim. Tutmuyorsa olculen 'suruklenme'
    modelden degil kodun kendisinden geliyor olurdu.
    """
    yerel = rastgele_rijit(400, tohum=8, olcek=1.0).float()
    kuresel = kuresel_biriktir(yerel)
    beklenen = torch.eye(4)
    for i in range(400):
        beklenen = beklenen @ yerel[i]
    fark = (kuresel[-1] - beklenen).abs().max().item()
    kontrol("A8 zincirleme tutarli", fark < 1e-4,
            f"400 adim, son karede en buyuk fark {fark:.2e}")


def main() -> int:
    print("=" * 66)
    print("FAZ 3 ANALIZ MATEMATIGI")
    print("=" * 66)
    for f in (a1_alti_vektor_gidis_donus, a2_tekil_ve_yiginli_ayni,
              a3_saf_yanlilik_tam_gideriliyor, a4_yansiz_gurultude_giderme_ise_yaramaz,
              a5_olcek_bilinen_degeri_buluyor, a6_guvenilirlik_kapisi_calisiyor,
              a7_sicrama_bulunuyor, a8_zincir_gercegi_geri_veriyor):
        try:
            f()
        except Exception as e:                      # noqa: BLE001
            kontrol(f.__name__, False, f"istisna: {type(e).__name__}: {e}")
    print(f"\n{_gecen}/{_gecen + _kalan} kontrol gecti")
    return 1 if _kalan else 0


if __name__ == "__main__":
    sys.exit(main())
