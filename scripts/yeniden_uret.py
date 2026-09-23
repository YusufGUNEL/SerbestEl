"""README'deki butun sayilari tek komutla yeniden uretir.

Yeni kod yok: mevcut betikleri README'deki sirayla zincirler. Her adim
kendi dizinine yazar ve tamamlanmis adimi tekrar kosmaz, yani kesilen bir
kosum ayni komutla kaldigi yerden surer.

ADIMLAR (toplam ~23 saat, RTX 3050 Ti 4 GB)
  hazirla   zip'leri acar, bolmeyi uretir, testleri kosar      ~10 dk
  merdiven  Faz 5 ablasyon merdiveni, 4 x 180 dk               ~12,5 sa
  uzun      en iyi yapilandirma, 480 dk                          ~8,5 sa
  tablo     merdiven tablosu (results/faz5/tablo.md)             saniyeler
  test      kazanani TEST kumesinde, bir kere olcer              ~20 dk

KULLANIM
  python scripts/yeniden_uret.py --kaynak D:/SerbestEl-veri/tusrec2024
  python scripts/yeniden_uret.py --adim merdiven tablo      # yalniz bunlar
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
PY = sys.executable
MERDIVEN = ["A1_referans", "A2_C", "A3_CB", "A4_CBG"]
ADIMLAR = ["hazirla", "merdiven", "uzun", "tablo", "test"]


def kos(komut: list[str]) -> None:
    print("\n$ " + " ".join(komut), flush=True)
    if subprocess.run(komut, cwd=KOK).returncode != 0:
        raise SystemExit(f"basarisiz: {' '.join(komut)}")


def bitti(kok: Path, ad: str) -> bool:
    # deney.json yalnizca uc adim (egitim, olcum, bagdasim) bittiginde yaziliyor
    return (kok / ad / "deney.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kaynak", type=Path, default=Path("D:/SerbestEl-veri/tusrec2024"),
                    help="TUS-REC2024 zip'lerinin durdugu dizin")
    ap.add_argument("--adim", nargs="*", default=ADIMLAR, choices=ADIMLAR)
    ap.add_argument("--kazanan", default="U_uzun_C",
                    help="uzun butceyle kosulup test kumesinde olculecek deney")
    a = ap.parse_args()
    veri = a.kaynak / "acilmis"
    deney = [PY, "-u", "scripts/faz4_deney.py", "--veri", str(veri)]

    if "hazirla" in a.adim:
        kos([PY, "-u", "scripts/faz2_hazirla.py", "--kaynak", str(a.kaynak)])

    if "merdiven" in a.adim:
        kalan = [ad for ad in MERDIVEN if not bitti(KOK / "results/faz5", ad)]
        if kalan:
            kos(deney + kalan + ["--sure", "180", "--kok", "results/faz5"])

    if "uzun" in a.adim and not bitti(KOK / "results/faz6", a.kazanan):
        kos(deney + [a.kazanan, "--sure", "480", "--kok", "results/faz6"])

    if "tablo" in a.adim:
        kos([PY, "-u", "scripts/faz4_tablo.py", "--kok", "results/faz5"])

    if "test" in a.adim:
        kos(deney + [a.kazanan, "--kok", "results/faz6", "--test-olcumu"])

    print("\nBITTI. Sayilar: results/faz5/tablo.md ve "
          f"results/faz6/{a.kazanan}/test_olculer.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
