"""Zenodo'dan veri kumelerini indirir — kesintiye dayanikli, dogrulamali.

NEDEN AYRI BETIK
  TUS-REC2024 toplam 88,6 GB. Tek parca `curl` cagrisi kesilirse bastan
  baslamak gerekir. Burada her dosya:
    - yarim kalmissa KALDIGI YERDEN devam eder (HTTP Range)
    - inince MD5 ile dogrulanir; tutmuyorsa dosya silinir, yeniden denenir
    - zaten indirilmis ve dogrulanmissa atlanir
  Yani betigi tekrar tekrar calistirmak guvenli.

KULLANIM
  python scripts/veri_indir.py --kume tusrec2024 --hedef D:\\SerbestEl-veri
  python scripts/veri_indir.py --kume tusrec2025-dogrulama --hedef data
  python scripts/veri_indir.py --kume tusrec2024 --sadece-listele
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Zenodo kayit numaralari. TUS-REC2025 EGITIM kumesi burada YOK: o kayit
# kisitli erisimde, Zenodo uzerinden talep ve onay gerektiriyor.
KUMELER: dict[str, dict[str, int]] = {
    "tusrec2024": {
        "egitim-1": 11178508,     # train_part1.zip + calib_matrix.csv
        "egitim-2": 11180794,     # train_part2.zip
        "egitim-3": 11355499,     # landmark.zip
        "dogrulama": 12979481,    # Freehand_US_data_val.zip
    },
    "tusrec2025-dogrulama": {
        "dogrulama": 15699958,
    },
}


def kayit_dosyalari(kayit_id: int) -> list[dict]:
    with urllib.request.urlopen(
        f"https://zenodo.org/api/records/{kayit_id}", timeout=60
    ) as r:
        d = json.load(r)
    if d["metadata"].get("access_right") != "open":
        raise SystemExit(
            f"kayit {kayit_id} acik erisimde degil "
            f"({d['metadata'].get('access_right')}). Zenodo uzerinden talep gerekiyor."
        )
    return [
        {"ad": f["key"], "boyut": f["size"], "md5": f["checksum"].split(":")[-1],
         "url": f["links"]["self"]}
        for f in d.get("files", [])
    ]


def md5_hesapla(yol: Path, blok: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(yol, "rb") as f:
        while parca := f.read(blok):
            h.update(parca)
    return h.hexdigest()


def indir(dosya: dict, hedef: Path, deneme: int = 3) -> bool:
    yol = hedef / dosya["ad"]
    imza = hedef / f".{dosya['ad']}.dogrulandi"

    # daha once indirilip dogrulanmis mi
    if imza.exists() and yol.exists() and yol.stat().st_size == dosya["boyut"]:
        print(f"  {dosya['ad']}: zaten var ve dogrulanmis, atlaniyor")
        return True

    gb = dosya["boyut"] / 1e9
    for d in range(1, deneme + 1):
        var = yol.stat().st_size if yol.exists() else 0
        if var and var < dosya["boyut"]:
            print(f"  {dosya['ad']}: {var/1e9:.2f}/{gb:.2f} GB var, devam ediliyor "
                  f"(deneme {d}/{deneme})")
        else:
            print(f"  {dosya['ad']}: {gb:.2f} GB indiriliyor (deneme {d}/{deneme})")

        bas = time.time()
        # -C -  : yarim dosyadan devam
        # --retry: gecici ag hatalarinda kendi icinde tekrar dener
        sonuc = subprocess.run(
            ["curl", "-L", "--fail", "-C", "-", "--retry", "5",
             "--retry-delay", "10", "--retry-all-errors",
             "-o", str(yol), dosya["url"]],
            capture_output=True, text=True,
        )
        sure = time.time() - bas

        if sonuc.returncode != 0:
            print(f"    curl hata {sonuc.returncode}: {sonuc.stderr.strip()[:200]}")
            continue

        if yol.stat().st_size != dosya["boyut"]:
            print(f"    boyut tutmadi: {yol.stat().st_size} / {dosya['boyut']}")
            continue

        print(f"    indi ({sure/60:.1f} dk, {gb/max(sure,1)*1000:.1f} MB/s), "
              f"MD5 dogrulaniyor...")
        bulunan = md5_hesapla(yol)
        if bulunan == dosya["md5"]:
            imza.write_text(bulunan)
            print(f"    MD5 TAMAM {bulunan}")
            return True
        print(f"    MD5 TUTMADI\n      bulunan : {bulunan}\n      beklenen: {dosya['md5']}")
        yol.unlink()   # bozuk dosyayla devam etmenin anlami yok, bastan
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", choices=sorted(KUMELER), required=True)
    ap.add_argument("--hedef", type=Path, required=True)
    ap.add_argument("--sadece-listele", action="store_true")
    a = ap.parse_args()

    print(f"kume: {a.kume}")
    tum: list[dict] = []
    for ad, rid in KUMELER[a.kume].items():
        for f in kayit_dosyalari(rid):
            f["kayit"] = ad
            tum.append(f)

    toplam = sum(f["boyut"] for f in tum) / 1e9
    print(f"{len(tum)} dosya, toplam {toplam:.1f} GB\n")
    for f in tum:
        print(f"  [{f['kayit']:<10}] {f['ad']:<44} {f['boyut']/1e9:7.2f} GB")
    if a.sadece_listele:
        return 0

    a.hedef.mkdir(parents=True, exist_ok=True)
    # yer kontrolu: indirilen + acilmis hali icin kabaca iki kati
    import shutil

    bos = shutil.disk_usage(a.hedef).free / 1e9
    gerekli = toplam * 2.1
    print(f"\nhedef: {a.hedef}  (bos {bos:.0f} GB, acilmis hali dahil "
          f"~{gerekli:.0f} GB gerekir)")
    if bos < gerekli:
        print(f"UYARI: yer dar olabilir. Zip'leri acildiktan sonra silmeyi dusun.")

    basarisiz = []
    for i, f in enumerate(tum, 1):
        print(f"\n[{i}/{len(tum)}] {f['ad']}")
        if not indir(f, a.hedef):
            basarisiz.append(f["ad"])

    print("\n" + "=" * 60)
    if basarisiz:
        print(f"BASARISIZ ({len(basarisiz)}): {', '.join(basarisiz)}")
        print("Betigi tekrar calistir - tamamlananlar atlanir, yarimlar devam eder.")
        return 1
    print(f"HEPSI TAMAM - {len(tum)} dosya, {toplam:.1f} GB, MD5 dogrulandi")
    print(f"konum: {a.hedef}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
