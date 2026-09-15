"""Faz 2'yi tek komutla baslatilabilir hale getirir.

Yaptiklari, sirayla:
  1. Zip'leri acar (acilmissa atlar — tekrar calistirmak guvenli)
  2. Veri duzenini cozup yazdirir
  3. Denek bazli bolmeyi uretir (varsa dokunmaz)
  4. Butun testleri kosar
  5. Egitimin GERCEKTEN basladigini 2 epokla dogrular
Sonunda "HAZIR" ya da nerede takildigi yazar.

KULLANIM
  python scripts/faz2_hazirla.py --kaynak D:/SerbestEl-veri/tusrec2024
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
PY = sys.executable


def adim(no: int, ad: str) -> None:
    print(f"\n{'='*62}\n{no}. {ad}\n{'='*62}", flush=True)


def zip_ac(zip_yolu: Path, hedef: Path) -> bool:
    """Idempotent acma: bitmis isaretleyicisi varsa atlar."""
    imza = hedef / f".{zip_yolu.stem}.acildi"
    if imza.exists():
        print(f"  {zip_yolu.name}: zaten acilmis, atlaniyor")
        return True
    if not zip_yolu.exists():
        print(f"  {zip_yolu.name}: YOK")
        return False

    hedef.mkdir(parents=True, exist_ok=True)
    gb = zip_yolu.stat().st_size / 1e9
    bos = shutil.disk_usage(hedef).free / 1e9
    if bos < gb * 1.2:
        print(f"  {zip_yolu.name}: YER YOK ({bos:.0f} GB bos, ~{gb:.0f} GB gerekli)")
        return False

    print(f"  {zip_yolu.name}: {gb:.1f} GB aciliyor...", flush=True)
    bas = time.time()
    with zipfile.ZipFile(zip_yolu) as z:
        girdiler = z.infolist()
        for i, g in enumerate(girdiler, 1):
            z.extract(g, hedef)
            if i % 50 == 0 or i == len(girdiler):
                print(f"\r    {i}/{len(girdiler)}", end="", flush=True)
    print(f"\n    bitti ({(time.time()-bas)/60:.1f} dk)")
    imza.write_text("ok")
    return True


def kos(ad: str, komut: list[str]) -> bool:
    print(f"\n--- {ad}")
    s = subprocess.run(komut, cwd=KOK)
    return s.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", type=Path, default=Path("D:/SerbestEl-veri/tusrec2024"),
                    help="zip'lerin bulundugu dizin")
    ap.add_argument("--hedef", type=Path, default=None,
                    help="acilacak dizin (varsayilan: <kaynak>/acilmis)")
    ap.add_argument("--bolme", type=Path, default=KOK / "configs" / "bolme.json")
    ap.add_argument("--acma-atla", action="store_true")
    a = ap.parse_args()
    hedef = a.hedef or (a.kaynak / "acilmis")
    sorun: list[str] = []

    adim(1, "Zip'leri ac")
    if a.acma_atla:
        print("  atlandi (--acma-atla)")
    else:
        for ad in ("train_part1.zip", "train_part2.zip", "landmark.zip"):
            if not zip_ac(a.kaynak / ad, hedef):
                sorun.append(f"acilamadi: {ad}")
        kalib = a.kaynak / "calib_matrix.csv"
        if kalib.exists():
            shutil.copy2(kalib, hedef / "calib_matrix.csv")
            print("  calib_matrix.csv kopyalandi")
        else:
            sorun.append("calib_matrix.csv yok")

    adim(2, "Veri duzeni")
    try:
        from src.veri import ozet

        ozet(hedef)
    except SystemExit as e:
        sorun.append(f"veri duzeni cozulemedi: {e}")
        print(f"  SORUN: {e}")

    adim(3, "Denek bazli bolme")
    if a.bolme.exists():
        print(f"  {a.bolme.name} zaten var, DOKUNULMADI")
        kos("mevcut bolme", [PY, "src/bolme.py", "--veri", str(hedef),
                             "--cikti", str(a.bolme)])
    elif not kos("bolme uret", [PY, "src/bolme.py", "--veri", str(hedef),
                                "--cikti", str(a.bolme)]):
        sorun.append("bolme uretilemedi")

    adim(4, "Testler")
    for t in ("scripts/ortam_dogrula.py", "tests/test_geometri.py",
              "tests/test_hacim.py", "tests/test_olcum.py"):
        if not kos(t, [PY, t]):
            sorun.append(f"test basarisiz: {t}")

    adim(5, "Egitim gercekten basliyor mu (2 epok)")
    if not kos("egitim denemesi",
               [PY, "src/egit.py", "--veri", str(hedef), "--bolme", str(a.bolme),
                "--kayit", str(KOK / "results" / "faz2_hazirlik_denemesi"),
                "--epok", "2", "--dogrulama-sikligi", "1", "--isci", "2"]):
        sorun.append("egitim baslamadi")

    print(f"\n{'='*62}")
    if sorun:
        print("HAZIR DEGIL:")
        for s in sorun:
            print(f"  - {s}")
        return 1
    print("HAZIR. Faz 2 baslatmak icin:")
    print(f"  python src/egit.py --veri {hedef} --epok 20000")
    print(f"  python src/egit.py --devam          # kesilirse")
    print(f"\nSonra dort olcu:")
    print(f"  python src/olcum.py --veri {hedef} "
          f"--agirlik results/faz2_referans/en_iyi_model.pt --kume test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
