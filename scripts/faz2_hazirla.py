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
import concurrent.futures as vadeli
import os
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


def _parca_cikar(is_paketi: tuple[str, str, list[str]]) -> int:
    """Bir isci surecin payi. Kendi ZipFile taniticisini acar — paylasilamaz."""
    zip_yolu, hedef, adlar = is_paketi
    with zipfile.ZipFile(zip_yolu) as z:
        for ad in adlar:
            z.extract(ad, hedef)
    return len(adlar)


def _paylar(girdiler: list[zipfile.ZipInfo], n: int) -> list[list[str]]:
    """Girisleri sikistirilmis BOYUTA gore n kovaya dagitir.

    Dosya SAYISINA gore bolmek ise yaramaz: bu kumede tek dosyalar 100-400 MB
    arasinda degisiyor, esit sayida dosya alan isciler cok farkli surelerde
    biter ve en yavasi herkesi bekletir.
    """
    kovalar: list[list[str]] = [[] for _ in range(n)]
    yuk = [0] * n
    for g in sorted(girdiler, key=lambda g: g.compress_size, reverse=True):
        i = yuk.index(min(yuk))
        kovalar[i].append(g.filename)
        yuk[i] += g.compress_size
    return [k for k in kovalar if k]


def zip_ac(zip_yolu: Path, hedef: Path, isci: int = 0) -> bool:
    """Idempotent, paralel acma: bitmis isaretleyicisi varsa atlar.

    NEDEN PARALEL
      Bu zip'ler deflate sikistirmali (~2,2 kat) ve acilmis hali 96 GB.
      Cozme islemci baglidir; disk iki tarafta da NVMe SSD oldugu icin
      darbogaz cekirdek sayisidir, tek cekirdekte saatler surer. Her isci
      kendi ZipFile taniticisini acar (tanitici surecler arasinda
      paylasilamaz); ayri dosyalara yazdiklari icin kilit gerekmez.

    Isaretleyici ancak BUTUN parcalar bittikten sonra yazilir: yarim kalan
    bir acma "acilmis" sayilmaz, tekrar kosulunca bastan alinir.
    """
    imza = hedef / f".{zip_yolu.stem}.acildi"
    if imza.exists():
        print(f"  {zip_yolu.name}: zaten acilmis, atlaniyor")
        return True
    if not zip_yolu.exists():
        print(f"  {zip_yolu.name}: YOK")
        return False

    hedef.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_yolu) as z:
        girdiler = [g for g in z.infolist() if not g.is_dir()]
        dizinler = {g.filename.rsplit("/", 1)[0]
                    for g in girdiler if "/" in g.filename}
    acilmis_gb = sum(g.file_size for g in girdiler) / 1e9
    bos = shutil.disk_usage(hedef).free / 1e9
    if bos < acilmis_gb * 1.05:
        print(f"  {zip_yolu.name}: YER YOK ({bos:.0f} GB bos, "
              f"~{acilmis_gb:.0f} GB gerekli)")
        return False

    for d in dizinler:                    # isciler yarismasin diye onceden
        (hedef / d).mkdir(parents=True, exist_ok=True)

    isci = isci or min(8, os.cpu_count() or 4)
    # Isci sayisinin katı kadar pay: bir pay bitince ilerleme yazilabiliyor.
    # Tam isci sayisi kadar pay olsaydi ilerleme ancak en sonda gorunurdu.
    paylar = _paylar(girdiler, isci * 4)
    print(f"  {zip_yolu.name}: {len(girdiler)} dosya -> {acilmis_gb:.1f} GB, "
          f"{isci} isci, {len(paylar)} pay", flush=True)

    bas = time.time()
    bitti = 0
    with vadeli.ProcessPoolExecutor(max_workers=min(isci, len(paylar))) as havuz:
        isler = [havuz.submit(_parca_cikar, (str(zip_yolu), str(hedef), p))
                 for p in paylar]
        for tamam in vadeli.as_completed(isler):
            bitti += tamam.result()
            print(f"    {bitti}/{len(girdiler)} dosya  "
                  f"({(time.time()-bas)/60:.1f} dk)", flush=True)
    sure = time.time() - bas
    print(f"    bitti ({sure/60:.1f} dk, {acilmis_gb/sure:.2f} GB/s)")
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
