"""Faz 0 Adim 3 icin kucuk bir 'duman testi' veri kumesi uretir.

NEDEN GEREKLI
  Referans train.py iki sey bekler:
    1) data/frames_transfs/<denek>/<tarama>.h5  -> `frames` ve `tforms` AYNI dosyada.
       Acik dogrulama seti bunlari `frames/` ve `transfs/` diye AYRI tutuyor.
    2) partition_by_ratio(ratios=[1]*5) 5 kata bolunuyor. 3 denekle
       set_sizes=[1,1,1,0,0] cikar; dogrulama kati (fold_03) BOS kalir ve
       dogrulama dongusu coker. En az 5 denek gerekiyor.

  Acik dogrulama setinde 3 denek x 2 tarama = 6 tarama var. Her taramayi bir
  sozde-denek sayip iki yariya boluyoruz -> 6 sozde-denek x 2 tarama.

UYARI - BU KUME YALNIZCA BORU HATTI TESTI ICINDIR
  Ayni gercek kisi birden fazla sozde-denege dagiliyor, yani tanim geregi
  veri sizintisi var. Buradan cikan HICBIR sayi olcum olarak kullanilamaz.
  Amac tek sey: kodun hatasiz donmesi.
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

KOK = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", type=Path, default=KOK / "data",
                    help="acik dogrulama setinin acildigi dizin")
    ap.add_argument("--hedef", type=Path, default=KOK / "data" / "duman",
                    help="uretilecek duman kumesi")
    ap.add_argument("--kare", type=int, default=64,
                    help="sozde-tarama basina kare sayisi")
    a = ap.parse_args()

    frames_kok = a.kaynak / "frames"
    transfs_kok = a.kaynak / "transfs"
    calib = a.kaynak / "calib_matrix.csv"
    for p in (frames_kok, transfs_kok, calib):
        if not p.exists():
            raise SystemExit(f"bulunamadi: {p}\nDogrulama setini once ac.")

    # gercek taramalari topla
    taramalar = sorted(
        (d.name, f.name)
        for d in sorted(frames_kok.iterdir()) if d.is_dir()
        for f in sorted(d.glob("*.h5"))
    )
    print(f"{len(taramalar)} gercek tarama bulundu")
    if len(taramalar) < 5:
        raise SystemExit("5 katli bolme icin en az 5 gercek tarama gerekir")

    ft = a.hedef / "frames_transfs"
    if a.hedef.exists():
        shutil.rmtree(a.hedef)
    ft.mkdir(parents=True)

    for i, (denek, dosya) in enumerate(taramalar):
        with h5py.File(frames_kok / denek / dosya, "r") as fh, \
             h5py.File(transfs_kok / denek / dosya, "r") as th:
            n = fh["frames"].shape[0]
            gerekli = 2 * a.kare
            if n < gerekli:
                raise SystemExit(f"{denek}/{dosya}: {n} kare var, {gerekli} gerekli")
            # taramanin ortasindan iki BITISIK ve AYRIK dilim al:
            # bitisik olmasi kare-arasi hareketi gercekci tutar
            bas = (n - gerekli) // 2
            dilimler = [slice(bas, bas + a.kare),
                        slice(bas + a.kare, bas + gerekli)]

            hedef_denek = ft / f"{i:03d}"
            hedef_denek.mkdir()
            for d, ad in zip(dilimler, ("LH_rotation.h5", "RH_rotation.h5")):
                kareler = np.asarray(fh["frames"][d])
                tf = np.asarray(th["tforms"][d])
                with h5py.File(hedef_denek / ad, "w") as out:
                    out.create_dataset("frames", data=kareler, compression="gzip")
                    out.create_dataset("tforms", data=tf)
            print(f"  sozde-denek {i:03d}  <- {denek}/{dosya}  kare {bas}..{bas+gerekli}")

    shutil.copy2(calib, a.hedef / "calib_matrix.csv")

    # landmarks: train.py kullanmiyor, generate_DDF.py kullaniyor. Yapiyi kur.
    lm_kaynak = a.kaynak / "landmarks"
    if lm_kaynak.exists():
        lm_hedef = a.hedef / "landmarks"
        lm_hedef.mkdir()
        for i, (denek, dosya) in enumerate(taramalar):
            src = lm_kaynak / f"landmark_{denek}.h5"
            if not src.exists():
                continue
            anahtar = dosya[:-3]
            with h5py.File(src, "r") as sh, \
                 h5py.File(lm_hedef / f"landmark_{i:03d}.h5", "w") as oh:
                lm = np.asarray(sh[anahtar])
                # kare indeksi dilime gore kaydirilmali; duman testinde
                # sinirlar icine kirp - bilimsel gecerliligi yok, yapisal test
                for ad in ("LH_rotation", "RH_rotation"):
                    k = lm.copy()
                    k[:, 0] = np.clip(k[:, 0] % a.kare, 0, a.kare - 1)
                    oh.create_dataset(ad, data=k)

    boyut = sum(f.stat().st_size for f in a.hedef.rglob("*") if f.is_file())
    print(f"\nuretildi: {a.hedef}  ({boyut/1e6:.1f} MB, "
          f"{len(taramalar)} sozde-denek x 2 tarama x {a.kare} kare)")
    print("UYARI: sizintili - yalnizca boru hatti testi icin.")


if __name__ == "__main__":
    main()
