"""Faz 1 dogrulamalarini ELDEKI BUTUN taramalarda kosar.

NEDEN GEREKLI
  tests/test_geometri.py ve tests/test_hacim.py tek tarama (050/LH_rotation)
  uzerinde calisiyor. Denege ya da taramaya ozgu bir tuhaflik varsa orada
  gorunmez: bozuk bir kare, sifir olcekli bir donusum, isaret noktasi
  sinir disi, tarama uzunlugu farki...

  Bu betik ayni kontrolleri her taramada tekrarlayip tablo veriyor.
  Faz 2'ye gecmeden once bunun temiz olmasi gerekiyor.

Kosum:  python scripts/tum_taramalari_dogrula.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
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
from src.hacim import hacim_kur  # noqa: E402


def taramalari_bul(veri: Path) -> list[tuple[str, str]]:
    kok = veri / "frames"
    return sorted(
        (d.name, f.stem)
        for d in sorted(kok.iterdir()) if d.is_dir()
        for f in sorted(d.glob("*.h5"))
    )


def tarama_dogrula(veri: Path, denek: str, tarama: str, kalib: Kalibrasyon,
                   hacim_kontrolu: bool) -> dict:
    s: dict = {"denek": denek, "tarama": tarama}

    with h5py.File(veri / "frames" / denek / f"{tarama}.h5", "r") as f:
        kareler = np.asarray(f["frames"])
    with h5py.File(veri / "transfs" / denek / f"{tarama}.h5", "r") as f:
        tforms = torch.tensor(np.asarray(f["tforms"]))
    with h5py.File(veri / "landmarks" / f"landmark_{denek}.h5", "r") as f:
        isaretler = torch.from_numpy(np.asarray(f[tarama]))

    n, h, w = kareler.shape
    s["kare"] = n

    # --- veri saglikli mi -------------------------------------------------
    s["bicim_ok"] = (h, w) == (480, 640) and tforms.shape == (n, 4, 4)
    s["nan"] = bool(torch.isnan(tforms).any() or torch.isinf(tforms).any())
    detR = torch.linalg.det(tforms[:, 0:3, 0:3])
    s["tforms_gecerli"] = bool((detR > 0.9).all() and (detR < 1.1).all())
    # isaret noktalari sinirlar icinde mi
    s["isaret_ok"] = bool(
        (isaretler[:, 0] >= 0).all() and (isaretler[:, 0] < n).all()
        and (isaretler[:, 1] >= 1).all() and (isaretler[:, 1] <= w).all()
        and (isaretler[:, 2] >= 1).all() and (isaretler[:, 2] <= h).all()
    )
    s["isaret_kare_araligi"] = f"{int(isaretler[:,0].min())}-{int(isaretler[:,0].max())}"

    # --- geometri ic tutarliligi -----------------------------------------
    kur = kuresel_donusumler(tforms, kalib)
    yer = yerel_donusumler(tforms, kalib)
    R = kur[:, 0:3, 0:3]
    s["ortogonallik"] = (R @ R.transpose(-1, -2) - torch.eye(3)).abs().max().item()

    p = piksel_noktalari(h, w, yogunluk=(9, 9))
    mm = kalib.olcek @ p
    ileri = noktalari_tasi(kur, p, kalib)
    ileri_h = torch.cat([ileri, torch.ones(ileri.shape[0], 1, ileri.shape[2])], 1)
    geri = (torch.linalg.inv(kur) @ ileri_h)[:, 0:3, :]
    s["gidis_donus_mm"] = (geri - mm[None, 0:3, :]).abs().max().item()

    # zincir: yerel carpimlari kuresele esit olmali
    birikim, en_kotu = yer[0].clone(), 0.0
    for i in range(1, min(yer.shape[0], 400)):
        birikim = birikim @ yer[i]
        en_kotu = max(en_kotu, (birikim - kur[i]).abs().max().item())
    s["zincir"] = en_kotu

    # --- referansla ayni mi (dort DDF) -----------------------------------
    from utils.generate_ddf_from_label import generate_ddf_from_label

    dilim = min(n, 25)
    ref = generate_ddf_from_label(str(veri / "calib_matrix.csv"), "cpu")
    i_dilim = isaretler.numpy()
    i_dilim = i_dilim[(i_dilim[:, 0] >= 1) & (i_dilim[:, 0] < dilim)]
    if len(i_dilim) == 0:
        i_dilim = np.array([[dilim // 2, 300, 200]], dtype=np.int64)
    r_gp, r_gl, r_lp, r_ll = ref.calculate_GT_DDF(
        kareler[:dilim], tforms[:dilim].numpy(), i_dilim)

    pt = piksel_noktalari(h, w)
    b_gp = yer_degistirme(kuresel_donusumler(tforms[:dilim], kalib), pt, kalib)
    b_lp = yer_degistirme(yerel_donusumler(tforms[:dilim], kalib), pt, kalib)
    it = torch.from_numpy(i_dilim)
    s["ddf_fark"] = max(
        float(np.abs(r_gp - b_gp.numpy()).max()),
        float(np.abs(r_lp - b_lp.numpy()).max()),
        float(np.abs(r_gl - isaret_ddf(b_gp, it).numpy()).max()),
        float(np.abs(r_ll - isaret_ddf(b_lp, it).numpy()).max()),
    )

    # --- hacim capraz dogrulamasi ----------------------------------------
    if hacim_kontrolu:
        import pytorch3d.transforms as t3

        don_tam = kuresel_donusumler(tforms, kalib, ilk_kare_dahil=True)
        disarida = [n // 3, 2 * n // 3]
        tut = np.ones(n, dtype=bool)
        for d in disarida:
            tut[max(d - 2, 0):d + 3] = False
        hacim = hacim_kur(kareler[tut], None, kalib, voksel_mm=0.6, kare_atla=3,
                          ilerleme=False, donusumler=don_tam[tut])

        a = np.deg2rad(4.0)
        B = torch.eye(4)
        B[0:3, 0:3] = t3.euler_angles_to_matrix(torch.tensor([a, a, a]), "ZYX")
        B[0:3, 3] = 4.0

        dogru, bozuk = [], []
        pt_tam = piksel_noktalari(h, w)
        for d in disarida:
            gercek = kareler[d].reshape(-1).astype(np.float32)
            for T, liste in ((don_tam[d], dogru), (B @ don_tam[d], bozuk)):
                koord = (T @ (kalib.olcek @ pt_tam))[0:3].numpy()
                idx = ((koord - hacim.kok_mm[:, None]) / hacim.voksel_mm).astype(np.int64)
                nz, ny, nx = hacim.ortalama.shape
                gec = ((idx[0] >= 0) & (idx[0] < nx) & (idx[1] >= 0) & (idx[1] < ny)
                       & (idx[2] >= 0) & (idx[2] < nz))
                oku = np.full(idx.shape[1], np.nan, np.float32)
                oku[gec] = hacim.ortalama[idx[2][gec], idx[1][gec], idx[0][gec]]
                m = gec & ~np.isnan(oku)
                liste.append(np.corrcoef(gercek[m], oku[m])[0, 1] if m.sum() > 1000
                             else np.nan)
        s["r_dogru"] = float(np.nanmean(dogru))
        s["r_bozuk"] = float(np.nanmean(bozuk))
        s["doluluk"] = hacim.doluluk
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", type=Path, default=KOK / "data")
    ap.add_argument("--hacim-atla", action="store_true",
                    help="hacim capraz dogrulamasini atla (hizli)")
    a = ap.parse_args()

    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    taramalar = taramalari_bul(a.veri)
    print(f"{len(taramalar)} tarama bulundu\n")

    satirlar = []
    for denek, tarama in taramalar:
        print(f"  {denek}/{tarama} ...", end=" ", flush=True)
        satirlar.append(tarama_dogrula(a.veri, denek, tarama, kalib,
                                       not a.hacim_atla))
        print("bitti")

    print(f"\n{'tarama':<20} {'kare':>5} {'ortog':>9} {'gidis-don':>10} "
          f"{'zincir':>9} {'DDF fark':>9}", end="")
    if not a.hacim_atla:
        print(f" {'r dogru':>8} {'r bozuk':>8} {'fark':>7}", end="")
    print()
    print("-" * (95 if not a.hacim_atla else 68))

    tum_ok = True
    for s in satirlar:
        ad = f"{s['denek']}/{s['tarama']}"
        print(f"{ad:<20} {s['kare']:>5} {s['ortogonallik']:>9.1e} "
              f"{s['gidis_donus_mm']:>10.1e} {s['zincir']:>9.1e} {s['ddf_fark']:>9.1e}",
              end="")
        ok = (s["bicim_ok"] and not s["nan"] and s["tforms_gecerli"]
              and s["isaret_ok"] and s["ortogonallik"] < 1e-4
              and s["gidis_donus_mm"] < 1e-2 and s["zincir"] < 1e-2
              and s["ddf_fark"] < 1e-3)
        if not a.hacim_atla:
            fark = s["r_dogru"] - s["r_bozuk"]
            print(f" {s['r_dogru']:>8.3f} {s['r_bozuk']:>8.3f} {fark:>+7.3f}", end="")
            ok = ok and s["r_dogru"] > 0.70 and fark > 0.15
        print("  " + ("OK" if ok else "SORUN"))
        tum_ok = tum_ok and ok

    print("\nveri saglik ozeti:")
    for s in satirlar:
        bayrak = []
        if not s["bicim_ok"]: bayrak.append("bicim")
        if s["nan"]: bayrak.append("NaN/Inf")
        if not s["tforms_gecerli"]: bayrak.append("gecersiz donme")
        if not s["isaret_ok"]: bayrak.append("isaret sinir disi")
        print(f"  {s['denek']}/{s['tarama']:<14} isaret kare araligi "
              f"{s['isaret_kare_araligi']:<10} "
              f"{'TEMIZ' if not bayrak else 'SORUN: ' + ', '.join(bayrak)}")

    print(f"\n{'HEPSI TEMIZ' if tum_ok else 'EN AZ BIR TARAMADA SORUN VAR'}")
    return 0 if tum_ok else 1


if __name__ == "__main__":
    sys.exit(main())
