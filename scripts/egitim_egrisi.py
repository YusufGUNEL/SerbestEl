"""Egitim egrisini cizer — `olcumler.csv` -> PNG.

NEDEN AYRI BIR ARAC
  Faz 4'te her deney ayni CSV'yi uretecek; egrileri yan yana koyabilmek
  karsilastirmanin kendisi. Birden cok kosum verilirse hepsi ayni eksene
  cizilir.

  Dogrulama mesafesi (mm) ile kayip AYRI panelde: kayip birimsiz ve olcegi
  deneyden deneye degisir, mesafe milimetredir ve dogrudan okunur. Ayni
  eksene koymak ikisini de okunmaz yapar.

KULLANIM
  python scripts/egitim_egrisi.py results/faz2_referans
  python scripts/egitim_egrisi.py results/a results/b --cikti results/kiyas.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

KOK = Path(__file__).resolve().parents[1]
RENKLER = ["#c0392b", "#2e7d32", "#1f6f8b", "#8e44ad", "#e08a2e"]


def oku(kosum: Path) -> dict[str, list]:
    yol = kosum / "olcumler.csv" if kosum.is_dir() else kosum
    e, ek, em, dk, dm = [], [], [], [], []
    with open(yol, newline="", encoding="utf-8") as f:
        for s in csv.DictReader(f):
            e.append(int(s["epok"]))
            ek.append(float(s["egitim_kayip"]))
            em.append(float(s["egitim_mesafe"]))
            # dogrulama her epokta yok — bos satirlar atlaniyor
            if s.get("dogrulama_mesafe"):
                dk.append((int(s["epok"]), float(s["dogrulama_kayip"])))
                dm.append((int(s["epok"]), float(s["dogrulama_mesafe"])))
    return {"epok": e, "e_kayip": ek, "e_mesafe": em, "d_kayip": dk,
            "d_mesafe": dm, "ad": (kosum.name if kosum.is_dir()
                                   else kosum.parent.name)}


def ciz(kosumlar: list[dict], cikti: Path) -> Path:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    f, (sol, sag) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, renk in zip(kosumlar, RENKLER * 4):
        sol.plot(k["epok"], k["e_kayip"], color=renk, lw=0.9, alpha=0.55,
                 label=f"{k['ad']} egitim")
        if k["d_kayip"]:
            sol.plot([x for x, _ in k["d_kayip"]], [y for _, y in k["d_kayip"]],
                     color=renk, lw=1.8, label=f"{k['ad']} dogrulama")
        sag.plot(k["epok"], k["e_mesafe"], color=renk, lw=0.9, alpha=0.55)
        if k["d_mesafe"]:
            x = [a for a, _ in k["d_mesafe"]]
            y = [b for _, b in k["d_mesafe"]]
            sag.plot(x, y, color=renk, lw=1.8)
            i = min(range(len(y)), key=lambda j: y[j])
            sag.scatter([x[i]], [y[i]], color=renk, s=30, zorder=3)
            sag.annotate(f"en iyi {y[i]:.3f} mm\n(epok {x[i]})",
                         (x[i], y[i]), textcoords="offset points",
                         xytext=(8, 10), fontsize=7.5, color=renk)
    sol.set_yscale("log")
    sol.set_xlabel("epok"); sol.set_ylabel("kayip (MSE, birimsiz)")
    sol.set_title("Kayip")
    sol.legend(fontsize=7.5)
    sag.set_yscale("log")
    sag.set_xlabel("epok"); sag.set_ylabel("nokta mesafesi (mm)")
    sag.set_title("Mesafe — ince cizgi egitim, kalin cizgi dogrulama")
    f.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(cikti)
    plt.close(f)
    return cikti


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kosum", type=Path, nargs="+",
                    help="kayit dizini ya da dogrudan olcumler.csv")
    ap.add_argument("--cikti", type=Path, default=None)
    a = ap.parse_args()

    kosumlar = [oku(k) for k in a.kosum]
    cikti = a.cikti or (a.kosum[0] if a.kosum[0].is_dir()
                        else a.kosum[0].parent) / "egitim_egrisi.png"
    ciz(kosumlar, cikti)
    for k in kosumlar:
        en = min(k["d_mesafe"], key=lambda t: t[1]) if k["d_mesafe"] else None
        print(f"{k['ad']:<20} {len(k['epok']):>5} epok   "
              + (f"en iyi dogrulama {en[1]:.4f} mm (epok {en[0]})"
                 if en else "dogrulama olcumu yok"))
    print(f"yazildi: {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
