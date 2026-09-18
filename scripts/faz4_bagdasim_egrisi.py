"""Bagdasim egrisi — cokus butceye mi bagli?

NEDEN BU OLCUM
  Faz 4'un dort deneyi de tek bir soruyu cevapsiz birakti. Baslangic
  agirligi, donme temsili, baglam uzunlugu ve veri yogunlugu tek tek
  degistirildi; dordu de dort olcuyu %2-6 oynatti, hicbiri bagdasimi
  kipirdatmadi. Geriye tek eksen kaldi: GORULEN PENCERE SAYISI.

  Elimizde o eksenin ust ucunu gosteren bir nokta zaten var — Faz 3'un
  kontrol kosumu. On egitimli referans modeli AYNI mimari, AYNI veri,
  r = 0,84. Yani mimari bu isi ogrenebiliyor. Bilinmeyen tek sey aradaki
  yol: bagdasim egitim olcegiyle nasil degisiyor?

  Tek bir bitmis modele bakarak bunu soylemek imkansiz. Gereken sey egri.
  Bu betik uzun kosumun biraktigi ara anlik goruntuleri sirayla yukluyor
  ve her biri icin bagdasimi olcuyor.

NASIL OKUNUR
  Duz cizgi      -> cokus butceye bagli DEGIL. Daha uzun egitmek cozmez;
                    bu donanimda referansin yontemi yeniden uretilemez ve
                    dogru cevap budur, uydurma bir sayi degil.
  Yukselen egri  -> bagli. Egimi, referansin 14,4 milyon penceresine ne
                    kadar uzakta oldugumuzu SAYIYLA soyler.

  Ikisi de yazilabilir bir sonuc. Hangisi oldugunu bilmemek en kotusu.

NEDEN AYRI BIR BETIK, analiz.py DEGIL
  `src/analiz.py` bir modelin TAM teshisini cikariyor: yanlilik giderme,
  olcek duzeltme, sicrama bulma, en kotu taramalarin gorselleri. Burada
  gereken tek sey alti bilesenin bagdasimi, ama ON KERE. Tam teshisi on
  kere kosturmak saatler surerdi; bu betik yalnizca gereken parcayi
  cagiriyor.

Kullanim:
  python scripts/faz4_bagdasim_egrisi.py --kosum results/faz4/U_uzun
  python scripts/faz4_bagdasim_egrisi.py --kosum results/faz4/U_uzun --tarama 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from src.analiz import BILESEN, alti_vektor, dogrusal_uydur  # noqa: E402
from src.bolme import yukle as bolme_yukle  # noqa: E402
from src.geometri import Kalibrasyon, yerel_donusumler  # noqa: E402
from src.olcum import model_yukle, yerel_tahmin  # noqa: E402
from src.veri import taramalari_bul  # noqa: E402


def taramalari_sec(veri: Path, denekler: list[str], tarama_basina: int):
    """Her denekten ilk N tarama — ada gore sirali, rastgelelik yok.

    Egrideki butun noktalar AYNI taramalarda olculmeli; yoksa egrinin
    inip cikmasi modelin degisiminden mi tarama secimindeki gurultuden mi
    geldigi anlasilmaz.
    """
    _, tum = taramalari_bul(veri)
    istenen = set(denekler)
    gruplu: dict[str, list] = {}
    for t in sorted((t for t in tum if t.denek in istenen),
                    key=lambda t: (t.denek, t.ad)):
        gruplu.setdefault(t.denek, []).append(t)
    return [t for g in gruplu.values() for t in g[:tarama_basina]]


def bir_modelin_bagdasimi(agirlik: Path, taramalar, kalib, aygit,
                          num_samples: int, num_pred: int,
                          donme_temsili: str) -> dict:
    """Alti bilesenin `tahmin ~ a . gercek + b` uydurmasi, taramalar birlesik."""
    model, ciftler, _ = model_yukle(agirlik, aygit, num_samples=num_samples,
                                    num_pred=num_pred,
                                    donme_temsili=donme_temsili)
    gercekler, tahminler = [], []
    for t in taramalar:
        with h5py.File(t.kare_yolu) as f:
            kareler = np.asarray(f["frames"])
        with h5py.File(t.tform_yolu) as f:
            tforms = torch.tensor(np.asarray(f["tforms"]))
        gt_yerel = yerel_donusumler(tforms, kalib)
        t_yerel = yerel_tahmin(model, kareler, kalib, ciftler, aygit,
                               num_samples=num_samples,
                               donme_temsili=donme_temsili)
        gercekler.append(alti_vektor(gt_yerel))
        tahminler.append(alti_vektor(t_yerel))

    g = torch.cat(gercekler)
    t = torch.cat(tahminler)
    a, b, r = dogrusal_uydur(g, t)
    return {"a": a.tolist(), "b": b.tolist(), "r": r.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kosum", type=Path, required=True,
                    help="ara_epok*.pt dosyalarini iceren deney dizini")
    ap.add_argument("--veri", type=Path,
                    default=Path("D:/SerbestEl-veri/tusrec2024/acilmis"))
    ap.add_argument("--bolme", type=Path, default=KOK / "configs" / "bolme.json")
    ap.add_argument("--kume", default="dogrulama")
    ap.add_argument("--tarama", type=int, default=4, metavar="N",
                    help="denek basina tarama; egrinin her noktasinda AYNI")
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--num-pred", type=int, default=1)
    ap.add_argument("--donme-temsili", default="euler",
                    choices=["euler", "6b", "kuaterniyon", "matris"])
    a = ap.parse_args()

    anlik = sorted(a.kosum.glob("ara_epok*.pt"))
    if not anlik:
        raise SystemExit(
            f"ara anlik goruntu yok: {a.kosum}\n"
            f"Egitimi `--ara-kayit EPOK` ile kosmak gerekiyor.")

    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    bolme = bolme_yukle(a.bolme)
    taramalar = taramalari_sec(a.veri, bolme.denek_kumesi(a.kume), a.tarama)
    print(f"{len(anlik)} anlik goruntu x {len(taramalar)} tarama "
          f"({a.kume} kumesi) | aygit {aygit}\n")

    satirlar = []
    for yol in anlik:
        epok = int(yol.stem.replace("ara_epok", ""))
        s = bir_modelin_bagdasimi(yol, taramalar, kalib, aygit,
                                  a.num_samples, a.num_pred, a.donme_temsili)
        mutlak = sorted(abs(x) for x in s["r"])
        satir = {"epok": epok, "pencere": epok * 720,
                 "r_ortanca": mutlak[len(mutlak) // 2],
                 "r_en_buyuk": mutlak[-1],
                 **{f"r_{ad.split()[0]}{ad.split()[1][0]}": v
                    for ad, v in zip(BILESEN, s["r"])},
                 **{f"a_{ad.split()[0]}{ad.split()[1][0]}": v
                    for ad, v in zip(BILESEN, s["a"])}}
        satirlar.append(satir)
        print(f"  epok {epok:>6}  |r| ortanca {satir['r_ortanca']:.3f}  "
              f"en buyuk {satir['r_en_buyuk']:.3f}", flush=True)

    cikti = a.kosum / "bagdasim_egrisi.csv"
    with open(cikti, "w", newline="", encoding="utf-8") as f:
        y = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        y.writeheader()
        y.writerows(satirlar)
    print(f"\nyazildi: {cikti}")

    ilk, son = satirlar[0]["r_ortanca"], satirlar[-1]["r_ortanca"]
    print(f"\n{'=' * 60}")
    print(f"ilk anlik goruntu  |r| = {ilk:.3f}")
    print(f"son anlik goruntu  |r| = {son:.3f}")
    print(f"degisim            {son - ilk:+.3f}")
    print("kiyas: Faz 3 kontrolu, on egitimli referans modeli  |r| ~ 0,84")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f, eksen = plt.subplots(figsize=(7.5, 4.6))
        eksen.plot([s["pencere"] / 1000 for s in satirlar],
                   [s["r_ortanca"] for s in satirlar], "o-", label="|r| ortanca")
        eksen.plot([s["pencere"] / 1000 for s in satirlar],
                   [s["r_en_buyuk"] for s in satirlar], "s--", alpha=.6,
                   label="|r| en buyuk")
        eksen.axhline(0.84, color="tab:green", ls=":",
                      label="on egitimli referans (Faz 3 kontrolu)")
        eksen.set_xlabel("gorulen pencere (bin)")
        eksen.set_ylabel("|bagdasim|")
        eksen.set_title("Bagdasim egitim olcegiyle nasil degisiyor")
        eksen.set_ylim(0, 1)
        eksen.grid(alpha=.3)
        eksen.legend(fontsize=8)
        f.tight_layout()
        f.savefig(a.kosum / "bagdasim_egrisi.png", dpi=130)
        print(f"yazildi: {a.kosum / 'bagdasim_egrisi.png'}")
    except ImportError:
        print("matplotlib yok, grafik atlandi")

    (a.kosum / "bagdasim_egrisi.json").write_text(
        json.dumps({"kume": a.kume, "tarama": len(taramalar),
                    "noktalar": satirlar}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
