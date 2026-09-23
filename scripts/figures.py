"""README gorsellerini uretir (Ingilizce etiketli) -> docs/figures/.

Depodaki README Ingilizce; gorseller de onunla ayni dilde olmali. Butun
sayilar kosumlarin kendi dosyalarindan okunuyor, elle yazilan sayi yok.

  coherence_ladder.png    Faz 5 merdiveninin bagdasim egrileri
  phase_transition.png    U_uzun'da cokusten cikis (epok 200-250)
  demo.png / demo.gif     videodan bir kare ve karsilastirma sahnesi onizlemesi
                          (video once scripts/video.py ile uretilmis olmali)

KULLANIM
  python scripts/figures.py
  python scripts/figures.py --video results/faz6/U_uzun_C/video/018_LH_Par_C_PtD.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.gorsel import RENK  # noqa: E402  (ayni stil ayarlarini da yukler)

CIKTI = KOK / "docs" / "figures"


def egri(dizin: Path) -> list[dict]:
    return json.loads((dizin / "bagdasim_egrisi.json").read_text())["noktalar"]


def merdiven() -> Path:
    satirlar = [("A1_referans", "A1 baseline config", RENK["ikincil"]),
                ("A2_C", "A2 + parameter-space loss", RENK["tahmin"]),
                ("A3_CB", "A3 + 6D rotation", RENK["gercek"]),
                ("A4_CBG", "A4 + ImageNet backbone", RENK["iyi"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for ad, etiket, renk in satirlar:
        n = egri(KOK / "results" / "faz5" / ad)
        ax.plot([p["epok"] for p in n], [p["r_ortanca"] for p in n], "-o", ms=3,
                color=renk, label=etiket)
    ax.axhline(0.84, ls="--", lw=0.8, color=RENK["vurgu"])
    ax.text(5, 0.80, "pretrained reference model (leaky upper bound) 0.84",
            fontsize=7, color=RENK["vurgu"], va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|r|  (median over 6 motion components)")
    ax.set_ylim(0, 0.9)
    ax.set_title("Prediction–motion correlation during training — ablation ladder, 180 min each",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False)
    fig.tight_layout()
    yol = CIKTI / "coherence_ladder.png"
    fig.savefig(yol)
    plt.close(fig)
    return yol


def faz_gecisi() -> Path:
    n = egri(KOK / "results" / "faz4" / "U_uzun")
    ep = [p["epok"] for p in n]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(200, 250, color=RENK["vurgu"], alpha=0.10, lw=0)
    ax.plot(ep, [p["r_en_buyuk"] for p in n], "-o", ms=3, color=RENK["gercek"],
            label="max |r| (best component)")
    ax.plot(ep, [p["r_ortanca"] for p in n], "-o", ms=3, color=RENK["tahmin"],
            label="median |r|")
    ax.text(225, 0.40, "collapse\nends", ha="center", fontsize=7, color=RENK["vurgu"])
    ax.axvline(115, ls=":", lw=0.9, color=RENK["ikincil"])
    ax.text(118, 0.62, "60-min budget\nstops here", fontsize=7, color=RENK["ikincil"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("|r|")
    ax.set_ylim(0, 1.02)
    ax.set_title("U_uzun (8 h): the model leaves regression collapse in a phase transition",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=False)
    fig.tight_layout()
    yol = CIKTI / "phase_transition.png"
    fig.savefig(yol)
    plt.close(fig)
    return yol


def video_onizleme(video: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not video.exists():
        print(f"atlandi: ffmpeg ya da video yok ({video})")
        return []
    kare, gif = CIKTI / "demo.png", CIKTI / "demo.gif"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", "20", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=960:-1", str(kare)], check=True)
    # karsilastirma sahnesi (45-60 sn): 12 sn, 10 kare/sn, 720 px. GIF'i PIL
    # kuruyor: her ffmpeg derlemesinde palettegen filtresi yok (olculdu).
    import numpy as np
    from PIL import Image
    gen, yuk = 720, 405
    ham = subprocess.run([ffmpeg, "-v", "error", "-ss", "46", "-t", "12", "-i", str(video),
                          "-vf", f"fps=10,scale={gen}:{yuk}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    kareler = np.frombuffer(ham, np.uint8).reshape(-1, yuk, gen, 3)
    # tek ortak palet: kareler arasi titreme olmasin
    palet = Image.fromarray(np.concatenate(kareler[::10], axis=0)).quantize(
        colors=96, method=Image.Quantize.MEDIANCUT)
    resimler = [Image.fromarray(k).quantize(palette=palet, dither=Image.Dither.NONE)
                for k in kareler]
    resimler[0].save(gif, save_all=True, append_images=resimler[1:], duration=100,
                     loop=0, optimize=True)
    return [kare, gif]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", type=Path,
                    default=KOK / "results/faz6/U_uzun_C/video/018_LH_Par_C_PtD.mp4")
    a = ap.parse_args()
    CIKTI.mkdir(parents=True, exist_ok=True)
    for yol in [merdiven(), faz_gecisi(), *video_onizleme(a.video)]:
        print(f"yazildi: {yol.relative_to(KOK)}  ({yol.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
