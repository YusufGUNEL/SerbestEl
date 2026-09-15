"""Gorselleştirme araclari — bir kere yazilir, proje boyunca kullanilir.

Yol haritasi Faz 1 Adim 5'te dort arac istiyor:
  1. bir taramanin kare kare oynatilmasi          -> kare_animasyonu()
  2. prob yorungesinin 3B cizimi (gercek+tahmin)  -> yorunge_cizimi()
  3. kare basina hata grafigi                      -> hata_grafigi()
  4. hacmin kesit gorunumleri                      -> hacim_dilimleri()

Ek olarak suruklenme_benzetimi(): kare basina hata verildiginde kuresel
hatanin nasil buyudugunu gosterir. Faz 2'de model gelene kadar hata
grafigini besleyen sey bu; ayrica "kare basina ne kadar dogru olmam
gerekiyor" sorusunun cevabini veriyor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # pencere yok, dosyaya yazar
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402

KOK = Path(__file__).resolve().parents[1]
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from src.geometri import (  # noqa: E402
    Kalibrasyon,
    kuresel_donusumler,
    piksel_noktalari,
    yerel_donusumler,
)

# tek yerden renk ve stil; butun cizimler ayni dili konussun
RENK = {"gercek": "#0E6E78", "tahmin": "#C4622A", "ikincil": "#848E95",
        "vurgu": "#A63A32", "iyi": "#1D7A4C"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


# --------------------------------------------------------------- arac 1
def kare_animasyonu(kareler: np.ndarray, cikti: Path, kare_atla: int = 10,
                    fps: int = 12, baslik: str = "") -> Path:
    """Taramayi GIF olarak kaydeder. 1570 kareyi hepsini koymak gereksiz;
    kare_atla ile seyreltilir."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    secili = kareler[::kare_atla]
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(secili[0], cmap="gray", vmin=0, vmax=255)
    yazi = ax.set_title(f"{baslik}  kare 0", fontsize=9)

    def kare_ver(i):
        im.set_data(secili[i])
        yazi.set_text(f"{baslik}  kare {i * kare_atla}")
        return im, yazi

    anim = FuncAnimation(fig, kare_ver, frames=len(secili), interval=1000 / fps,
                         blit=False)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    anim.save(cikti, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return cikti


# --------------------------------------------------------------- arac 2
def prob_konumlari(donusumler: torch.Tensor, kalib: Kalibrasyon,
                   h: int = 480, w: int = 640) -> np.ndarray:
    """Her karenin merkezinin ilk kare mm uzayindaki konumu. Cikti (N,3)."""
    merkez = torch.tensor([[(1 + w) / 2], [(1 + h) / 2], [0.0], [1.0]],
                          dtype=donusumler.dtype)
    mm = kalib.olcek.to(donusumler.dtype) @ merkez
    return (donusumler @ mm[None])[:, 0:3, 0].numpy()


def yorunge_cizimi(gercek: np.ndarray, cikti: Path,
                   tahmin: np.ndarray | None = None, baslik: str = "") -> Path:
    """Prob yorungesinin 3B cizimi. Tahmin verilirse ust uste bindirir.

    Uc ayri gorunum de veriliyor (XY, XZ, YZ): 3B cizimde suruklenmenin
    hangi eksende oldugunu gormek zordur, duzlem gorunumleri onu ayirir.
    """
    fig = plt.figure(figsize=(9.5, 5.2))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot(*gercek.T, color=RENK["gercek"], lw=1.4, label="gercek")
    ax3.scatter(*gercek[0], color=RENK["iyi"], s=28, label="baslangic")
    ax3.scatter(*gercek[-1], color=RENK["vurgu"], s=28, label="bitis")
    if tahmin is not None:
        ax3.plot(*tahmin.T, color=RENK["tahmin"], lw=1.4, ls="--", label="tahmin")
        ax3.scatter(*tahmin[-1], color=RENK["tahmin"], s=28, marker="x")
    ax3.set_xlabel("x (mm)"); ax3.set_ylabel("y (mm)"); ax3.set_zlabel("z (mm)")
    ax3.set_title(f"{baslik} prob yorungesi", fontsize=9)
    ax3.legend(fontsize=7, loc="upper left")

    eksen_adi = ["x", "y", "z"]
    for k, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = fig.add_subplot(3, 2, 2 * k + 2)
        ax.plot(gercek[:, a], gercek[:, b], color=RENK["gercek"], lw=1.1)
        ax.scatter(gercek[0, a], gercek[0, b], color=RENK["iyi"], s=16, zorder=3)
        if tahmin is not None:
            ax.plot(tahmin[:, a], tahmin[:, b], color=RENK["tahmin"], lw=1.1, ls="--")
        # esit en-boy: mesafeler karsilastirilabilir olmali, yoksa bir eksendeki
        # suruklenme digerine gore buyuk/kucuk gorunur
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"{eksen_adi[a]} (mm)", fontsize=7.5, labelpad=1)
        ax.set_ylabel(f"{eksen_adi[b]} (mm)", fontsize=7.5, labelpad=1)
        # gercek veri araligini yaz: esit en-boy limitleri genislettigi icin
        # cizimden okunan aralik yaniltici olabilir
        ax.set_title(
            f"{eksen_adi[a]}: {gercek[:, a].ptp():.0f} mm   "
            f"{eksen_adi[b]}: {gercek[:, b].ptp():.0f} mm",
            fontsize=7, pad=2, color=RENK["ikincil"])
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, bbox_inches="tight")
    plt.close(fig)
    return cikti


# --------------------------------------------------------------- arac 3
def hata_grafigi(seriler: dict[str, np.ndarray], cikti: Path,
                 baslik: str = "", y_etiket: str = "hata (mm)",
                 log: bool = False) -> Path:
    """Kare numarasina karsi hata. Suruklenmenin nerede basladigini gosterir.

    Iki panel: ustte hatanin kendisi, altta kareler arasi ARTIS. Artis paneli
    onemli — hata duzgun mu buyuyor, yoksa belirli anlarda siciriyor mu?
    Sicrama varsa Faz 3'te o anlarda ne oldugu aranacak.
    """
    fig, (ust, alt) = plt.subplots(2, 1, figsize=(7.6, 4.8), sharex=True,
                                   height_ratios=[2, 1])
    renkler = [RENK["gercek"], RENK["tahmin"], RENK["ikincil"], RENK["vurgu"]]
    for (ad, y), renk in zip(seriler.items(), renkler * 4):
        x = np.arange(1, len(y) + 1)
        ust.plot(x, y, label=ad, color=renk, lw=1.2)
        artis = np.diff(y, prepend=y[0])
        alt.plot(x, artis, color=renk, lw=0.8, alpha=0.85)
    if log:
        ust.set_yscale("log")
    ust.set_ylabel(y_etiket)
    ust.set_title(f"{baslik} kare basina hata", fontsize=9)
    ust.legend(fontsize=8)
    alt.axhline(0, color=RENK["ikincil"], lw=0.7)
    alt.set_ylabel("kare basi artis")
    alt.set_xlabel("kare numarasi")
    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, bbox_inches="tight")
    plt.close(fig)
    return cikti


# --------------------------------------------------------------- arac 4
def hacim_dilimleri(hacim, cikti: Path, baslik: str = "") -> Path:
    """Hacmin uc dik kesiti + doluluk haritasi.

    Dolu voksel sayisinin en yuksek oldugu dilimler secilir; ortadan kesmek
    doner taramada bos bolgeye denk gelebiliyor.
    """
    ort, say = hacim.ortalama, hacim.sayim
    dolu = say > 0
    z = int(np.argmax(dolu.sum(axis=(1, 2))))
    y = int(np.argmax(dolu.sum(axis=(0, 2))))
    x = int(np.argmax(dolu.sum(axis=(0, 1))))

    # PENCERELEME: ultrason parlakligi dusuk degerlerde yogunlasiyor, 0-255
    # araliginin ust yarisi neredeyse bos. Ham aralikla cizilince kesitler
    # simsiyah cikiyor. Dolu vokseller uzerinden yuzdelik alip pencereliyoruz.
    dolu_deger = ort[dolu]
    alt_p, ust_p = np.percentile(dolu_deger, [2, 99.5])
    fig, eksenler = plt.subplots(2, 2, figsize=(8.4, 7.0))
    v = hacim.voksel_mm
    nz, ny, nx = ort.shape
    panolar = [
        (ort[z], f"z = {hacim.kok_mm[2] + z * v:.0f} mm (eksenel)", nx * v, ny * v),
        (ort[:, y], f"y = {hacim.kok_mm[1] + y * v:.0f} mm (koronal)", nx * v, nz * v),
        (ort[:, :, x], f"x = {hacim.kok_mm[0] + x * v:.0f} mm (sagital)", ny * v, nz * v),
    ]
    for ax, (dilim, ad, gen, yuk) in zip(eksenler.ravel(), panolar):
        ax.grid(False)
        ax.imshow(dilim, cmap="gray", vmin=alt_p, vmax=ust_p, origin="lower",
                  extent=[0, gen, 0, yuk], aspect="equal",
                  interpolation="nearest")
        ax.set_title(ad, fontsize=8.5)
        ax.set_xlabel("mm", fontsize=7.5); ax.tick_params(labelsize=7)

    ax = eksenler.ravel()[3]
    ax.grid(False)
    kapsama = say[z].astype(np.float32)
    g = ax.imshow(kapsama, cmap="magma", origin="lower",
                  extent=[0, nx * v, 0, ny * v], aspect="equal")
    ax.set_title(f"ayni dilimde kapsama (voksele dusen piksel)", fontsize=8.5)
    ax.set_xlabel("mm", fontsize=7.5); ax.tick_params(labelsize=7)
    fig.colorbar(g, ax=ax, fraction=0.046)

    fig.suptitle(f"{baslik}  {nx}x{ny}x{nz} voksel @ {v} mm, "
                 f"doluluk {hacim.doluluk:.1%}  "
                 f"(parlaklik penceresi {alt_p:.0f}-{ust_p:.0f})", fontsize=9)
    fig.tight_layout()
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, bbox_inches="tight")
    plt.close(fig)
    return cikti


# ------------------------------------------------------- suruklenme benzetimi
def suruklenme_benzetimi(tforms: torch.Tensor, kalib: Kalibrasyon,
                         aci_std_derece: float = 0.0, oteleme_std_mm: float = 0.0,
                         aci_yanlilik_derece: float = 0.0,
                         oteleme_yanlilik_mm: float = 0.0,
                         tohum: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Kare basina hata verildiginde kuresel hata nasil buyuyor.

    Modelin yaptigi is: her kare cifti icin YEREL donusumu tahmin etmek.
    Sonra bunlar zincirleme carpilarak kuresel konum bulunuyor. Burada gercek
    yerel donusumlere kucuk hata ekleyip zinciri kuruyoruz.

    GURULTU ve YANLILIK AYRI PARAMETRE — cunku etkileri ayni degil:
      std      sifir ortalamali rastgele hata. Zincirde kismen birbirini goturur.
      yanlilik her karede AYNI yonde sapma. Zincirde toplanir, goturmez.
    Olculen fark buyuk: yansiz 0,15 der/mm -> 1569. karede 8 mm ve plato;
    yanli 0,02 der/mm (7,5 kat kucuk) -> 78 mm ve dogrusal buyume.
    Yani onemli olan hatanin buyuklugu degil, yanliligi.

    Dondurur: (kuresel_hata_mm, yerel_hata_mm), her biri (N-1,)
    """
    import pytorch3d.transforms as t3

    yerel_gercek = yerel_donusumler(tforms, kalib)
    kuresel_gercek = kuresel_donusumler(tforms, kalib)
    n = yerel_gercek.shape[0]

    g = torch.Generator().manual_seed(tohum)
    aci = (torch.randn(n, 3, generator=g) * np.deg2rad(aci_std_derece)
           + np.deg2rad(aci_yanlilik_derece))
    ote = torch.randn(n, 3, generator=g) * oteleme_std_mm + oteleme_yanlilik_mm
    bozma = torch.eye(4).repeat(n, 1, 1)
    bozma[:, 0:3, 0:3] = t3.euler_angles_to_matrix(aci, "ZYX")
    bozma[:, 0:3, 3] = ote
    yerel_tahmin = bozma @ yerel_gercek

    # zincirleme: kuresel konum yerel tahminlerin carpimi
    p = piksel_noktalari(480, 640, yogunluk=(9, 9))
    mm = kalib.olcek @ p
    birikim = torch.eye(4)
    kuresel_hata, yerel_hata = [], []
    for i in range(n):
        birikim = birikim @ yerel_tahmin[i]
        t_k = (birikim @ mm)[0:3]
        g_k = (kuresel_gercek[i] @ mm)[0:3]
        kuresel_hata.append((t_k - g_k).norm(dim=0).mean().item())
        t_y = (yerel_tahmin[i] @ mm)[0:3]
        g_y = (yerel_gercek[i] @ mm)[0:3]
        yerel_hata.append((t_y - g_y).norm(dim=0).mean().item())
    return np.array(kuresel_hata), np.array(yerel_hata)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", type=Path, default=KOK / "data")
    ap.add_argument("--denek", default="050")
    ap.add_argument("--tarama", default="LH_rotation")
    ap.add_argument("--cikti", type=Path, default=KOK / "results" / "faz1_gorsel")
    ap.add_argument("--voksel-mm", type=float, default=0.6)
    ap.add_argument("--kare-atla", type=int, default=2)
    a = ap.parse_args()

    from src.hacim import hacim_kur, tarama_oku

    etiket = f"{a.denek}/{a.tarama}"
    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    kareler, tforms = tarama_oku(a.veri, a.denek, a.tarama)
    print(f"{etiket}: {kareler.shape[0]} kare")
    a.cikti.mkdir(parents=True, exist_ok=True)
    ad = f"{a.denek}_{a.tarama}"

    print("1/4 kare animasyonu")
    print("   ", kare_animasyonu(kareler, a.cikti / f"{ad}_tarama.gif",
                                 kare_atla=20, baslik=etiket).name)

    print("2/4 prob yorungesi")
    donusumler = kuresel_donusumler(tforms, kalib, ilk_kare_dahil=True)
    konum = prob_konumlari(donusumler, kalib)
    print("   ", yorunge_cizimi(konum, a.cikti / f"{ad}_yorunge.png",
                                baslik=etiket).name)

    print("3/4 suruklenme benzetimi + hata grafigi")
    # Yansiz gurultu ile yanliligi YAN YANA koyuyoruz: ayni grafikte
    # gorulmezse aralarindaki buyuklugu farki fark edilmiyor.
    senaryolar = [
        ("yansiz 0.05", dict(aci_std_derece=0.05, oteleme_std_mm=0.05)),
        ("yansiz 0.15", dict(aci_std_derece=0.15, oteleme_std_mm=0.15)),
        ("yansiz 0.40", dict(aci_std_derece=0.40, oteleme_std_mm=0.40)),
        ("YANLI 0.02", dict(aci_yanlilik_derece=0.02, oteleme_yanlilik_mm=0.02)),
    ]
    seriler = {}
    for etiket_s, kw in senaryolar:
        kuresel, yerel = suruklenme_benzetimi(tforms, kalib, **kw)
        seriler[f"{etiket_s} (kare basi {yerel.mean():.2f} mm)"] = kuresel
        print(f"    {etiket_s:12s}: yerel {yerel.mean():.3f} mm -> "
              f"son karede kuresel {kuresel[-1]:6.1f} mm")
    print("   ", hata_grafigi(seriler, a.cikti / f"{ad}_suruklenme.png",
                              baslik=f"{etiket} benzetim:",
                              log=True).name)

    print("4/4 hacim dilimleri")
    hacim = hacim_kur(kareler, tforms, kalib, voksel_mm=a.voksel_mm,
                      kare_atla=a.kare_atla, ilerleme=False)
    print("   ", hacim_dilimleri(hacim, a.cikti / f"{ad}_hacim.png",
                                 baslik=etiket).name)
    print(f"\nhepsi: {a.cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
