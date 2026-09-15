"""Gercek referans donusumlerle 3B hacim kurma — Faz 1 kontrol adimi.

NE YAPIYOR, NE YAPMIYOR
  Model yok, tahmin yok. Yalnizca optik izleyiciden gelen gercek donusumler
  kullanilarak kareler uzaya yerlestiriliyor. Cikan hacim duzgunse koordinat
  matematigi dogrudur; bozuksa zincirde bir sira ya da ters hatasi var.

  Bu hacim ayni zamanda **TAVAN**: model en iyi ihtimalle buna yaklasir,
  gecemez. Faz 2'den sonraki her sonuc buna gore okunacak.

YONTEM — ileri serpme (forward splatting)
  Her karenin her pikseli, ilk karenin mm uzayinda bir noktaya gidiyor.
  O nokta hangi vokselin icine dusuyorsa vokselin toplamina parlaklik,
  sayacina 1 ekleniyor. Sonunda toplam/sayac = ortalama parlaklik.

  Neden ortalama: taramada ayni bolge birden fazla kareden geciliyor
  (doner protokol tam bunu yapiyor). Ortalama, ust uste gelen kareleri
  birlestirir; son yazan kazansin yaklasimi ise gurultulu ve sirayla
  degisen bir hacim verir.

  Bu yontem bosluk (hole) birakir: kareler arasi mesafe voksel boyundan
  buyukse aradaki vokseller hic doldurulmaz. Faz 1 icin sorun degil —
  amac dogrulama, gorsel kalite degil. Doldurma Faz 5'in isi.

BELLEK
  1570 kare x 307200 piksel = 482 milyon nokta. Hepsini birden tutmak
  float32 koordinatlarla ~5,8 GB eder. Bu yuzden kare bloklariyla
  ilerliyoruz; tepe bellek blok boyutuyla belirleniyor, tarama uzunluguyla
  degil.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

from src.geometri import Kalibrasyon, kuresel_donusumler, piksel_noktalari  # noqa: E402


@dataclass
class Hacim:
    """Voksel izgarasi ve onu mm uzayina baglayan bilgiler."""

    ortalama: np.ndarray      # (nz, ny, nx) float32, doldurulmayan voksel = nan
    sayim: np.ndarray         # (nz, ny, nx) int32, voksele dusen piksel sayisi
    kok_mm: np.ndarray        # (3,) izgaranin (x,y,z) kosesi, mm
    voksel_mm: float

    @property
    def doluluk(self) -> float:
        return float((self.sayim > 0).mean())

    def dunya_koordinati(self, eksen: int) -> np.ndarray:
        """Bir eksenin voksel merkezlerinin mm koordinatlari. eksen: 0=x,1=y,2=z."""
        n = self.ortalama.shape[2 - eksen]
        return self.kok_mm[eksen] + (np.arange(n) + 0.5) * self.voksel_mm

    def __repr__(self) -> str:
        nz, ny, nx = self.ortalama.shape
        return (f"Hacim({nx}x{ny}x{nz} voksel @ {self.voksel_mm} mm, "
                f"doluluk {self.doluluk:.1%})")


def _sinir_kutusu(donusumler: torch.Tensor, kalib: Kalibrasyon,
                  h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Butun karelerin kapladigi mm kutusu.

    Yalnizca dort kose noktasi yeterli: donusum afin, kare de dikdortgen,
    dolayisiyla butun pikseller koselerin olusturdugu kutunun icinde kalir.
    307200 nokta yerine 4 nokta ile ayni sonuc.
    """
    koseler = torch.tensor(
        [[1.0, w, 1.0, w], [1.0, 1.0, h, h], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    )
    mm = kalib.olcek @ koseler
    noktalar = (donusumler @ mm[None])[:, 0:3, :]          # (N,3,4)
    duz = noktalar.permute(1, 0, 2).reshape(3, -1).numpy()
    return duz.min(axis=1), duz.max(axis=1)


def hacim_kur(
    kareler: np.ndarray,
    tforms: torch.Tensor | None,
    kalib: Kalibrasyon,
    voksel_mm: float = 0.5,
    kare_atla: int = 1,
    blok: int = 32,
    ilerleme: bool = True,
    donusumler: torch.Tensor | None = None,
) -> Hacim:
    """Gercek donusumlerle hacim kurar.

    kareler     (N,H,W) uint8
    tforms      (N,4,4) izleyiciden gelen gercek donusumler
    kare_atla   >1 ise her n'inci kare kullanilir (hizli deneme icin)
    blok        ayni anda islenen kare sayisi; tepe bellegi bu belirler
    donusumler  (N,4,4) hazir kuresel donusumler; verilirse tforms yok sayilir.
                Capraz dogrulamada gerekli: kare cikarilirken referans kare
                degismemeli, yoksa karsilastirma anlamsiz olur.
    """
    if donusumler is None:
        if tforms is None:
            raise ValueError("tforms ya da donusumler verilmeli")
        if kareler.shape[0] != tforms.shape[0]:
            raise ValueError(
                f"kare sayisi uyusmuyor: {kareler.shape[0]} / {tforms.shape[0]}")
        secim = slice(None, None, kare_atla)
        kareler, tforms = kareler[secim], tforms[secim]
        # ilk kare dahil: o da hacme yazilmali (kimlik donusumuyle)
        donusumler = kuresel_donusumler(tforms, kalib, ilk_kare_dahil=True)
    else:
        if kareler.shape[0] != donusumler.shape[0]:
            raise ValueError(
                f"kare sayisi uyusmuyor: {kareler.shape[0]} / {donusumler.shape[0]}")
        if kare_atla != 1:
            kareler = kareler[::kare_atla]
            donusumler = donusumler[::kare_atla]

    n, h, w = kareler.shape
    alt, ust = _sinir_kutusu(donusumler, kalib, h, w)

    # izgara: kutuyu bir voksel payla sar, boylece kenardaki nokta tasmasin
    kok = alt - voksel_mm
    boyut = np.ceil((ust - kok + voksel_mm) / voksel_mm).astype(np.int64)
    nx, ny, nz = (int(b) for b in boyut)
    if nx * ny * nz > 600_000_000:
        raise MemoryError(
            f"izgara cok buyuk: {nx}x{ny}x{nz}. voksel_mm'i buyut."
        )

    toplam = np.zeros(nz * ny * nx, dtype=np.float64)
    sayim = np.zeros(nz * ny * nx, dtype=np.int64)
    p = piksel_noktalari(h, w)
    mm_yerel = (kalib.olcek @ p)                      # (4, H*W)

    for bas in range(0, n, blok):
        son = min(bas + blok, n)
        T = donusumler[bas:son]
        # (B,3,H*W) -> mm, ilk karenin uzayinda
        noktalar = (T @ mm_yerel[None])[:, 0:3, :].numpy()
        parlaklik = kareler[bas:son].reshape(son - bas, -1).astype(np.float64)

        # mm -> voksel indeksi
        idx = ((noktalar - kok[None, :, None]) / voksel_mm).astype(np.int64)
        gecerli = (
            (idx[:, 0] >= 0) & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < nz)
        )
        duz = idx[:, 2] * (ny * nx) + idx[:, 1] * nx + idx[:, 0]
        duz, parlaklik = duz[gecerli], parlaklik[gecerli]

        # bincount, np.add.at'ten cok daha hizli
        toplam += np.bincount(duz, weights=parlaklik, minlength=toplam.size)
        sayim += np.bincount(duz, minlength=sayim.size)

        if ilerleme:
            print(f"\r  kare {son}/{n}", end="", flush=True)
    if ilerleme:
        print()

    with np.errstate(invalid="ignore", divide="ignore"):
        ortalama = np.where(sayim > 0, toplam / np.maximum(sayim, 1), np.nan)

    return Hacim(
        ortalama=ortalama.reshape(nz, ny, nx).astype(np.float32),
        sayim=sayim.reshape(nz, ny, nx).astype(np.int32),
        kok_mm=kok.astype(np.float32),
        voksel_mm=voksel_mm,
    )


def tarama_oku(veri: Path, denek: str, tarama: str, kare_siniri: int | None = None):
    """Dogrulama seti duzeninden (frames/ ve transfs/ ayri) bir tarama okur."""
    import h5py

    with h5py.File(veri / "frames" / denek / f"{tarama}.h5", "r") as f:
        kareler = np.asarray(f["frames"][:kare_siniri])
    with h5py.File(veri / "transfs" / denek / f"{tarama}.h5", "r") as f:
        tforms = torch.tensor(np.asarray(f["tforms"][:kare_siniri]))
    return kareler, tforms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--veri", type=Path, default=KOK / "data")
    ap.add_argument("--denek", default="050")
    ap.add_argument("--tarama", default="LH_rotation")
    ap.add_argument("--voksel-mm", type=float, default=0.5)
    ap.add_argument("--kare-atla", type=int, default=1)
    ap.add_argument("--kare-siniri", type=int, default=None,
                    help="yalnizca ilk N kare (hizli deneme)")
    ap.add_argument("--cikti", type=Path, default=KOK / "results" / "faz1_hacim")
    a = ap.parse_args()

    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    kareler, tforms = tarama_oku(a.veri, a.denek, a.tarama, a.kare_siniri)
    print(f"{a.denek}/{a.tarama}: {kareler.shape[0]} kare, {kalib}")

    hacim = hacim_kur(kareler, tforms, kalib,
                      voksel_mm=a.voksel_mm, kare_atla=a.kare_atla)
    print(hacim)

    a.cikti.mkdir(parents=True, exist_ok=True)
    ad = f"{a.denek}_{a.tarama}_v{a.voksel_mm}"
    np.savez_compressed(
        a.cikti / f"{ad}.npz",
        ortalama=hacim.ortalama, sayim=hacim.sayim,
        kok_mm=hacim.kok_mm, voksel_mm=hacim.voksel_mm,
    )
    print(f"kaydedildi: {a.cikti / (ad + '.npz')}")

    dolu = hacim.sayim[hacim.sayim > 0]
    print(f"dolu voksel {dolu.size:,}, voksel basina piksel: "
          f"ortanca {np.median(dolu):.0f}, en fazla {dolu.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
