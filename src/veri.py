"""Veri yukleyici — uc farkli dizin duzenini ve degisken tarama sayisini kaldirir.

NEDEN REFERANSIN YUKLEYICISI YETMIYOR
  `utils/loader.py` iki sey varsayiyor:
    1) Denek basina TAM IKI tarama (`if len(scans) != 2: raise`)
    2) `<kok>/<denek>/<tarama>.h5` duzeni, kareler ve donusumler AYNI dosyada

  Olculen gercek: bu varsayimlar elimizdeki uc kumeden yalnizca birinde tutuyor.

  | Kume                  | Duzen                                  | Tarama/denek |
  |-----------------------|----------------------------------------|--------------|
  | TUS-REC2025 egitim    | `frames_transfs/<denek>/<tarama>.h5`   | 2            |
  | TUS-REC2025 dogrulama | `frames/` ve `transfs/` AYRI dizinler  | 2            |
  | TUS-REC2024           | `<denek>/<tarama>.h5` (sarmalayici yok)| 24           |

  TUS-REC2024'te tarama adlari sistematik:
    {LH,RH} x {Par,Per} x {C,L,S} x {DtP,PtD}
    kol x prob yonu x yorunge sekli x tarama yonu
  2024 taramalari ayrica cok daha kisa (yuzlerce kare; 2025'te ~1500).

ORNEKLEME REFERANSLA AYNI
  `frame_sampler` referanstaki ile birebir ayni mantik: [0, n-sample_range]
  araligindan rastgele bir baslangic, oradan sample_range genisliginde
  pencerede num_samples kare, sirali. Faz 2'nin amaci referansi uretmek,
  ornekleme degistirilirse sayilar karsilastirilamaz.

REFERANSTAKI DOSYA SIZINTISI DUZELTILDI
  Referans `__getitem__` her cagrida h5 dosyasini aciyor ve HIC KAPATMIYOR.
  Kisa kosumda fark etmiyor; gunlerce surecek egitimde isci surec basina
  binlerce acik tanitici birikir. Burada dosya `with` icinde aciliyor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class Tarama:
    denek: str
    ad: str
    kare_yolu: Path
    tform_yolu: Path          # ayri duzende farkli, birlesikte ayni dosya

    @property
    def kimlik(self) -> str:
        return f"{self.denek}/{self.ad}"


def duzeni_coz(kok: Path) -> tuple[str, Path, Path | None]:
    """(duzen_adi, kare_koku, tform_koku). tform_koku None ise ayni dosyada."""
    if (kok / "frames_transfs").is_dir():
        return "birlesik", kok / "frames_transfs", None
    if (kok / "frames").is_dir() and (kok / "transfs").is_dir():
        return "ayri", kok / "frames", kok / "transfs"
    if any(d.is_dir() and any(d.glob("*.h5")) for d in kok.iterdir()):
        return "duz", kok, None
    raise SystemExit(f"taninmayan veri duzeni: {kok}")


def taramalari_bul(kok: Path) -> tuple[str, list[Tarama]]:
    duzen, kare_kok, tform_kok = duzeni_coz(kok)
    taramalar = []
    for denek_dizin in sorted(d for d in kare_kok.iterdir() if d.is_dir()):
        for dosya in sorted(denek_dizin.glob("*.h5")):
            taramalar.append(Tarama(
                denek=denek_dizin.name,
                ad=dosya.stem,
                kare_yolu=dosya,
                tform_yolu=(tform_kok / denek_dizin.name / dosya.name
                            if tform_kok else dosya),
            ))
    if not taramalar:
        raise SystemExit(f"tarama bulunamadi: {kok}")
    return duzen, taramalar


class TaramaKumesi(torch.utils.data.Dataset):
    """Bir kume denegin taramalari. Referansla ayni ornekleme.

    num_samples=-1 ise taramanin TAMAMI dondurulur (olcum icin).
    """

    def __init__(self, taramalar: list[Tarama], num_samples: int = 2,
                 sample_range: int | None = None):
        if num_samples != -1 and num_samples < 2:
            raise ValueError("num_samples >= 2 ya da -1 olmali")
        self.taramalar = list(taramalar)
        self.num_samples = num_samples
        self.sample_range = sample_range if sample_range is not None else num_samples
        self._uzunluk: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.taramalar)

    def kare_sayisi(self, i: int) -> int:
        t = self.taramalar[i]
        if t.kimlik not in self._uzunluk:
            with h5py.File(t.kare_yolu, "r") as f:
                self._uzunluk[t.kimlik] = f["frames"].shape[0]
        return self._uzunluk[t.kimlik]

    def kare_ornekle(self, n: int) -> list[int]:
        """Referans `frame_sampler` ile birebir ayni mantik."""
        if self.sample_range > n:
            raise ValueError(f"sample_range {self.sample_range} > tarama uzunlugu {n}")
        n0 = random.randint(0, n - self.sample_range)
        idx = random.sample(range(n0, n0 + self.sample_range), self.num_samples)
        idx.sort()
        return idx

    def __getitem__(self, i: int):
        t = self.taramalar[i]
        # referanstan farkli: dosyalar `with` ile aciliyor ve kapaniyor
        if self.num_samples == -1:
            with h5py.File(t.kare_yolu, "r") as f:
                kareler = f["frames"][()]
            with h5py.File(t.tform_yolu, "r") as f:
                tforms = f["tforms"][()]
        else:
            with h5py.File(t.kare_yolu, "r") as f:
                n = f["frames"].shape[0]
                sec = self.kare_ornekle(n)
                kareler = f["frames"][sec]
            with h5py.File(t.tform_yolu, "r") as f:
                tforms = f["tforms"][sec]
        # Kareler uint8 DONDURULUYOR, float'a burada cevrilmiyor: float32'ye
        # cevirmek veriyi 4 katina cikarir ve o hali isci surecten ana surece,
        # oradan GPU'ya kopyalanir. Bolme (/255) GPU'da yapiliyor.
        return (torch.from_numpy(np.asarray(kareler)),
                torch.from_numpy(np.asarray(tforms)).float(),
                t.denek, t.ad)


def kume_kur(kok: Path, denekler: list[str], num_samples: int = 2,
             sample_range: int | None = None,
             min_kare: int | None = None) -> TaramaKumesi:
    """Verilen deneklerin taramalarindan kume kurar.

    min_kare verilirse daha kisa taramalar ATILIR ve kac tanesi atildigi
    yazdirilir. TUS-REC2024'te taramalar cok kisa olabiliyor; ornekleme
    penceresi sigmayan tarama egitimde hata verir.
    """
    _, tum = taramalari_bul(kok)
    istenen = set(denekler)
    secili = [t for t in tum if t.denek in istenen]
    eksik = istenen - {t.denek for t in secili}
    if eksik:
        raise SystemExit(f"veride bulunmayan denekler: {sorted(eksik)}")

    atilan = 0
    if min_kare:
        gecici = TaramaKumesi(secili, num_samples, sample_range)
        tutulan = []
        for i, t in enumerate(secili):
            if gecici.kare_sayisi(i) >= min_kare:
                tutulan.append(t)
            else:
                atilan += 1
        secili = tutulan
    if not secili:
        raise SystemExit("secilen deneklerde kullanilabilir tarama kalmadi")
    if atilan:
        print(f"    {atilan} tarama {min_kare} kareden kisa oldugu icin atildi")
    return TaramaKumesi(secili, num_samples, sample_range)


def ozet(kok: Path) -> None:
    """Veri kumesinin yapisini yazdirir — yeni bir kumeyle calismadan once kos."""
    duzen, taramalar = taramalari_bul(kok)
    denekler = sorted({t.denek for t in taramalar})
    adlar = sorted({t.ad for t in taramalar})
    basina = len(taramalar) / len(denekler)
    print(f"duzen        : {duzen}")
    print(f"denek        : {len(denekler)}  ({denekler[0]} ... {denekler[-1]})")
    print(f"tarama       : {len(taramalar)}  (denek basina {basina:.1f})")
    print(f"tarama adlari: {len(adlar)} benzersiz")
    for a in adlar[:6]:
        print(f"    {a}")
    if len(adlar) > 6:
        print(f"    ... +{len(adlar)-6} tane")

    kume = TaramaKumesi(taramalar, num_samples=2)
    ornek = min(len(taramalar), 40)
    uzunluk = [kume.kare_sayisi(i) for i in range(ornek)]
    print(f"kare sayisi  : {ornek} tarama ornegi -> en az {min(uzunluk)}, "
          f"ortanca {int(np.median(uzunluk))}, en cok {max(uzunluk)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="veri kumesinin yapisini yazdirir")
    ap.add_argument("kok", type=Path)
    ozet(ap.parse_args().kok)
