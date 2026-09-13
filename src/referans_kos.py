"""Referans TUS-REC2025 betiklerini Windows'ta, depoyu DEGISTIRMEDEN kosturur.

NEDEN GEREKLI
  Referans `train.py` ve `generate_DDF.py` duz betik: butun kod modul
  seviyesinde, `if __name__ == "__main__"` korumasi yok. Ama DataLoader
  `num_workers=8` ile kuruluyor.

  Linux'ta multiprocessing `fork` kullanir, cocuk surec ana modulu yeniden
  ICE AKTARMAZ, sorun gorunmez. Windows'ta `spawn` kullanilir: cocuk surec
  ana modulu bastan calistirir, o da yeni DataLoader kurup yeni surec
  dogurmaya kalkar ve Python bunu yakalayip hata verir:

      RuntimeError: An attempt has been made to start a new process before
      the current process has finished its bootstrapping phase.

  Yani bu referansin Windows'a ozgu bir kusuru, bizim kurulumumuzun degil.

COZUM
  Betigi calistirmadan once DataLoader'in num_workers'ini sabitliyoruz.
  0 ise hic ikincil surec dogmaz, koruma da gerekmez. Matematik aynen ayni
  kalir - yalnizca veri okuma tek surecte yapilir.

  num_workers > 0 gerektiginde referans betigi yetmez; korumali kendi
  egitim dongumuz gerekir (Faz 2).

KULLANIM
  python src/referans_kos.py train.py --NUM_EPOCHS 2 --MINIBATCH_SIZE 4
  python src/referans_kos.py generate_DDF.py
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"


def dataloader_isci_sabitle(isci: int) -> None:
    """DataLoader kurulurken num_workers'i zorla degistirir."""
    import torch.utils.data as tud

    orijinal = tud.DataLoader.__init__

    def yamali(self, *args, **kwargs):
        istenen = kwargs.get("num_workers")
        if istenen != isci:
            kwargs["num_workers"] = isci
            if isci == 0:
                # num_workers=0 ile bu ikisi anlamsiz, uyari uretirler
                kwargs.pop("prefetch_factor", None)
                kwargs["persistent_workers"] = False
        return orijinal(self, *args, **kwargs)

    tud.DataLoader.__init__ = yamali


def calisma_dizini_hazirla(dizin: Path) -> None:
    """Referans betikleri cwd'ye gore dosya arar; gerekenleri baglar.

    train.py on egitimli agirligi `os.getcwd()/TUS-REC2024_model/model_weights`
    yolundan okur. Depoyu kirletmemek icin cwd'yi ayri tutup agirliga
    bir baglanti veriyoruz.
    """
    dizin.mkdir(parents=True, exist_ok=True)
    for ad in ("TUS-REC2024_model",):
        hedef = dizin / ad
        kaynak = REFERANS / ad
        if hedef.exists() or not kaynak.exists():
            continue
        try:
            hedef.symlink_to(kaynak, target_is_directory=True)
        except OSError:
            # Windows'ta sembolik baglanti yetki isteyebilir; kavsak (junction)
            # istemez. Son care: kopyala.
            if os.name == "nt":
                os.system(f'mklink /J "{hedef}" "{kaynak}" >nul 2>&1')
            if not hedef.exists():
                import shutil

                shutil.copytree(kaynak, hedef)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("betik", help="referans deposundaki betik, or. train.py")
    ap.add_argument("--isci", type=int, default=0,
                    help="DataLoader num_workers (Windows'ta 0 olmali)")
    ap.add_argument("--calisma-dizini", type=Path, default=None,
                    help="cwd; ciktilar buraya yazilir (varsayilan: mevcut dizin)")
    ap.add_argument("-h", "--help", action="store_true")
    bizim, kalan = ap.parse_known_args()

    if bizim.help:
        print(__doc__)
        print("\n--- referans betigin kendi secenekleri icin: ---")
        print(f"  python {REFERANS / bizim.betik} --help")
        return 0

    betik = REFERANS / bizim.betik
    if not betik.exists():
        print(f"HATA: bulunamadi: {betik}", file=sys.stderr)
        return 2

    if bizim.calisma_dizini is not None:
        calisma_dizini_hazirla(bizim.calisma_dizini)
        os.chdir(bizim.calisma_dizini)
    calisma_dizini_hazirla(Path.cwd())

    if os.name == "nt" and bizim.isci != 0:
        print("UYARI: Windows'ta korumasiz betikle num_workers>0 coker. "
              "Yine de deneniyor.", file=sys.stderr)

    # referans deposu sys.path'te olmali: `from utils.loader import Dataset`
    sys.path.insert(0, str(REFERANS))
    dataloader_isci_sabitle(bizim.isci)

    # betigin kendi argparse'i sys.argv'yi okuyacak
    sys.argv = [str(betik)] + kalan

    print(f"--- referans: {bizim.betik}  (num_workers={bizim.isci})")
    print(f"--- cwd     : {Path.cwd()}")
    runpy.run_path(str(betik), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
