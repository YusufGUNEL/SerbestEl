"""Denek bazli egitim/dogrulama/test bolmesi — bir kere uretilir, SABITLENIR.

EN KRITIK KURAL
  Ayni denegin bir taramasi egitimde, digeri testte OLAMAZ. Ayni kolun dokusu
  ikisinde de var demektir; model dokuyu ezberler, skor siser, sonuc yalan olur.
  Bolme DENEGE gore yapilir, taramaya gore degil.

NEDEN DOSYAYA YAZILIYOR
  Her deney ayni bolmeyi kullanmali. Bolme her kosuda yeniden uretilirse
  deneyler birbiriyle karsilastirilamaz — bir modelin digerinden iyi cikmasi
  gercek bir fark mi, yoksa sans eseri kolay bir test kumesi mi, bilinemez.
  Bu yuzden `configs/bolme.json` bir kere yazilir ve bir daha DEGISMEZ.
  Betik mevcut dosyanin uzerine sessizce yazmaz; --yeniden-uret istemek gerekir.

REFERANSTAN FARKI
  Referans `train.py` her kosuda `partition_by_ratio(randomise=True)` cagirip
  5 kat uretiyor ve `fold_XX.json` yaziyor. Ayni tohumla ayni sonucu verir, ama
  bolme denek KLASOR SIRASINA bagli: veri dizinine bir denek eklenirse ya da
  cikarilirsa butun bolme kayar. Burada bolmeyi denek KIMLIKLERIYLE saklıyoruz,
  sirayla degil; dizin degisse de bolme ayni kalir.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
if str(KOK) not in sys.path:      # betik olarak kosuldugunda depo koku eklenmeli
    sys.path.insert(0, str(KOK))
VARSAYILAN = KOK / "configs" / "bolme.json"


@dataclass
class Bolme:
    egitim: list[str]
    dogrulama: list[str]
    test: list[str]
    tohum: int
    kaynak: str

    def denek_kumesi(self, ad: str) -> list[str]:
        return {"egitim": self.egitim, "dogrulama": self.dogrulama,
                "test": self.test}[ad]

    def dogrula(self) -> None:
        """Ayni denek iki kumede olamaz — bu kontrol sessizce atlanamaz."""
        kumeler = {"egitim": set(self.egitim), "dogrulama": set(self.dogrulama),
                   "test": set(self.test)}
        adlar = list(kumeler)
        for i in range(len(adlar)):
            for j in range(i + 1, len(adlar)):
                ortak = kumeler[adlar[i]] & kumeler[adlar[j]]
                if ortak:
                    raise ValueError(
                        f"VERI SIZINTISI: {adlar[i]} ve {adlar[j]} kumelerinde "
                        f"ayni denekler var: {sorted(ortak)}")
        for ad, k in kumeler.items():
            if len(k) != len(self.denek_kumesi(ad)):
                raise ValueError(f"{ad} kumesinde tekrar eden denek var")

    def __repr__(self) -> str:
        return (f"Bolme(egitim {len(self.egitim)}, dogrulama {len(self.dogrulama)}, "
                f"test {len(self.test)} denek, tohum {self.tohum})")


def denekleri_bul(veri_kok: Path) -> tuple[list[str], str, int]:
    """Denekleri bulur. Duzen tanimayi `src/veri.py` yapiyor — tek yerden.

    Dondurur: (denekler, duzen_adi, tarama_sayisi)
    """
    from src.veri import taramalari_bul

    duzen, taramalar = taramalari_bul(veri_kok)
    return sorted({t.denek for t in taramalar}), duzen, len(taramalar)


def bolme_uret(denekler: list[str], oranlar=(0.6, 0.2, 0.2),
               tohum: int = 20260915, kaynak: str = "") -> Bolme:
    if len(denekler) < 3:
        raise SystemExit(f"en az 3 denek gerekir, {len(denekler)} bulundu")
    karisik = sorted(denekler)          # once sabit sira: tohum tek belirleyici olsun
    random.Random(tohum).shuffle(karisik)

    n = len(karisik)
    n_egitim = max(1, round(n * oranlar[0]))
    n_dogrulama = max(1, round(n * oranlar[1]))
    # kalan teste; her kume en az 1 denek almali
    if n_egitim + n_dogrulama >= n:
        n_egitim = max(1, n - 2)
        n_dogrulama = 1

    b = Bolme(
        egitim=sorted(karisik[:n_egitim]),
        dogrulama=sorted(karisik[n_egitim:n_egitim + n_dogrulama]),
        test=sorted(karisik[n_egitim + n_dogrulama:]),
        tohum=tohum,
        kaynak=kaynak,
    )
    b.dogrula()
    return b


def yukle(yol: Path = VARSAYILAN) -> Bolme:
    if not yol.exists():
        raise SystemExit(
            f"bolme dosyasi yok: {yol}\n"
            f"Once uret:  python src/bolme.py --veri <dizin>")
    b = Bolme(**json.loads(yol.read_text(encoding="utf-8")))
    b.dogrula()
    return b


def kaydet(b: Bolme, yol: Path = VARSAYILAN) -> None:
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(asdict(b), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", type=Path, required=True,
                    help="denek klasorlerini iceren dizin")
    ap.add_argument("--cikti", type=Path, default=VARSAYILAN)
    ap.add_argument("--tohum", type=int, default=20260915)
    ap.add_argument("--oranlar", type=float, nargs=3, default=(0.6, 0.2, 0.2),
                    metavar=("EGITIM", "DOGRULAMA", "TEST"))
    ap.add_argument("--yeniden-uret", action="store_true",
                    help="mevcut bolmenin uzerine yaz (deneyleri karsilastirilamaz kilar)")
    a = ap.parse_args()

    if a.cikti.exists() and not a.yeniden_uret:
        mevcut = yukle(a.cikti)
        print(f"bolme zaten var: {a.cikti}")
        print(f"  {mevcut}")
        print("  egitim    :", ", ".join(mevcut.egitim))
        print("  dogrulama :", ", ".join(mevcut.dogrulama))
        print("  test      :", ", ".join(mevcut.test))
        print("\nDEGISTIRILMEDI. Uzerine yazmak icin --yeniden-uret ver;")
        print("ama o zaman onceki butun olcumler karsilastirilamaz hale gelir.")
        return 0

    denekler, duzen, tarama_sayisi = denekleri_bul(a.veri)
    b = bolme_uret(denekler, tuple(a.oranlar), a.tohum, kaynak=str(a.veri))
    kaydet(b, a.cikti)
    print(f"{duzen} duzen, {len(denekler)} denek, {tarama_sayisi} tarama "
          f"(denek basina {tarama_sayisi/len(denekler):.1f}) -> {b}")
    print("  egitim    :", ", ".join(b.egitim))
    print("  dogrulama :", ", ".join(b.dogrulama))
    print("  test      :", ", ".join(b.test))
    print(f"\nyazildi: {a.cikti}")
    print("Bu dosya artik SABIT. Butun deneyler bunu kullanacak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
