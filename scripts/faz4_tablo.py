"""Faz 4 deneylerini tek tabloda toplar — calismayanlar dahil.

NEDEN AYRI BIR BETIK
  Faz 4'un ciktisi tek bir iyi sayi degil, KARSILASTIRILABILIR bir kayit.
  Tabloyu elle yazmak iki sey yapar: (1) zamanla gercek sayilardan ayrisir,
  (2) insan istemeden kotu satirlari atlar. Betik her deneyin kendi
  dosyalarindan okuyor, atlamiyor, ve basarisiz deneyi de satir olarak
  yaziyor.

SUTUNLAR
  GP/GL/LP/LL  dort olcu, DOGRULAMA kumesinde (test kumesine dokunulmadi)
  GP/LP        suruklenme orani — yerelde iyi kuresel kotu olmak bu
               problemin klasik tuzagi, tek sayi okuyani yaniltir
  r            Faz 3'un bagdasim teshisi: alti bilesenin |r| ortancasi.
               Sifira yakinsa model goruntuye bakmiyor, veri kumesinin
               ortalamasini soyluyor. O satirdaki dort sayi "iyi" gorunse
               bile bir fikrin katkisini olcmuyordur.
  epok         sabit sure butcesinde kac epok sigdi — bir fikrin GIZLI
               bedeli burada gorunur (tutarlilik kisiti gibi)

Kullanim:
  python scripts/faz4_tablo.py                  # ekrana + results/faz4/tablo.md
  python scripts/faz4_tablo.py --test           # test kumesi olcumlerini goster
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
FAZ4 = KOK / "results" / "faz4"


def _oku(yol: Path):
    if not yol.exists():
        return None
    try:
        return json.loads(yol.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def epok_sayisi(dizin: Path) -> tuple[int | None, float | None]:
    """(kac epok kosuldu, en iyi dogrulama mesafesi)."""
    yol = dizin / "olcumler.csv"
    if not yol.exists():
        return None, None
    epok, en_iyi = None, None
    with open(yol, encoding="utf-8") as f:
        for satir in csv.DictReader(f):
            epok = int(satir["epok"])
            d = satir.get("dogrulama_mesafe")
            if d:
                v = float(d)
                en_iyi = v if en_iyi is None else min(en_iyi, v)
    return (epok + 1 if epok is not None else None), en_iyi


def deney_satiri(ad: str, dosya: str) -> dict:
    dizin = FAZ4 / ad
    s: dict = {"ad": ad}
    d = _oku(dizin / "deney.json") or {}
    s["hat"] = d.get("hat", "?")
    s["baslik"] = d.get("baslik", "")
    s["epok"], s["en_iyi_dogrulama"] = epok_sayisi(dizin)

    olcu = _oku(dizin / dosya)
    if olcu:
        for k in ("GP", "GL", "LP", "LL"):
            s[k] = olcu["ozet"][k]["ortalama"]
        s["tarama"] = len(olcu["taramalar"])
        s["GP/LP"] = s["GP"] / max(s["LP"], 1e-9)

    bag = _oku(dizin / "bagdasim" / "analiz.json")
    if bag:
        r = bag["ozet"].get("olcek_bagdasim_ortanca")
        if r:
            mutlak = sorted(abs(x) for x in r)
            s["r_ortanca"] = mutlak[len(mutlak) // 2]
            s["r_en_buyuk"] = mutlak[-1]
        s["olcek"] = bag["ozet"].get("olcek_ortanca")
    return s


def yazdir(satirlar: list[dict], baslik: str) -> str:
    sut = ["ad", "hat", "GP", "GL", "LP", "LL", "GP/LP", "r_ortanca", "epok"]
    genislik = {"ad": 14, "hat": 3, "GP": 8, "GL": 8, "LP": 7, "LL": 7,
                "GP/LP": 7, "r_ortanca": 6, "epok": 5}
    cizgi = "  ".join(s.ljust(genislik[s]) for s in sut)
    cikti = [baslik, "=" * len(cizgi), cizgi, "-" * len(cizgi)]
    for s in satirlar:
        hucre = []
        for k in sut:
            v = s.get(k)
            if v is None:
                hucre.append("—".ljust(genislik[k]))
            elif k in ("ad", "hat"):
                hucre.append(str(v).ljust(genislik[k]))
            elif k == "epok":
                hucre.append(f"{v}".rjust(genislik[k]))
            elif k in ("LP", "LL"):
                hucre.append(f"{v:.4f}".rjust(genislik[k]))
            elif k in ("GP/LP",):
                hucre.append(f"{v:.0f}x".rjust(genislik[k]))
            elif k == "r_ortanca":
                hucre.append(f"{v:.3f}".rjust(genislik[k]))
            else:
                hucre.append(f"{v:.3f}".rjust(genislik[k]))
        cikti.append("  ".join(hucre))
    return "\n".join(cikti)


def markdown(satirlar: list[dict], taban: dict | None) -> str:
    # Sutun adi "tabana gore GP" belirsizdi: GP DUSUNCE iyilesme oluyor, o
    # yuzden isaret ters okunuyordu. Artik acikca IYILESME yaziyor —
    # arti = tabandan iyi, eksi = tabandan kotu.
    sat = ["| Deney | Hat | GP | GL | LP | LL | GP/LP | \\|r\\| | Epok | GP iyilesmesi |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for s in satirlar:
        def f(k, bicim="{:.3f}"):
            return bicim.format(s[k]) if s.get(k) is not None else "—"
        fark = "—"
        if taban and s.get("GP") is not None and taban.get("GP"):
            if s["ad"] == taban["ad"]:
                fark = "(cipa)"
            else:
                yuzde = 100 * (1 - s["GP"] / taban["GP"])
                fark = f"%{yuzde:+.1f}"
        sat.append(
            f"| {s['ad']} | {s['hat']} | {f('GP')} | {f('GL')} | "
            f"{f('LP', '{:.4f}')} | {f('LL', '{:.4f}')} | "
            f"{f('GP/LP', '{:.0f}x')} | {f('r_ortanca')} | "
            f"{s.get('epok') or '—'} | {fark} |")
    return "\n".join(sat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", action="store_true",
                    help="dogrulama yerine test kumesi olcumlerini oku")
    ap.add_argument("--cikti", type=Path, default=FAZ4 / "tablo.md")
    a = ap.parse_args()

    if not FAZ4.exists():
        raise SystemExit(f"deney dizini yok: {FAZ4}")
    dosya = "test_olculer.json" if a.test else "dogrulama_olculer.json"
    adlar = sorted(d.name for d in FAZ4.iterdir() if d.is_dir())
    satirlar = [deney_satiri(ad, dosya) for ad in adlar]
    # taban once, sonra GP'ye gore siralanmis digerleri; olcumu olmayanlar sona
    taban = next((s for s in satirlar if s["ad"] == "taban"), None)
    kalan = [s for s in satirlar if s is not taban]
    kalan.sort(key=lambda s: (s.get("GP") is None, s.get("GP", 0)))
    sirali = ([taban] if taban else []) + kalan

    kume = "TEST" if a.test else "DOGRULAMA"
    print(yazdir(sirali, f"FAZ 4 DENEYLERI — {kume} KUMESI"))
    print("\nr = alti bilesenin |bagdasim| ortancasi. Sifira yakin = model "
          "goruntuye bakmiyor,\nveri kumesinin ortalama hareketini soyluyor "
          "(Faz 3 teshisi). O satirda dort sayi\n'iyi' gorunse bile bir "
          "fikrin katkisini olcmuyordur.")

    a.cikti.parent.mkdir(parents=True, exist_ok=True)
    a.cikti.write_text(
        f"# Faz 4 deneyleri — {kume.lower()} kumesi\n\n"
        + markdown(sirali, taban) + "\n", encoding="utf-8")
    print(f"\nyazildi: {a.cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
