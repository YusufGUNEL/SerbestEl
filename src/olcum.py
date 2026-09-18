"""Dort olcuyu (GP, GL, LP, LL) kendi elimizle hesaplar — Faz 2 Adim 3.

DORT OLCU
  GP  tum piksellerin ILK kareye gore yer degistirmesi   -> suruklenmeyi yakalar
  GL  isaret noktalarinin ILK kareye gore                -> klinik, kuresel
  LP  tum piksellerin ONCEKI kareye gore                 -> adim adim hareket
  LL  isaret noktalarinin ONCEKI kareye gore             -> klinik, yerel
  Dordu de raporlanir. Yerelde iyi kuresel de kotu olmak bu problemin klasik
  tuzagi; tek sayi vermek okuyani yaniltir.

IKI YERDE REFERANSTAN AYRILIYORUZ — MATEMATIK AYNI KALARAK

1) TOPLU ILERI GECIS
   Referans `Prediction.cal_pred_transformations` kareleri TEK TEK modelden
   geciriyor (1570 kare -> 1570 ileri gecis, yigin 1). Oysa NUM_SAMPLES=2 iken
   pencereler birbirinden BAGIMSIZ: her tahmin yalnizca kendi kare ciftine
   bakiyor. Sirali olan tek sey biriktirme, o da sadece matris carpimi.
   Bu yuzden butun pencereleri yiginlayip bir kerede geciriyoruz. Ayni sayilar,
   cok daha hizli.
   NOT: NUM_SAMPLES>2 icin referans son karelerin donusumunu uretmiyor ve
   bosluklari dolduruyor; o davranis burada da birebir korundu.

2) DDF'LERI HIC KURMADAN OLCMEK
   Referans dort DDF'yi bellekte kuruyor. Bir tarama icin
   (1569, 3, 307200) float32 = 5,8 GB, gercek+tahmin dort alan icin ~23 GB.
   README'nin ">=32 GB RAM" uyarisi bundan.

   Oysa olculen sey iki DDF'nin FARKI ve fark sadelesiyor:
       DDF_gercek - DDF_tahmin
         = (T_gercek . p - p) - (T_tahmin . p - p)
         = T_gercek . p  -  T_tahmin . p
   Yani "- p" terimi dusuyor; DDF'leri kurmaya hic gerek yok, iki donusumun
   ayni noktalara etkisi karsilastirilabilir. Kare bloklariyla ilerleyince
   tepe bellek yuzlerce MB'a iniyor.
   Bu sadelesme `tests/test_olcum.py` icinde referansa karsi dogrulaniyor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"
for p in (str(KOK), str(REFERANS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.geometri import (  # noqa: E402
    Kalibrasyon,
    kuresel_donusumler,
    piksel_noktalari,
    yerel_donusumler,
)
from src.omurga import kur as omurga_kur  # noqa: E402
from src.temsil import cikti_boyutu, donusume_matrise  # noqa: E402
from utils.funs import pair_samples  # noqa: E402
from utils.plot_functions import reference_image_points  # noqa: E402


def model_yukle(agirlik: Path, aygit, num_samples=2, num_pred=1,
                donme_temsili="euler", model_name="efficientnet_b1"):
    ciftler = pair_samples(num_samples, num_pred, 0)
    pred_dim = cikti_boyutu(donme_temsili, ciftler.shape[0])
    # ImageNet agirligi burada gereksiz: hemen uzerine egitilmis agirlik biniyor
    model = omurga_kur(model_name, in_frames=num_samples, pred_dim=pred_dim).to(aygit)
    model.load_state_dict(torch.load(agirlik, map_location=aygit))
    model.eval()
    return model, ciftler, pred_dim


@torch.no_grad()
def yerel_tahmin(model, kareler: np.ndarray, kalib: Kalibrasyon, ciftler,
                 aygit, num_samples=2, donme_temsili="euler",
                 yigin=16, amp=True) -> torch.Tensor:
    """Modelden kare basina YEREL donusum. Cikti (N-1, 4, 4) cpu."""
    n = kareler.shape[0]
    kalib_kw = dict(
        image_points=reference_image_points([480, 640], 2).to(aygit),
        tform_image_to_tool=kalib.tam.to(aygit),
        tform_image_mm_to_tool=kalib.rijit.to(aygit),
        tform_image_pixel_to_mm=kalib.olcek.to(aygit),
    )

    def donusum(cikti):
        return donusume_matrise(donme_temsili, cikti, ciftler.shape[0], **kalib_kw)
    aralik = int(ciftler[0][1] - ciftler[0][0])

    # referansin pencere baslangiclari: 0, aralik, 2*aralik, ... son pencereye kadar
    basliklar = list(range(0, n - num_samples + 1, aralik))
    yerel = torch.eye(4).repeat(n - 1, 1, 1)

    for b in range(0, len(basliklar), yigin):
        kume = basliklar[b:b + yigin]
        pencere = np.stack([kareler[i:i + num_samples] for i in kume])
        x = torch.from_numpy(pencere).to(aygit).float() / 255
        with torch.cuda.amp.autocast(enabled=amp and aygit.type == "cuda"):
            cikti = model(x)
        T = donusum(cikti.float())[:, 0, ...].cpu()   # ilk cift, referansla ayni
        for k, i in enumerate(kume):
            yerel[i] = T[k]
    return yerel


def kuresel_biriktir(yerel: torch.Tensor) -> torch.Tensor:
    """T_kuresel[i] = T_kuresel[i-1] @ T_yerel[i]. Suruklenme tam burada doguyor."""
    kuresel = torch.empty_like(yerel)
    birikim = torch.eye(4, dtype=yerel.dtype)
    for i in range(yerel.shape[0]):
        birikim = birikim @ yerel[i]
        kuresel[i] = birikim
    return kuresel


def _piksel_mesafesi(T_a: torch.Tensor, T_b: torch.Tensor, mm: torch.Tensor,
                     blok: int = 24) -> float:
    """Iki donusum dizisinin ayni noktalara etkisi arasindaki ortalama mesafe (mm).

    DDF'ler kurulmuyor: fark alindiginda "- p" terimi sadelesiyor (bkz. modul
    aciklamasi). Kare bloklariyla ilerlenerek tepe bellek dusuk tutuluyor.
    """
    toplam, adet = 0.0, 0
    for b in range(0, T_a.shape[0], blok):
        a, c = T_a[b:b + blok], T_b[b:b + blok]
        fark = ((a - c) @ mm)[:, 0:3, :]          # (B,3,P)
        d = fark.pow(2).sum(dim=1).sqrt()          # (B,P)
        toplam += d.sum().item()
        adet += d.numel()
    return toplam / adet


def _isaret_mesafesi(T_a: torch.Tensor, T_b: torch.Tensor, kalib: Kalibrasyon,
                     isaretler: torch.Tensor, w: int = 640) -> float:
    """Isaret noktalarinda ortalama mesafe (mm).

    isaretler (100,3): [kare no, x, y]. Kare no 0'dan, x/y 1'den basliyor.
    Donusum dizileri ilk kareyi ICERMEDIGI icin kare f -> indeks f-1.
    """
    kare = isaretler[:, 0].long() - 1
    x = isaretler[:, 1].to(torch.float32)
    y = isaretler[:, 2].to(torch.float32)
    p = torch.stack([x, y, torch.zeros_like(x), torch.ones_like(x)])   # (4,100)
    mm = kalib.olcek @ p
    gecerli = (kare >= 0) & (kare < T_a.shape[0])
    kare, mm = kare[gecerli], mm[:, gecerli]
    if kare.numel() == 0:
        return float("nan")
    fark = (T_a[kare] - T_b[kare]) @ mm.T.unsqueeze(-1)   # (K,4,1)
    return fark[:, 0:3, 0].pow(2).sum(dim=1).sqrt().mean().item()


def isaret_yolu(veri_kok: Path, denek: str) -> Path | None:
    """Denege ait isaret dosyasini bulur — kumeler farkli yerlere koyuyor.

    TUS-REC2025: `landmarks/landmark_<denek>.h5`
    TUS-REC2024: landmark.zip duz aciliyor, dosyalar kokte duruyor.
    Ikisi de denenir; bulunamazsa None doner ve isaret olculeri NaN olur
    (GP/LP yine hesaplanir — olcum isaret yok diye durmaz).
    """
    for aday in (veri_kok / "landmarks" / f"landmark_{denek}.h5",
                 veri_kok / f"landmark_{denek}.h5"):
        if aday.exists():
            return aday
    return None


def tarama_olc(kareler, tforms, isaretler, model, kalib, ciftler, aygit,
               nokta_yogunlugu=None, **kw) -> dict:
    """Bir tarama icin dort olcu. nokta_yogunlugu=None -> butun 307200 piksel."""
    gt_kuresel = kuresel_donusumler(tforms, kalib)
    gt_yerel = yerel_donusumler(tforms, kalib)
    t_yerel = yerel_tahmin(model, kareler, kalib, ciftler, aygit, **kw)
    t_kuresel = kuresel_biriktir(t_yerel)

    p = piksel_noktalari(480, 640, yogunluk=nokta_yogunlugu)
    mm = kalib.olcek @ p
    return {
        "GP": _piksel_mesafesi(gt_kuresel, t_kuresel, mm),
        "GL": _isaret_mesafesi(gt_kuresel, t_kuresel, kalib, isaretler),
        "LP": _piksel_mesafesi(gt_yerel, t_yerel, mm),
        "LL": _isaret_mesafesi(gt_yerel, t_yerel, kalib, isaretler),
        "kare": int(kareler.shape[0]),
    }


def main() -> int:
    from src.bolme import yukle as bolme_yukle

    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", type=Path, default=KOK / "data")
    ap.add_argument("--agirlik", type=Path, required=True)
    ap.add_argument("--bolme", type=Path, default=KOK / "configs" / "bolme.json")
    ap.add_argument("--kume", default="test", choices=["egitim", "dogrulama", "test"])
    ap.add_argument("--cikti", type=Path, default=None)
    ap.add_argument("--seyrek", type=int, default=None,
                    help="piksel izgarasini seyrelt (hizli deneme; None = 307200)")
    # --- kosumun mimarisi: egitimdeki degerlerle AYNI verilmeli ---
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--num-pred", type=int, default=1)
    ap.add_argument("--donme-temsili", default="euler",
                    choices=["euler", "6b", "kuaterniyon", "matris"])
    ap.add_argument("--tarama-basina", type=int, default=None, metavar="N",
                    help="denek basina ilk N tarama (ada gore sirali, rastgelelik "
                         "yok). Faz 4 deneylerini siralarken kullaniliyor: 240 "
                         "taramanin tamami deney basina ~26 dakika suruyor")
    a = ap.parse_args()

    import h5py

    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    model, ciftler, _ = model_yukle(a.agirlik, aygit, num_samples=a.num_samples,
                                    num_pred=a.num_pred,
                                    donme_temsili=a.donme_temsili)
    bolme = bolme_yukle(a.bolme)
    denekler = bolme.denek_kumesi(a.kume)
    print(f"{a.kume} kumesi: {len(denekler)} denek  |  agirlik: {a.agirlik.name}")

    # Duzen tanimayi src/veri.py yapiyor: uc duzen de (birlesik / ayri / duz)
    # ayni Tarama nesnesine iniyor, burada ayrim kalmiyor.
    from src.veri import taramalari_bul

    _, tum_taramalar = taramalari_bul(a.veri)
    istenen = set(denekler)
    secili = [t for t in tum_taramalar if t.denek in istenen]
    if a.tarama_basina:
        gruplu: dict[str, list] = {}
        for t in sorted(secili, key=lambda t: (t.denek, t.ad)):
            gruplu.setdefault(t.denek, []).append(t)
        secili = [t for g in gruplu.values() for t in g[:a.tarama_basina]]
    print(f"  {len(secili)} tarama olculecek")

    satirlar = []
    for t in secili:
        with h5py.File(t.kare_yolu) as f:
            kareler = np.asarray(f["frames"])
        with h5py.File(t.tform_yolu) as f:
            tforms = torch.tensor(np.asarray(f["tforms"]))

        ly = isaret_yolu(a.veri, t.denek)
        if ly is None:
            isaretler = torch.zeros((0, 3), dtype=torch.int64)
        else:
            with h5py.File(ly) as f:
                isaretler = torch.from_numpy(np.asarray(f[t.ad]))

        s = tarama_olc(kareler, tforms, isaretler, model, kalib, ciftler,
                       aygit, nokta_yogunlugu=(a.seyrek, a.seyrek) if a.seyrek else None,
                       num_samples=a.num_samples, donme_temsili=a.donme_temsili)
        s.update(denek=t.denek, tarama=t.ad)
        satirlar.append(s)
        print(f"  {t.denek}/{t.ad:<14} {s['kare']:>5} kare   "
              f"GP {s['GP']:7.3f}  GL {s['GL']:7.3f}  "
              f"LP {s['LP']:6.4f}  LL {s['LL']:6.4f}   mm", flush=True)

    print("\n" + "=" * 64)
    print(f"{a.kume.upper()} KUMESI ORTALAMASI ({len(satirlar)} tarama)")
    print("=" * 64)
    ozet = {}
    for k in ("GP", "GL", "LP", "LL"):
        v = np.array([s[k] for s in satirlar], dtype=float)
        ozet[k] = {"ortalama": float(np.nanmean(v)), "std": float(np.nanstd(v))}
        print(f"  {k}  {np.nanmean(v):8.4f} mm   (std {np.nanstd(v):.4f})")
    print("\nYerelde iyi kuresel de kotu olmak bu problemin klasik tuzagi.")
    print("Dort sayi birlikte okunur; GP/LP orani suruklenmenin olcusudur.")
    if satirlar:
        oran = ozet["GP"]["ortalama"] / max(ozet["LP"]["ortalama"], 1e-9)
        print(f"GP/LP = {oran:.1f}x")

    if a.cikti:
        a.cikti.parent.mkdir(parents=True, exist_ok=True)
        a.cikti.write_text(json.dumps(
            {"kume": a.kume, "agirlik": str(a.agirlik),
             "num_samples": a.num_samples, "num_pred": a.num_pred,
             "donme_temsili": a.donme_temsili,
             "tarama_basina": a.tarama_basina,
             "taramalar": satirlar, "ozet": ozet}, indent=2), encoding="utf-8")
        print(f"\nyazildi: {a.cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
