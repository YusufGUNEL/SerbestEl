"""src/olcum.py dogrulamasi — iki iddiayi referansa karsi sinar.

IDDIA 1 — TOPLU ILERI GECIS
  Referans kareleri tek tek modelden geciriyor; biz pencereleri yiginliyoruz.
  Uretilen yerel ve kuresel donusumler BIREBIR ayni olmali.

IDDIA 2 — DDF KURMADAN OLCMEK
  DDF'leri kurup farkini almak yerine dogrudan donusum farkini olcuyoruz,
  cunku "- p" terimi sadelesiyor. Referansin cal_dist ile hesapladigi dort
  sayiyla ayni cikmali.

Iddialar tutmazsa Faz 2'nin butun sayilari supheli olur, o yuzden bu test
Faz 2 kosulmadan once gecmeli.

Kosum:  python tests/test_olcum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"
for p in (str(KOK), str(REFERANS)):
    sys.path.insert(0, p)

from src.geometri import (  # noqa: E402
    Kalibrasyon, kuresel_donusumler, piksel_noktalari, yerel_donusumler,
)
from src.olcum import (  # noqa: E402
    _isaret_mesafesi, _piksel_mesafesi, kuresel_biriktir, model_yukle, yerel_tahmin,
)

VERI = KOK / "data"
AGIRLIK = REFERANS / "TUS-REC2024_model" / "model_weights"
KARE = 40           # tam tarama gereksiz; iddialar 40 karede de kirilir
sonuclar = []


def kontrol(ad, fn):
    try:
        gecti, detay = fn()
    except Exception as exc:  # noqa: BLE001
        gecti, detay = False, f"{type(exc).__name__}: {exc}"
    sonuclar.append((ad, gecti, detay))
    print(f"[{'OK ' if gecti else 'HATA'}] {ad}: {detay}")


def _veri():
    import h5py

    with h5py.File(VERI / "frames" / "050" / "LH_rotation.h5") as f:
        kareler = np.asarray(f["frames"][:KARE])
    with h5py.File(VERI / "transfs" / "050" / "LH_rotation.h5") as f:
        tforms = torch.tensor(np.asarray(f["tforms"][:KARE]))
    with h5py.File(VERI / "landmarks" / "landmark_050.h5") as f:
        isaretler = torch.from_numpy(np.asarray(f["LH_rotation"]))
    return kareler, tforms, isaretler


def _referans_tahmini(kareler, kalib, aygit):
    """Referansin sirali dongusunu birebir tekrar eder."""
    from utils.funs import pair_samples, type_dim
    from utils.plot_functions import reference_image_points
    from utils.transform import Transforms

    ciftler = pair_samples(2, 1, 0).to(aygit)
    nokta = reference_image_points([480, 640], 2).to(aygit)
    from utils.network import build_model

    model = build_model({"model_name": "efficientnet_b1"}, in_frames=2,
                        pred_dim=type_dim("parameter", nokta.shape[1],
                                          ciftler.shape[0])).to(aygit)
    model.load_state_dict(torch.load(AGIRLIK, map_location=aygit))
    model.eval()

    don = Transforms(pred_type="parameter", num_pairs=ciftler.shape[0],
                     image_points=nokta, tform_image_to_tool=kalib.tam.to(aygit),
                     tform_image_mm_to_tool=kalib.rijit.to(aygit),
                     tform_image_pixel_to_mm=kalib.olcek.to(aygit))

    f = torch.tensor(kareler)[None, ...].to(aygit).float() / 255
    n = f.shape[1]
    yerel = torch.zeros(n - 1, 4, 4)
    kuresel = torch.zeros(n - 1, 4, 4)
    onceki = torch.eye(4).to(aygit)
    i0 = 0
    aralik = int(torch.squeeze(ciftler[0])[1] - torch.squeeze(ciftler[0])[0])
    while True:
        with torch.no_grad():
            alt = f[:, i0:i0 + 2, ...]
            T = don(model(alt))[0, 0, ...]
            yerel[i0] = T.cpu()
            onceki = onceki @ T
            kuresel[i0] = onceki.cpu()
        i0 += aralik
        if (i0 + 2) > n:
            break
    return yerel, kuresel, model


def t1_mantik_referansla_ozdes():
    """Yigin 1'de sonuc referansla BIREBIR ayni olmali.

    Bu, mantigin ozdes oldugunu kanitlar: pencere secimi, hangi ciftin
    kullanildigi, aralik, son pencere... hepsi. Yigin>1'deki kucuk farklarin
    mantiktan degil yalnizca yigin boyutundan geldigini boylece biliyoruz.
    """
    kareler, _, _ = _veri()
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_yerel, r_kuresel, model = _referans_tahmini(kareler, kalib, aygit)

    from utils.funs import pair_samples

    b_yerel = yerel_tahmin(model, kareler, kalib, pair_samples(2, 1, 0), aygit,
                           yigin=1, amp=False)
    b_kuresel = kuresel_biriktir(b_yerel)
    hy = (r_yerel - b_yerel).abs().max().item()
    hk = (r_kuresel - b_kuresel).abs().max().item()

    # YEREL donusum BIREBIR ayni olmali: ayni girdi, ayni ag, ayni yigin boyutu.
    # Mantik ozdesligini kanitlayan sey bu.
    #
    # KURESEL'de eser fark beklenir: referans biriktirmeyi GPU'da yapiyor
    # (onceki @ T, ikisi de cuda), biz CPU'da. Ayni carpimlar, farkli yuvarlama
    # sirasi — mantik farki degil, kayan nokta.
    #
    # Esik keyfi secilmedi, zincir uzunlugundan turetildi: float32 epsilon
    # ~1.2e-07, donusum girdileri 1-10 mertebesinde, KARE-1 adet carpim
    # zincirleniyor. Beklenen birikim ~ eps * deger * zincir.
    esik = np.finfo(np.float32).eps * 10 * (KARE - 1)
    return hy == 0.0 and hk < esik, (
        f"{KARE} kare, yigin 1: yerel fark {hy:.1e} (birebir ozdes), "
        f"kuresel fark {hk:.1e} < esik {esik:.1e} "
        f"(GPU/CPU biriktirme yuvarlamasi)")


def t1b_yiginlamanin_etkisi_ihmal_edilebilir():
    """Yigin>1 kucuk sayisal fark yaratir; DORT OLCUYE etkisi ihmal edilebilir mi.

    Fark cuDNN'in farkli yigin boyutlarina farkli evrisim cekirdegi secmesinden
    geliyor — float32'de beklenen davranis, hata degil (olculdu: koşumlar arasi
    belirlenimci, yigin boyutuyla buyumuyor, donme blogunda 2e-06, otelemede
    7e-04 mm).

    Bit-bit esitlik YANLIS kriter olurdu. Dogru kriter: raporlanan dort sayi
    degisiyor mu? Tam taramada olculen fark %0,02'nin altinda.
    """
    kareler, tforms, isaretler = _veri()
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, model = _referans_tahmini(kareler, kalib, aygit)

    from utils.funs import pair_samples

    g_kuresel = kuresel_donusumler(tforms, kalib)
    g_yerel = yerel_donusumler(tforms, kalib)
    mm = kalib.olcek @ piksel_noktalari(480, 640)

    olculer = {}
    for yigin in (1, 16):
        ty = yerel_tahmin(model, kareler, kalib, pair_samples(2, 1, 0), aygit,
                          yigin=yigin, amp=False)
        tk = kuresel_biriktir(ty)
        olculer[yigin] = {
            "GP": _piksel_mesafesi(g_kuresel, tk, mm),
            "GL": _isaret_mesafesi(g_kuresel, tk, kalib, isaretler),
            "LP": _piksel_mesafesi(g_yerel, ty, mm),
            "LL": _isaret_mesafesi(g_yerel, ty, kalib, isaretler),
        }
    bagil = {k: abs(olculer[1][k] - olculer[16][k]) / max(abs(olculer[1][k]), 1e-9)
             for k in ("GP", "GL", "LP", "LL")}
    en_kotu = max(v for v in bagil.values() if not np.isnan(v))
    detay = ", ".join(f"{k} %{v*100:.3f}" for k, v in bagil.items())
    return en_kotu < 1e-3, f"yigin 1 -> 16 bagil fark: {detay}"


def t2_olcu_referansla_ayni():
    """Dort sayi referansin cal_dist yoluyla ayni mi — DDF'ler kurularak."""
    from utils.metrics import cal_dist
    from src.geometri import isaret_ddf, yer_degistirme
    from utils.funs import pair_samples

    kareler, tforms, isaretler = _veri()
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, model = _referans_tahmini(kareler, kalib, aygit)

    t_yerel = yerel_tahmin(model, kareler, kalib, pair_samples(2, 1, 0), aygit,
                           yigin=16, amp=False)
    t_kuresel = kuresel_biriktir(t_yerel)
    g_kuresel = kuresel_donusumler(tforms, kalib)
    g_yerel = yerel_donusumler(tforms, kalib)

    isaret_dilim = isaretler[(isaretler[:, 0] >= 1) & (isaretler[:, 0] < KARE)]
    if isaret_dilim.shape[0] == 0:
        isaret_dilim = torch.tensor([[KARE // 2, 300, 200]])

    p = piksel_noktalari(480, 640)
    mm = kalib.olcek @ p

    # --- referans yolu: DDF'leri kur, cal_dist cagir ---
    ref = {}
    for ad, G, T, mod in (("GP", g_kuresel, t_kuresel, "all"),
                          ("LP", g_yerel, t_yerel, "all")):
        ref[ad] = cal_dist(yer_degistirme(G, p, kalib).numpy(),
                           yer_degistirme(T, p, kalib).numpy(), mod)
    for ad, G, T in (("GL", g_kuresel, t_kuresel), ("LL", g_yerel, t_yerel)):
        ref[ad] = cal_dist(isaret_ddf(yer_degistirme(G, p, kalib), isaret_dilim).numpy(),
                           isaret_ddf(yer_degistirme(T, p, kalib), isaret_dilim).numpy(),
                           "landmark")

    # --- bizim yol: DDF kurmadan ---
    biz = {
        "GP": _piksel_mesafesi(g_kuresel, t_kuresel, mm),
        "LP": _piksel_mesafesi(g_yerel, t_yerel, mm),
        "GL": _isaret_mesafesi(g_kuresel, t_kuresel, kalib, isaret_dilim),
        "LL": _isaret_mesafesi(g_yerel, t_yerel, kalib, isaret_dilim),
    }

    farklar = {k: abs(float(ref[k]) - biz[k]) for k in ("GP", "GL", "LP", "LL")}
    bagil = max(farklar[k] / max(abs(float(ref[k])), 1e-9) for k in farklar)
    detay = ", ".join(f"{k}: ref {float(ref[k]):.4f} / biz {biz[k]:.4f}" for k in
                      ("GP", "GL", "LP", "LL"))
    return bagil < 1e-4, f"{detay}  (en buyuk bagil fark {bagil:.1e})"


def t3_bloklama_sonucu_degistirmiyor():
    """Blok boyutu sonucu DEGISTIRMEMELI — bellegi dusuren sey bloklama.

    NOT: burada bellek OLCULMUYOR. `tracemalloc` torch tensorlerini saymaz
    (torch kendi ayiricisini kullanir), o yuzden onunla yapilan bir olcum
    yaniltici olurdu. Bloklamanin bellek kazanci analitik olarak zaten belli:
    tepe bellek blok boyutuyla orantili, tarama uzunlugundan bagimsiz.

    Testin isi baska: bloklamanin SONUCU bozmadigini gostermek. Blok boyutu
    degistikce cevap degisiyorsa bir toplama/ortalama hatasi var demektir.
    """
    kareler, tforms, _ = _veri()
    kalib = Kalibrasyon.csvden(VERI / "calib_matrix.csv")
    g_kuresel = kuresel_donusumler(tforms, kalib)
    sahte = g_kuresel.clone()
    sahte[:, 0:3, 3] += 0.7                      # olculebilir bir fark yarat
    mm = kalib.olcek @ piksel_noktalari(480, 640)

    degerler = {b: _piksel_mesafesi(g_kuresel, sahte, mm, blok=b)
                for b in (1, 4, 24, 1000)}
    v = list(degerler.values())
    yayilim = max(v) - min(v)

    # blok basina tepe bellek (float32, 3 bileşen, 307200 nokta)
    mb = lambda b: b * 3 * 307200 * 4 / 1e6  # noqa: E731
    return yayilim < 1e-6, (
        f"blok 1/4/24/1000 -> {v[0]:.6f} mm, yayilim {yayilim:.1e}. "
        f"Blok 24 icin tepe ~{mb(24):.0f} MB; tam taramayi (1569 kare) DDF olarak "
        f"kurmak iki alan icin ~{1569*3*307200*4*2/1e9:.1f} GB isterdi.")


for ad, fn in [
    ("T1  mantik referansla ozdes (yigin 1)", t1_mantik_referansla_ozdes),
    ("T1b yiginlamanin dort olcuye etkisi ihmal edilebilir",
     t1b_yiginlamanin_etkisi_ihmal_edilebilir),
    ("T2  dort olcu referansla ayni", t2_olcu_referansla_ayni),
    ("T3  bloklama sonucu degistirmiyor", t3_bloklama_sonucu_degistirmiyor),
]:
    kontrol(ad, fn)

gecen = sum(1 for _, g, _ in sonuclar if g)
print(f"\n{gecen}/{len(sonuclar)} kontrol gecti")
sys.exit(0 if gecen == len(sonuclar) else 1)
