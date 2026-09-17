"""Faz 3 — hata analizi. Dort sayidan sonra gelen soru: hata NEDEN birikiyor.

Yol haritasinin Faz 3 adimlari burada kodlanmis durumda:
  1. Suruklenme nerede basliyor       -> kare basina kuresel hata egrisi
  2. Hangi hareket tipinde cokuyor    -> hiz / donme / duraklama ayrimi
  3. Yerel ve kuresel hatayi ayristir -> yanlilik olcumu ve giderme deneyi
  4. En kotu on tarama                -> siralama, gozle izlemek icin liste

ANA SORU — FAZ 1'DEN GELIYOR
  Faz 1'de kontrollu benzetimle olculdu: suruklenmeyi kare basina hatanin
  BUYUKLUGU degil, YANLILIGI belirliyor. Yansiz 0,40 der/mm hata 1570 karede
  22 mm kuresel hata biriktiriyor; 20 kat KUCUK ama yanli 0,02 der/mm hata
  77 mm biriktiriyor.

  Burada ayni soru gercek modele soruluyor. Yerel hata donusumu
      E_i = T_gercek_yerel_i^-1 . T_tahmin_yerel_i
  kare basina "modelin ne kadar yanildigi"dir. Bunun alti bileseninin
  (oteleme x/y/z mm, donme x/y/z derece) ORTALAMASI yanliligi,
  STANDART SAPMASI gurultuyu verir. |ortalama| / sapma orani 1'e yaklasiyorsa
  hata yanli demektir ve zincirlendiginde dogrusal buyur.

YANLILIK GIDERME DENEYI — iki surum, ikisi de raporlanir
  kehanet : yanlilik her taramanin KENDI gercek degerlerinden olculur ve geri
            alinir. Uygulanabilir bir yontem DEGIL (test etiketini kullaniyor);
            suruklenmenin ne kadarinin yanliliktan geldigini gosteren UST SINIR.
  durust  : yanlilik test disi bir kumede (varsayilan: dogrulama) olculur ve
            test taramalarina oldugu gibi uygulanir. Test etiketine
            dokunulmaz — bu, Faz 4'te gercekten kullanilabilecek bir duzeltme.

  Ikisinin arasindaki fark yanliligin taramaya mi modele mi ait oldugunu
  soyler: durust surum de kazandiriyorsa yanlilik modelin sabit bir kusurudur.

KULLANIM
  python src/analiz.py --veri <veri> --agirlik <model.pt> --cikti results/faz3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
if str(KOK) not in sys.path:
    sys.path.insert(0, str(KOK))

import matplotlib                                        # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402

from src.bolme import yukle as bolme_yukle               # noqa: E402
from src.geometri import (                               # noqa: E402
    Kalibrasyon,
    kuresel_donusumler,
    piksel_noktalari,
    yerel_donusumler,
)
from src.olcum import kuresel_biriktir, model_yukle, yerel_tahmin  # noqa: E402
from src.veri import taramalari_bul                      # noqa: E402

DER = 180.0 / math.pi
BILESEN = ["oteleme x (mm)", "oteleme y (mm)", "oteleme z (mm)",
           "donme x (der)", "donme y (der)", "donme z (der)"]


# ----------------------------------------------------------------- matematik

def hata_donusumu(gercek: torch.Tensor, tahmin: torch.Tensor) -> torch.Tensor:
    """E_i = T_gercek_i^-1 @ T_tahmin_i. Kimlik matrisi ise hata yok."""
    return torch.linalg.inv(gercek.double()) @ tahmin.double()


def alti_vektor(T: torch.Tensor) -> torch.Tensor:
    """(K,4,4) -> (K,6): [oteleme xyz mm, donme xyz derece].

    Donme eksen-aci olarak aciliyor: Euler'in aksine tekilligi yok ve kucuk
    acilarda bilesenler dogrusal davraniyor — ortalamasini almak anlamli.
    """
    ote = T[:, 0:3, 3]
    don = matrix_to_axis_angle(T[:, 0:3, 0:3]) * DER
    return torch.cat([ote, don], dim=1)


def alti_vektordan(v: torch.Tensor) -> torch.Tensor:
    """(6,) -> (4,4). alti_vektor'un tersi."""
    T = torch.eye(4, dtype=torch.float64)
    T[0:3, 0:3] = axis_angle_to_matrix(v[3:6].double() / DER)
    T[0:3, 3] = v[0:3].double()
    return T


def kare_basi_mesafe(T_a: torch.Tensor, T_b: torch.Tensor,
                     mm: torch.Tensor) -> np.ndarray:
    """Iki donusum dizisinin ayni noktalara etkisi — KARE BASINA ortalama mm.

    olcum.py'deki _piksel_mesafesi ile ayni matematik; tek fark tek sayiya
    indirgemeyip kare basina degeri dondurmesi. Suruklenme egrisi budur.
    """
    fark = ((T_a.float() - T_b.float()) @ mm)[:, 0:3, :]
    return fark.pow(2).sum(dim=1).sqrt().mean(dim=1).numpy()


def yanliligi_gider(t_yerel: torch.Tensor, yanlilik: torch.Tensor) -> torch.Tensor:
    """Tahmin edilen yerel donusumlerden sabit yanlilik bileseni cikarilir.

    T_tahmin_i = T_gercek_i . E_i oldugundan, E_i ~ E_ortalama ise
    T_tahmin_i . E_ortalama^-1 ~ T_gercek_i olur.
    """
    E_ters = torch.linalg.inv(alti_vektordan(yanlilik))
    return (t_yerel.double() @ E_ters[None]).float()


def sicrama_bul(egri: np.ndarray, esik: float = 6.0) -> list[int]:
    """Yerel hata egrisinde sicrayan kareler — dayanikli z skoru.

    Ortalama/sapma yerine ortanca/MAD kullaniliyor: birkac buyuk sicrama
    ortalamayi ve sapmayi kendi yukari cekip kendini gizler.
    """
    ortanca = float(np.median(egri))
    mad = float(np.median(np.abs(egri - ortanca)))
    if mad < 1e-12:
        return []
    z = 0.6745 * (egri - ortanca) / mad
    return np.flatnonzero(z > esik).tolist()


# ------------------------------------------------------------------- tarama

def tarama_analiz(kareler, tforms, model, kalib, ciftler, aygit,
                  yogunluk=(24, 32), kuresel_yanlilik=None) -> dict:
    """Bir taramanin butun Faz 3 olcumleri."""
    gt_kuresel = kuresel_donusumler(tforms, kalib)
    gt_yerel = yerel_donusumler(tforms, kalib)
    t_yerel = yerel_tahmin(model, kareler, kalib, ciftler, aygit)
    t_kuresel = kuresel_biriktir(t_yerel)

    mm = kalib.olcek @ piksel_noktalari(480, 640, yogunluk=yogunluk)

    kuresel_egri = kare_basi_mesafe(gt_kuresel, t_kuresel, mm)
    yerel_egri = kare_basi_mesafe(gt_yerel, t_yerel, mm)

    # --- hatanin yonu: yanlilik mi gurultu mu
    E = hata_donusumu(gt_yerel, t_yerel)
    v = alti_vektor(E)                                    # (K,6)
    yanlilik = v.mean(dim=0)
    gurultu = v.std(dim=0)
    oran = yanlilik.abs() / gurultu.clamp_min(1e-12)

    # --- yanlilik giderme: kehanet (bu taramanin kendi yanliligi)
    kehanet = kare_basi_mesafe(
        gt_kuresel, kuresel_biriktir(yanliligi_gider(t_yerel, yanlilik)), mm)

    # --- yanlilik giderme: durust (test disi kumede olculmus yanlilik)
    durust = None
    if kuresel_yanlilik is not None:
        durust = kare_basi_mesafe(
            gt_kuresel,
            kuresel_biriktir(yanliligi_gider(t_yerel, kuresel_yanlilik)), mm)

    # --- hareket tanimlayicilari GERCEK yerel donusumlerden
    g = alti_vektor(gt_yerel.double())
    hiz = g[:, 0:3].norm(dim=1).numpy()                   # mm / kare
    donme = g[:, 3:6].norm(dim=1).numpy()                 # derece / kare

    def bagdasim(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    yavas = hiz < np.percentile(hiz, 10)
    hizli = hiz > np.percentile(hiz, 90)

    return {
        "kare": int(kareler.shape[0]),
        "GP": float(kuresel_egri.mean()),
        "LP": float(yerel_egri.mean()),
        "GP_son": float(kuresel_egri[-1]),
        "kuresel_egri": kuresel_egri,
        "yerel_egri": yerel_egri,
        "yanlilik": yanlilik.numpy(),
        "gurultu": gurultu.numpy(),
        "yanlilik_orani": oran.numpy(),
        "GP_kehanet": float(kehanet.mean()),
        "GP_son_kehanet": float(kehanet[-1]),
        "GP_durust": float(durust.mean()) if durust is not None else None,
        "GP_son_durust": float(durust[-1]) if durust is not None else None,
        "hiz_ort": float(hiz.mean()),
        "donme_ort": float(donme.mean()),
        "bagdasim_hiz": bagdasim(yerel_egri, hiz),
        "bagdasim_donme": bagdasim(yerel_egri, donme),
        "duraklamada_hata": float(yerel_egri[yavas].mean()),
        "hareketde_hata": float(yerel_egri[hizli].mean()),
        "sicramalar": sicrama_bul(yerel_egri),
        "_v": v.numpy(),                  # yanlilik havuzu icin; JSON'a girmez
    }


# ----------------------------------------------------------------- cizimler

def cizimler(satirlar: list[dict], cikti: Path) -> None:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    izgara = np.linspace(0, 1, 100)
    yigin = np.array([np.interp(izgara,
                                np.linspace(0, 1, len(s["kuresel_egri"])),
                                s["kuresel_egri"]) for s in satirlar])
    ortanca = np.median(yigin, axis=0)

    # 1) suruklenme egrileri ve buyume bicimi
    f, eks = plt.subplots(1, 2, figsize=(10, 3.6))
    for s in satirlar:
        e = s["kuresel_egri"]
        eks[0].plot(np.linspace(0, 1, len(e)), e, lw=0.4, alpha=0.25,
                    color="#4a6fa5")
    eks[0].plot(izgara, ortanca, lw=2, color="#c0392b", label="ortanca")
    eks[0].set_xlabel("taramanin ne kadari gecti")
    eks[0].set_ylabel("kuresel hata (mm)")
    eks[0].set_title("Suruklenme egrileri — her tarama bir cizgi")
    eks[0].legend()

    n = ortanca / max(ortanca[-1], 1e-9)
    eks[1].plot(izgara, n, lw=2, color="#c0392b", label="olculen (ortanca)")
    eks[1].plot(izgara, izgara, "--", lw=1, color="#555", label="dogrusal (yanli)")
    eks[1].plot(izgara, np.sqrt(izgara), ":", lw=1.2, color="#2e7d32",
                label="karekok (yansiz rastgele yuruyus)")
    eks[1].set_xlabel("taramanin ne kadari gecti")
    eks[1].set_ylabel("hata / son hata")
    eks[1].set_title("Buyume bicimi")
    eks[1].legend()
    f.tight_layout()
    f.savefig(cikti / "suruklenme_egrileri.png")
    plt.close(f)

    # 2) yanlilik ve gurultu
    yan = np.array([s["yanlilik"] for s in satirlar])
    gur = np.array([s["gurultu"] for s in satirlar])
    oran = np.array([s["yanlilik_orani"] for s in satirlar])
    f, eks = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(6)
    eks[0].bar(x - 0.2, np.abs(yan).mean(axis=0), 0.4, label="|yanlilik|",
               color="#c0392b")
    eks[0].bar(x + 0.2, gur.mean(axis=0), 0.4, label="gurultu (std)",
               color="#4a6fa5")
    eks[0].set_xticks(x)
    eks[0].set_xticklabels(BILESEN, rotation=30, ha="right")
    eks[0].set_yscale("log")
    eks[0].set_title("Kare basina hatanin yonu")
    eks[0].legend()

    eks[1].boxplot([oran[:, i] for i in range(6)], labels=BILESEN)
    eks[1].axhline(1.0, ls="--", color="#c0392b", lw=1)
    eks[1].set_ylabel("|yanlilik| / gurultu")
    eks[1].set_yscale("log")
    eks[1].set_title("1'in uzerinde = yanlilik baskin")
    plt.setp(eks[1].get_xticklabels(), rotation=30, ha="right")
    f.tight_layout()
    f.savefig(cikti / "yanlilik.png")
    plt.close(f)

    # 3) yanlilik giderme
    f, e = plt.subplots(figsize=(5.4, 4.2))
    gp = np.array([s["GP"] for s in satirlar])
    keh = np.array([s["GP_kehanet"] for s in satirlar])
    e.scatter(gp, keh, s=12, alpha=0.6, color="#c0392b", label="kehanet")
    if satirlar[0]["GP_durust"] is not None:
        dur = np.array([s["GP_durust"] for s in satirlar])
        e.scatter(gp, dur, s=12, alpha=0.6, color="#2e7d32", label="durust")
    lim = [float(min(gp.min(), keh.min())) * 0.9, float(gp.max()) * 1.1]
    e.plot(lim, lim, "--", color="#555", lw=1, label="degisim yok")
    e.set_xscale("log")
    e.set_yscale("log")
    e.set_xlabel("GP (mm)")
    e.set_ylabel("yanlilik giderildikten sonra GP (mm)")
    e.set_title("Suruklenmenin ne kadari yanliliktan")
    e.legend()
    f.tight_layout()
    f.savefig(cikti / "yanlilik_giderme.png")
    plt.close(f)

    # 4) hareket ve hata
    f, eks = plt.subplots(1, 2, figsize=(10, 3.6))
    eks[0].scatter([s["hiz_ort"] for s in satirlar],
                   [s["LP"] for s in satirlar], s=12, alpha=0.6, color="#4a6fa5")
    eks[0].set_xlabel("ortalama hiz (mm/kare)")
    eks[0].set_ylabel("LP (mm)")
    eks[0].set_title("Hizli tarama daha mi kotu")
    eks[1].scatter([s["donme_ort"] for s in satirlar],
                   [s["LP"] for s in satirlar], s=12, alpha=0.6, color="#8e44ad")
    eks[1].set_xlabel("ortalama donme (derece/kare)")
    eks[1].set_ylabel("LP (mm)")
    eks[1].set_title("Donme daha mi zor")
    f.tight_layout()
    f.savefig(cikti / "hareket_hata.png")
    plt.close(f)


# ---------------------------------------------------------------------- ana

def _taramalari_oku(veri: Path, denekler: list[str]):
    _, tum = taramalari_bul(veri)
    return [t for t in tum if t.denek in set(denekler)]


def _kos(taramalar, model, kalib, ciftler, aygit, yogunluk, kuresel_yanlilik,
         sinir=None, baslik=""):
    satirlar = []
    n = len(taramalar) if sinir is None else min(sinir, len(taramalar))
    for i, t in enumerate(taramalar[:n], 1):
        with h5py.File(t.kare_yolu) as f:
            kareler = np.asarray(f["frames"])
        with h5py.File(t.tform_yolu) as f:
            tforms = torch.tensor(np.asarray(f["tforms"]))
        s = tarama_analiz(kareler, tforms, model, kalib, ciftler, aygit,
                          yogunluk=yogunluk, kuresel_yanlilik=kuresel_yanlilik)
        s.update(denek=t.denek, tarama=t.ad)
        satirlar.append(s)
        print(f"  [{baslik}{i}/{n}] {t.denek}/{t.ad:<14} {s['kare']:>5} kare  "
              f"GP {s['GP']:8.3f}  LP {s['LP']:6.4f}  "
              f"kehanet {s['GP_kehanet']:8.3f} mm", flush=True)
    return satirlar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--veri", type=Path, required=True)
    ap.add_argument("--agirlik", type=Path, required=True)
    ap.add_argument("--bolme", type=Path, default=KOK / "configs" / "bolme.json")
    ap.add_argument("--kume", default="test")
    ap.add_argument("--cikti", type=Path, default=KOK / "results" / "faz3")
    ap.add_argument("--yogunluk", type=int, nargs=2, default=(24, 32),
                    metavar=("SATIR", "SUTUN"),
                    help="piksel izgarasi; tam 480x640 izgara gereksiz yavas")
    ap.add_argument("--yanlilik-kumesi", default="dogrulama",
                    help="durust yanliligin olculecegi kume ('yok' ile kapat)")
    ap.add_argument("--yanlilik-tarama", type=int, default=24,
                    help="yanlilik olcumu icin kac tarama yeter")
    ap.add_argument("--sinir", type=int, default=None,
                    help="yalnizca ilk N test taramasi (hizli deneme)")
    a = ap.parse_args()

    a.cikti.mkdir(parents=True, exist_ok=True)
    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    model, ciftler, _ = model_yukle(a.agirlik, aygit)
    bolme = bolme_yukle(a.bolme)
    yogunluk = tuple(a.yogunluk)
    print(f"aygit {aygit} | agirlik {a.agirlik.name} | "
          f"izgara {yogunluk[0]}x{yogunluk[1]}")

    # --- 1) durust yanlilik: TEST DISI bir kumede olculur
    kuresel_yanlilik = None
    if a.yanlilik_kumesi != "yok":
        kaynak = _taramalari_oku(a.veri, bolme.denek_kumesi(a.yanlilik_kumesi))
        print(f"\nyanlilik olcumu: {a.yanlilik_kumesi} kumesi, "
              f"{min(a.yanlilik_tarama, len(kaynak))} tarama")
        on = _kos(kaynak, model, kalib, ciftler, aygit, yogunluk, None,
                  sinir=a.yanlilik_tarama, baslik="y ")
        havuz = np.concatenate([s["_v"] for s in on])
        kuresel_yanlilik = torch.tensor(havuz.mean(axis=0))
        print("  olculen kuresel yanlilik:")
        for ad, d in zip(BILESEN, kuresel_yanlilik.tolist()):
            print(f"    {ad:<16} {d:+.6f}")

    # --- 2) test kumesi
    test = _taramalari_oku(a.veri, bolme.denek_kumesi(a.kume))
    print(f"\n{a.kume} kumesi: {len(test)} tarama")
    satirlar = _kos(test, model, kalib, ciftler, aygit, yogunluk,
                    kuresel_yanlilik, sinir=a.sinir)

    cizimler(satirlar, a.cikti)

    # --- 3) ozet ve teshis
    gp = np.array([s["GP"] for s in satirlar])
    lp = np.array([s["LP"] for s in satirlar])
    keh = np.array([s["GP_kehanet"] for s in satirlar])
    oran = np.array([s["yanlilik_orani"] for s in satirlar])
    izgara = np.linspace(0, 1, 100)
    yigin = np.array([np.interp(izgara,
                                np.linspace(0, 1, len(s["kuresel_egri"])),
                                s["kuresel_egri"]) for s in satirlar])
    ortanca = np.median(yigin, axis=0)
    n = ortanca / max(ortanca[-1], 1e-9)
    n_dogru = float(np.mean((n - izgara) ** 2))
    n_karekok = float(np.mean((n - np.sqrt(izgara)) ** 2))

    print("\n" + "=" * 66)
    print(f"FAZ 3 OZET — {len(satirlar)} tarama")
    print("=" * 66)
    print(f"  GP {gp.mean():8.3f} mm    LP {lp.mean():7.4f} mm    "
          f"GP/LP = {gp.mean()/max(lp.mean(), 1e-9):.1f}x")
    print(f"  buyume bicimi: dogrusal artik {n_dogru:.4f} | "
          f"karekok artik {n_karekok:.4f}  -> "
          f"{'DOGRUSAL' if n_dogru < n_karekok else 'KAREKOK'}")
    print("  |yanlilik| / gurultu (ortanca):")
    for i, ad in enumerate(BILESEN):
        print(f"    {ad:<16} {np.median(oran[:, i]):.3f}")
    print(f"  yanlilik giderilince GP: {gp.mean():.3f} -> {keh.mean():.3f} mm "
          f"(kehanet, %{100*(1-keh.mean()/max(gp.mean(), 1e-9)):.1f} dusus)")
    dur_ort = None
    if satirlar[0]["GP_durust"] is not None:
        dur = np.array([s["GP_durust"] for s in satirlar])
        dur_ort = float(dur.mean())
        print(f"  yanlilik giderilince GP: {gp.mean():.3f} -> {dur_ort:.3f} mm "
              f"(durust, %{100*(1-dur_ort/max(gp.mean(), 1e-9)):.1f} dusus)")

    b_hiz = np.array([s["bagdasim_hiz"] for s in satirlar])
    b_don = np.array([s["bagdasim_donme"] for s in satirlar])
    durak = np.array([s["duraklamada_hata"] for s in satirlar])
    hizli = np.array([s["hareketde_hata"] for s in satirlar])
    print(f"  kare basi hata ~ hiz bagdasimi   : {np.nanmedian(b_hiz):+.3f}")
    print(f"  kare basi hata ~ donme bagdasimi : {np.nanmedian(b_don):+.3f}")
    print(f"  en yavas %10 karede hata {durak.mean():.4f} mm | "
          f"en hizli %10 karede {hizli.mean():.4f} mm")

    sirali = sorted(satirlar, key=lambda s: -s["GP"])[:10]
    print("\n  EN KOTU ON TARAMA (gozle izlenecek):")
    for s in sirali:
        print(f"    {s['denek']}/{s['tarama']:<14} GP {s['GP']:8.3f} mm  "
              f"LP {s['LP']:.4f}  hiz {s['hiz_ort']:.3f} mm/kare  "
              f"donme {s['donme_ort']:.3f} der/kare  "
              f"sicrama {len(s['sicramalar'])}")

    # --- 4) dosyaya yaz
    def temiz(s):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in s.items() if not k.startswith("_")}

    (a.cikti / "analiz.json").write_text(json.dumps({
        "agirlik": str(a.agirlik), "kume": a.kume,
        "kuresel_yanlilik": (kuresel_yanlilik.tolist()
                             if kuresel_yanlilik is not None else None),
        "ozet": {
            "GP": float(gp.mean()), "LP": float(lp.mean()),
            "GP_kehanet": float(keh.mean()), "GP_durust": dur_ort,
            "buyume_dogrusal_artik": n_dogru,
            "buyume_karekok_artik": n_karekok,
            "yanlilik_orani_ortanca": np.median(oran, axis=0).tolist(),
            "bagdasim_hiz_ortanca": float(np.nanmedian(b_hiz)),
            "bagdasim_donme_ortanca": float(np.nanmedian(b_don)),
            "duraklamada_hata": float(durak.mean()),
            "hareketde_hata": float(hizli.mean()),
        },
        "en_kotu_on": [{"denek": s["denek"], "tarama": s["tarama"],
                        "GP": s["GP"]} for s in sirali],
        "taramalar": [temiz(s) for s in satirlar],
    }, indent=2), encoding="utf-8")
    print(f"\nyazildi: {a.cikti / 'analiz.json'} + 4 gorsel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
