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


def alti_vektordan_yigin(V: torch.Tensor) -> torch.Tensor:
    """(K,6) -> (K,4,4). alti_vektordan'in yiginli surumu."""
    V = V.double()
    T = torch.eye(4, dtype=torch.float64).repeat(V.shape[0], 1, 1)
    T[:, 0:3, 0:3] = axis_angle_to_matrix(V[:, 3:6] / DER)
    T[:, 0:3, 3] = V[:, 0:3]
    return T


def dogrusal_uydur(gercek: torch.Tensor, tahmin: torch.Tensor):
    """Bilesen bilesen  tahmin ~ a . gercek + b  en kucuk kareler uydurmasi.

    NEDEN BU OLCUM
      Sabit yanlilik (b) hatanin yalnizca bir turu. Model hareketi tutarli
      bicimde KUCUK ya da BUYUK tahmin ediyorsa hata hareketle ORANTILI olur;
      bu carpimsal bir sapmadir ve ortalama cikarmakla gitmez. a katsayisi
      bunu dogrudan olcer: a < 1 ise model hareketi kucumsuyor.

      Ayrim onemli, cunku carpimsal sapma hareket duzgun oldugu surece
      zincirde AYNI YONDE toplanir — yani yanlilik gibi davranir, ama
      yanlilik olarak olculmez.

    Dondurur: (a, b, r) her biri (6,). r bilesen basina bagdasim.
    """
    g = gercek.double()
    t = tahmin.double()
    gm, tm = g.mean(0), t.mean(0)
    gc, tc = g - gm, t - tm
    var = (gc * gc).sum(0)
    a = torch.where(var > 1e-18, (gc * tc).sum(0) / var.clamp_min(1e-18),
                    torch.ones_like(var))
    b = tm - a * gm
    r = torch.where(
        var > 1e-18,
        (gc * tc).sum(0) / (var.sqrt() * (tc * tc).sum(0).sqrt()).clamp_min(1e-18),
        torch.zeros_like(var))
    return a, b, r


def olcegi_duzelt(tahmin_v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                  r: torch.Tensor, r_esik: float = 0.5,
                  a_alt: float = 0.2, a_ust: float = 5.0) -> torch.Tensor:
    """tahmin ~ a.gercek + b uydurmasini tersine cevirir: (tahmin - b) / a.

    GUVENILIRLIK KAPISI — bu olmadan deney anlamsiz
      a katsayisi ancak bilesen gercekten tahmin ediliyorsa anlamli. Bagdasim
      sifira yakinsa (r ~ 0) model o bileseni hic bilmiyor demektir; a da
      sifira yakin cikar ve a'ya bolmek gurultuyu buyutur.

      Olmayan bir sinyal olceklenerek geri getirilemez.

    IKI KAPI GEREKIYOR — tek basina r YETMEDI, olculdu
      Ilk surumde yalnizca r kapisi vardi. Kendi modelimizde butun
      bilesenlerde a ~ 0,0001 iken, 240 taramanin birkacinda bir bilesenin
      r'si sans eseri 0,5'i gecti; 0,0001'e bolunce o taramalarin GP'si
      patladi ve ORTALAMA 12.669 mm'ye cikti. Tek bir gecersiz tarama
      kume ortalamasini tek basina bozuyor.

      Bu yuzden a'nin kendisi de sinirli: a_alt <= a <= a_ust. a negatifse
      ya da sifira yakinsa duzeltme yok — orada duzeltilecek bir olcek yok,
      tahmin edilmemis bir bilesen var.

    Raporlanmasi gereken sey hangi bilesenlerin duzeltildigi DEGIL,
    hangilerinin duzeltilemedigidir: duzeltilemeyen bilesen modelin hic
    ogrenmedigi bilesendir.
    """
    duzelt = (r.abs() >= r_esik) & (a >= a_alt) & (a <= a_ust)
    a_g = torch.where(duzelt, a, torch.ones_like(a))
    b_g = torch.where(duzelt, b, torch.zeros_like(b))
    return (tahmin_v.double() - b_g) / a_g


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
                  yogunluk=(24, 32), kuresel_yanlilik=None,
                  kuresel_olcek=None) -> dict:
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

    # --- carpimsal sapma: model hareketi kucumsuyor mu
    t_v = alti_vektor(t_yerel.double())
    a, b, r = dogrusal_uydur(g, t_v)
    olcek_keh = kare_basi_mesafe(
        gt_kuresel,
        kuresel_biriktir(
            alti_vektordan_yigin(olcegi_duzelt(t_v, a, b, r)).float()),
        mm)
    olcek_dur = None
    if kuresel_olcek is not None:
        ka, kb, kr = kuresel_olcek
        olcek_dur = kare_basi_mesafe(
            gt_kuresel,
            kuresel_biriktir(
                alti_vektordan_yigin(olcegi_duzelt(t_v, ka, kb, kr)).float()),
            mm)

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
        "olcek": a.numpy(),
        "olcek_kesme": b.numpy(),
        "olcek_bagdasim": r.numpy(),
        "GP_olcek_kehanet": float(olcek_keh.mean()),
        "GP_olcek_durust": (float(olcek_dur.mean())
                            if olcek_dur is not None else None),
        "hiz_ort": float(hiz.mean()),
        "donme_ort": float(donme.mean()),
        "bagdasim_hiz": bagdasim(yerel_egri, hiz),
        "bagdasim_donme": bagdasim(yerel_egri, donme),
        "duraklamada_hata": float(yerel_egri[yavas].mean()),
        "hareketde_hata": float(yerel_egri[hizli].mean()),
        "sicramalar": sicrama_bul(yerel_egri),
        # alt cizgili alanlar kume genelinde havuzlanmak icin; JSON'a girmez
        "_v": v.numpy(),                  # kare basi hata 6-vektoru
        "_g": g.numpy(),                  # gercek yerel hareket 6-vektoru
        "_t": t_v.numpy(),                # tahmin edilen yerel hareket
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
    olc = np.array([s["GP_olcek_kehanet"] for s in satirlar])
    e.scatter(gp, keh, s=12, alpha=0.6, color="#c0392b",
              label="sabit yanlilik (kehanet)")
    if satirlar[0]["GP_durust"] is not None:
        dur = np.array([s["GP_durust"] for s in satirlar])
        e.scatter(gp, dur, s=12, alpha=0.6, color="#e08a2e",
                  label="sabit yanlilik (durust)")
    e.scatter(gp, olc, s=12, alpha=0.6, color="#2e7d32", marker="^",
              label="olcek de (kehanet)")
    if satirlar[0]["GP_olcek_durust"] is not None:
        okd = np.array([s["GP_olcek_durust"] for s in satirlar])
        e.scatter(gp, okd, s=12, alpha=0.6, color="#1f6f8b", marker="^",
                  label="olcek de (durust)")
    lim = [float(min(gp.min(), keh.min(), olc.min())) * 0.9,
           float(gp.max()) * 1.1]
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

def en_kotu_gorselleri(sirali, veri: Path, cikti: Path, model, kalib, ciftler,
                       aygit, n: int) -> int:
    """Yol haritasi Faz 3 adim 4: en kotu taramalari GOZLE izlenebilir yap.

    Sayi nerede kotu oldugunu soyler, NEDEN kotu oldugunu soylemez. Gercek
    ve tahmin edilen prob yorungesi ust uste cizilince suruklenmenin hangi
    eksende ve taramanin neresinde basladigi dogrudan gorunuyor.
    """
    from src.gorsel import hata_grafigi, prob_konumlari, yorunge_cizimi

    dizin = cikti / "en_kotu"
    dizin.mkdir(parents=True, exist_ok=True)
    _, tum = taramalari_bul(veri)
    dizinlenmis = {(t.denek, t.ad): t for t in tum}

    for s in sirali[:n]:
        t = dizinlenmis[(s["denek"], s["tarama"])]
        with h5py.File(t.kare_yolu) as f:
            kareler = np.asarray(f["frames"])
        with h5py.File(t.tform_yolu) as f:
            tforms = torch.tensor(np.asarray(f["tforms"]))
        gt_kuresel = kuresel_donusumler(tforms, kalib)
        t_kuresel = kuresel_biriktir(
            yerel_tahmin(model, kareler, kalib, ciftler, aygit))
        ad = f"{s['denek']}_{s['tarama']}"
        yorunge_cizimi(prob_konumlari(gt_kuresel, kalib),
                       dizin / f"{ad}_yorunge.png",
                       tahmin=prob_konumlari(t_kuresel, kalib),
                       baslik=f"{ad}  GP {s['GP']:.1f} mm")
        hata_grafigi({"kuresel (ilk kareye gore)": s["kuresel_egri"],
                      "yerel (onceki kareye gore)": s["yerel_egri"]},
                     dizin / f"{ad}_hata.png",
                     baslik=f"{ad}  GP {s['GP']:.1f} mm", log=True)
        print(f"    {ad}: yorunge + hata grafigi", flush=True)
    return min(n, len(sirali))


def _taramalari_oku(veri: Path, denekler: list[str]):
    _, tum = taramalari_bul(veri)
    return [t for t in tum if t.denek in set(denekler)]


def _kos(taramalar, model, kalib, ciftler, aygit, yogunluk, kuresel_yanlilik,
         sinir=None, baslik="", kuresel_olcek=None):
    satirlar = []
    n = len(taramalar) if sinir is None else min(sinir, len(taramalar))
    for i, t in enumerate(taramalar[:n], 1):
        with h5py.File(t.kare_yolu) as f:
            kareler = np.asarray(f["frames"])
        with h5py.File(t.tform_yolu) as f:
            tforms = torch.tensor(np.asarray(f["tforms"]))
        s = tarama_analiz(kareler, tforms, model, kalib, ciftler, aygit,
                          yogunluk=yogunluk, kuresel_yanlilik=kuresel_yanlilik,
                          kuresel_olcek=kuresel_olcek)
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
    ap.add_argument("--yanlilik-denek", nargs="+", default=None,
                    metavar="DENEK",
                    help="yanliligi kume yerine bu deneklerden olc; test "
                         "denekleriyle kesisirse hata verir")
    ap.add_argument("--yanlilik-tarama", type=int, default=24,
                    help="yanlilik olcumu icin kac tarama yeter")
    ap.add_argument("--sinir", type=int, default=None,
                    help="yalnizca ilk N test taramasi (hizli deneme)")
    ap.add_argument("--gorsel", type=int, default=10, metavar="N",
                    help="en kotu N tarama icin yorunge ve hata grafigi "
                         "(0 = uretme)")
    a = ap.parse_args()

    a.cikti.mkdir(parents=True, exist_ok=True)
    aygit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    model, ciftler, _ = model_yukle(a.agirlik, aygit)
    bolme = bolme_yukle(a.bolme)
    yogunluk = tuple(a.yogunluk)
    print(f"aygit {aygit} | agirlik {a.agirlik.name} | "
          f"izgara {yogunluk[0]}x{yogunluk[1]}")

    # --- 1) durust yanlilik ve olcek: TEST DISI deneklerde olculur
    kuresel_yanlilik = None
    kuresel_olcek = None
    test_denekler = set(bolme.denek_kumesi(a.kume))
    if a.yanlilik_denek:
        ortak = set(a.yanlilik_denek) & test_denekler
        if ortak:
            raise SystemExit(
                f"yanlilik denekleri test kumesiyle kesisiyor: {sorted(ortak)}\n"
                f"Test etiketinden olculen yanlilik 'durust' degil, kehanettir.")
        yanlilik_denekler = list(a.yanlilik_denek)
    elif a.yanlilik_kumesi != "yok":
        yanlilik_denekler = bolme.denek_kumesi(a.yanlilik_kumesi)
    else:
        yanlilik_denekler = []
    kaynak = _taramalari_oku(a.veri, yanlilik_denekler) if yanlilik_denekler else []
    if yanlilik_denekler and not kaynak:
        # Bolmede o kume bos olabilir (orn. yalnizca olcum icin hazirlanmis
        # bolmeler). Sessizce test kumesine dusmek YASAK: yanlilik testte
        # olculup teste uygulanirsa "durust" surum kehanete doner ve sayi
        # yalan olur. Onun yerine ozellik kapatiliyor.
        print(f"\nUYARI: yanlilik kaynagi bos ({yanlilik_denekler}) — durust "
              f"yanlilik giderme ATLANDI. Yalnizca kehanet raporlanacak.")
    elif kaynak:
        print(f"\nyanlilik olcumu: {len(yanlilik_denekler)} denek, "
              f"{min(a.yanlilik_tarama, len(kaynak))} tarama")
        on = _kos(kaynak, model, kalib, ciftler, aygit, yogunluk, None,
                  sinir=a.yanlilik_tarama, baslik="y ")
        kuresel_yanlilik = torch.tensor(
            np.concatenate([s["_v"] for s in on]).mean(axis=0))
        ka, kb, kr = dogrusal_uydur(
            torch.tensor(np.concatenate([s["_g"] for s in on])),
            torch.tensor(np.concatenate([s["_t"] for s in on])))
        kuresel_olcek = (ka, kb, kr)
        print("  olculen kuresel yanlilik ve olcek:")
        for i, ad in enumerate(BILESEN):
            print(f"    {ad:<16} yanlilik {kuresel_yanlilik[i]:+.6f}   "
                  f"olcek {ka[i]:.4f}   kesme {kb[i]:+.6f}   r {kr[i]:+.3f}")

    # --- 2) test kumesi
    test = _taramalari_oku(a.veri, bolme.denek_kumesi(a.kume))
    print(f"\n{a.kume} kumesi: {len(test)} tarama")
    satirlar = _kos(test, model, kalib, ciftler, aygit, yogunluk,
                    kuresel_yanlilik, sinir=a.sinir,
                    kuresel_olcek=kuresel_olcek)

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
    olcekler = np.array([s["olcek"] for s in satirlar])
    print("  hareket olcegi (tahmin ~ a . gercek):  a<1 = model hareketi kucumsuyor")
    for i, ad in enumerate(BILESEN):
        print(f"    {ad:<16} a = {np.median(olcekler[:, i]):.4f}   "
              f"r = {np.median(np.array([s['olcek_bagdasim'] for s in satirlar])[:, i]):+.3f}")

    ok = np.array([s["GP_olcek_kehanet"] for s in satirlar])
    print(f"\n  GP {gp.mean():.3f} mm  ->")
    print(f"    sabit yanlilik giderilince   {keh.mean():8.3f} mm  "
          f"(kehanet, %{100*(1-keh.mean()/max(gp.mean(), 1e-9)):.1f})")
    dur_ort = None
    if satirlar[0]["GP_durust"] is not None:
        dur = np.array([s["GP_durust"] for s in satirlar])
        dur_ort = float(dur.mean())
        print(f"    sabit yanlilik giderilince   {dur_ort:8.3f} mm  "
              f"(durust,  %{100*(1-dur_ort/max(gp.mean(), 1e-9)):.1f})")
    print(f"    olcek de duzeltilince        {ok.mean():8.3f} mm  "
          f"(kehanet, %{100*(1-ok.mean()/max(gp.mean(), 1e-9)):.1f})")
    ok_dur = None
    if satirlar[0]["GP_olcek_durust"] is not None:
        okd = np.array([s["GP_olcek_durust"] for s in satirlar])
        ok_dur = float(okd.mean())
        print(f"    olcek de duzeltilince        {ok_dur:8.3f} mm  "
              f"(durust,  %{100*(1-ok_dur/max(gp.mean(), 1e-9)):.1f})")

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

    if a.gorsel:
        print(f"\n  en kotu {a.gorsel} tarama gorsellestiriliyor:")
        en_kotu_gorselleri(sirali, a.veri, a.cikti, model, kalib, ciftler,
                           aygit, a.gorsel)

    # --- 4) dosyaya yaz
    def temiz(s):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in s.items() if not k.startswith("_")}

    (a.cikti / "analiz.json").write_text(json.dumps({
        "agirlik": str(a.agirlik), "kume": a.kume,
        "kuresel_yanlilik": (kuresel_yanlilik.tolist()
                             if kuresel_yanlilik is not None else None),
        "kuresel_olcek": ([k.tolist() for k in kuresel_olcek]
                          if kuresel_olcek is not None else None),
        "ozet": {
            "GP": float(gp.mean()), "LP": float(lp.mean()),
            "GP_kehanet": float(keh.mean()), "GP_durust": dur_ort,
            "GP_olcek_kehanet": float(ok.mean()), "GP_olcek_durust": ok_dur,
            "olcek_ortanca": np.median(olcekler, axis=0).tolist(),
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
