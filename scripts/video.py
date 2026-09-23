"""60 saniyelik tanitim videosu: solda kareler akiyor, sagda hacim olusuyor.

Yol haritasi Faz 6 Adim 1. Sahne akisi:
  0-44 sn   solda ultrason karesi, sagda o ana kadar gelen karelerin
            MODELIN TAHMIN ETTIGI konumlarla olusturdugu hacim. Turuncu
            cizgi probun tahmin edilen yolu, turuncu cerceve o anki kare.
  44-45 sn  bekleme
  45-60 sn  tahmin ve gercek (izleyicinin olctugu konumlar) yan yana,
            ayni kamerayla donerek. Aradaki fark suruklenmenin kendisi.

TARAMA SECIMI — ELLE SECILMIYOR
  Varsayilan tarama, modelin test olcumunde GP'si ORTANCAYA en yakin olan.
  En iyi taramayi gostermek videoyu guzellestirir ama yalan soyler; ortanca
  "tipik bir tarama boyle gorunuyor" demenin durust yolu.

HACIM NASIL CIZILIYOR
  Voksel izgarasi kurup hacim isleme (volume rendering) yapmak yerine her
  karenin pikselleri seyreltilip 3B noktaya cevriliyor ve ekrana
  izdusurulerek MAKSIMUM YOGUNLUK IZDUSUMU (MIP) biriktiriliyor. Ultrasonda
  parlak yapilar (kemik, tendon kilifi) bilgiyi tasiyor, MIP tam onlari
  one cikarir. Derinlik ipucu icin yakin noktalar biraz daha parlak.
  Sabit kamerada tampon kare kare biriktirilebildigi icin olusum sahnesi
  ucuz; donen karsilastirma sahnesi her karede yeniden izdusuruluyor.

CIKARIM ISLEMCIDE
  Varsayilan aygit cpu: video ayni makinede suren bir egitimi rahatsiz
  etmesin diye. Bir tarama (~550 pencere) birkac dakika surer.

KULLANIM
  python scripts/video.py --kosum results/faz4/U_uzun
  python scripts/video.py --kosum results/faz4/U_uzun --tarama 003/RH_Per_L_PtD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))

from src.geometri import Kalibrasyon, kuresel_donusumler, piksel_noktalari  # noqa: E402
from src.olcum import kuresel_biriktir, model_yukle, yerel_tahmin  # noqa: E402
from src.veri import taramalari_bul  # noqa: E402

GEN, YUK, FPS = 1280, 720, 30
ARKA = (14, 17, 20)
TAHMIN = (214, 118, 58)      # src/gorsel.py RENK["tahmin"]in koyu zeminde okunan hali
GERCEK = (46, 170, 180)      # RENK["gercek"] icin ayni
SOLUK = (120, 130, 138)


# ------------------------------------------------------------------ veri
def ortanca_tarama(kosum: Path) -> str:
    satirlar = json.loads((kosum / "test_olculer.json").read_text())["taramalar"]
    gp = np.array([s["GP"] for s in satirlar])
    s = satirlar[int(np.argmin(np.abs(gp - np.median(gp))))]
    print(f"ortanca tarama: {s['denek']}/{s['tarama']}  GP {s['GP']:.2f} mm "
          f"(test ortancasi {np.median(gp):.2f})")
    return f"{s['denek']}/{s['tarama']}"


def tarama_yukle(veri: Path, kimlik: str):
    import h5py
    denek, ad = kimlik.split("/")
    _, hepsi = taramalari_bul(veri)
    t = next(t for t in hepsi if t.denek == denek and t.ad == ad)
    with h5py.File(t.kare_yolu) as f:
        kareler = np.asarray(f["frames"])
    with h5py.File(t.tform_yolu) as f:
        tforms = torch.tensor(np.asarray(f["tforms"]))
    return kareler, tforms


# ------------------------------------------------------------ geometri
def dunya_noktalari(T: torch.Tensor, kalib: Kalibrasyon, adim: int):
    """Her karenin seyreltilmis piksellerinin ilk kare uzayindaki mm konumu.

    Cikti (N, P, 3) float32. Koseler ve merkez ayrica: (N, 5, 3).
    """
    p = piksel_noktalari(480, 640, yogunluk=(480 // adim, 640 // adim))
    mm = kalib.olcek @ p
    nok = (T.float() @ mm[None])[:, :3].permute(0, 2, 1).numpy()
    kose = torch.tensor([[1.0, 640, 640, 1, 320], [1.0, 1, 480, 480, 240],
                         [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    kose = (T.float() @ (kalib.olcek @ kose)[None])[:, :3].permute(0, 2, 1).numpy()
    return nok.astype(np.float32), kose.astype(np.float32)


def piksel_indisleri(adim: int) -> np.ndarray:
    """dunya_noktalari'yla ayni sirada, karenin duz dizideki indisleri."""
    p = piksel_noktalari(480, 640, yogunluk=(480 // adim, 640 // adim))
    x = p[0].round().long().clamp(1, 640) - 1
    y = p[1].round().long().clamp(1, 480) - 1
    return (y * 640 + x).numpy()


class Kamera:
    """Ortografik kamera. Taban eksenleri gercek yorungenin PCA'sindan:
    birinci eksen tarama yonu, boylece hacim ekranda soldan saga uzanir."""

    def __init__(self, noktalar: np.ndarray, boyut: int, pay: float = 0.08):
        merkez = noktalar.mean(0)
        _, _, vt = np.linalg.svd(noktalar - merkez, full_matrices=False)
        self.merkez, self.taban, self.boyut = merkez, vt, boyut
        self.pay = pay
        self.olcek = 1.0

    def dondur(self, az: float, el: float) -> np.ndarray:
        a, e = np.radians(az), np.radians(el)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        return (Rx @ Rz @ self.taban).astype(np.float32)

    def sigdir(self, noktalar: np.ndarray, acilar) -> None:
        """Butun acilarda sigacak tek olcek — donerken zoom oynamasin."""
        yaricap = 0.0
        for az, el in acilar:
            q = (noktalar - self.merkez) @ self.dondur(az, el).T
            yaricap = max(yaricap, np.abs(q[:, :2]).max())
        self.olcek = (self.boyut * (0.5 - self.pay)) / yaricap

    def izdusur(self, noktalar: np.ndarray, R: np.ndarray):
        q = (noktalar - self.merkez) @ R.T
        u = (q[:, 0] * self.olcek + self.boyut / 2).astype(np.int32)
        v = (-q[:, 1] * self.olcek + self.boyut / 2).astype(np.int32)
        return u, v, q[:, 2]


class MIP:
    """Maksimum yogunluk izdusumu tamponu, derinlikle hafif golgeli."""

    def __init__(self, boyut: int, derinlik_araligi: tuple[float, float]):
        self.boyut = boyut
        self.tampon = np.zeros(boyut * boyut, np.float32)
        self.d0, self.d1 = derinlik_araligi

    def ekle(self, u, v, d, deger) -> None:
        g = (u >= 0) & (u < self.boyut) & (v >= 0) & (v < self.boyut)
        yakinlik = np.clip((d[g] - self.d0) / (self.d1 - self.d0 + 1e-6), 0, 1)
        np.maximum.at(self.tampon, v[g] * self.boyut + u[g],
                      deger[g] * (0.55 + 0.45 * yakinlik))

    def goruntu(self) -> np.ndarray:
        g = np.clip(self.tampon / 255.0, 0, 1) ** 0.8
        return (g * 255).astype(np.uint8).reshape(self.boyut, self.boyut)


# ------------------------------------------------------------- cizim
def yazi_tipi(boy: int):
    for ad in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(ad, boy)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(gri: np.ndarray, zemin=ARKA) -> Image.Image:
    rgb = np.stack([gri] * 3, -1).astype(np.float32)
    z = np.array(zemin, np.float32)
    rgb = np.where(gri[..., None] > 0, rgb, z)
    return Image.fromarray(rgb.astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kosum", type=Path, default=KOK / "results" / "faz4" / "U_uzun")
    ap.add_argument("--veri", type=Path, default=Path("D:/SerbestEl-veri/tusrec2024/acilmis"))
    ap.add_argument("--tarama", default=None, help="denek/tarama; verilmezse ortanca GP")
    ap.add_argument("--donme-temsili", default=None,
                    help="verilmezse kosumun ayarlar.json'undan okunur")
    ap.add_argument("--aygit", default="cpu")
    ap.add_argument("--cikti", type=Path, default=None)
    ap.add_argument("--sure", type=float, default=60.0)
    ap.add_argument("--onizleme", action="store_true",
                    help="video yerine birkac anlik kareyi png olarak yaz")
    a = ap.parse_args()

    ayar = json.loads((a.kosum / "ayarlar.json").read_text())
    temsil = a.donme_temsili or ayar.get("donme_temsili", "euler")
    kimlik = a.tarama or ortanca_tarama(a.kosum)
    cikti = a.cikti or (a.kosum / "video" / (kimlik.replace("/", "_") + ".mp4"))
    cikti.parent.mkdir(parents=True, exist_ok=True)

    kareler, tforms = tarama_yukle(a.veri, kimlik)
    kalib = Kalibrasyon.csvden(a.veri / "calib_matrix.csv")
    n = kareler.shape[0]

    # --- tahmin: onbellekli, ayni taramayi tekrar cizerken yeniden hesaplanmaz
    onbellek = cikti.with_suffix(".tahmin.pt")
    if onbellek.exists():
        t_kuresel = torch.load(onbellek)
    else:
        aygit = torch.device(a.aygit)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        model, ciftler, _ = model_yukle(a.kosum / "en_iyi_model.pt", aygit,
                                        donme_temsili=temsil)
        print(f"cikarim ({a.aygit}), {n} kare ...", flush=True)
        t_yerel = yerel_tahmin(model, kareler, kalib, ciftler, aygit,
                               donme_temsili=temsil, yigin=8)
        t_kuresel = torch.cat([torch.eye(4)[None], kuresel_biriktir(t_yerel)])
        torch.save(t_kuresel, onbellek)
    g_kuresel = kuresel_donusumler(tforms, kalib, ilk_kare_dahil=True)

    adim = 4
    t_nok, t_kose = dunya_noktalari(t_kuresel, kalib, adim)
    g_nok, g_kose = dunya_noktalari(g_kuresel, kalib, adim)
    idx = piksel_indisleri(adim)
    parlak = kareler.reshape(n, -1)[:, idx].astype(np.float32)   # (N, P)
    esik = 20.0                       # yelpazenin disindaki siyah zemin cizilmez

    gp_mm = float(np.linalg.norm(t_nok - g_nok, axis=-1).mean())
    print(f"bu taramada ortalama piksel hatasi {gp_mm:.2f} mm")

    # --- kameralar: iki hacim ayni kamerayla, ayni olcekte
    BOY_T, BOY_K = 600, 610
    birlesik = np.concatenate([t_nok[::5, ::7].reshape(-1, 3), g_nok[::5, ::7].reshape(-1, 3)])
    kam_t = Kamera(g_nok[::5, ::7].reshape(-1, 3), BOY_T)
    AZ0, EL0 = -35.0, 28.0
    kam_t.sigdir(birlesik, [(AZ0, EL0)])
    kam_k = Kamera(g_nok[::5, ::7].reshape(-1, 3), BOY_K)
    donus = [(AZ0 + 360 * i / 24, EL0) for i in range(24)]
    kam_k.sigdir(birlesik, donus)

    def derinlik_araligi(kam, R):
        d = ((birlesik - kam.merkez) @ R.T)[:, 2]
        return float(d.min()), float(d.max())

    R0 = kam_t.dondur(AZ0, EL0)
    mip = MIP(BOY_T, derinlik_araligi(kam_t, R0))
    iz_t = [kam_t.izdusur(t_kose[i], R0) for i in range(n)]      # kose+merkez

    kucuk, orta = yazi_tipi(18), yazi_tipi(22)
    toplam = int(a.sure * FPS)
    n_olusum = int(toplam * 44 / 60)
    n_bekle = int(toplam * 1 / 60)
    n_don = toplam - n_olusum - n_bekle

    def olusum_karesi(i: int) -> Image.Image:
        im = Image.new("RGB", (GEN, YUK), ARKA)
        us = Image.fromarray(kareler[i]).convert("RGB")
        im.paste(us, (20, (YUK - 480) // 2))
        hac = panel(mip.goruntu())
        d = ImageDraw.Draw(hac)
        yol = [(int(iz_t[k][0][4]), int(iz_t[k][1][4])) for k in range(0, i + 1)]
        if len(yol) > 1:
            d.line(yol, fill=TAHMIN, width=2)
        u, v, _ = iz_t[i]
        d.polygon([(int(u[k]), int(v[k])) for k in range(4)], outline=TAHMIN, width=2)
        im.paste(hac, (GEN - BOY_T - 20, (YUK - BOY_T) // 2))
        return im

    def karsilastirma_karesi(az: float) -> Image.Image:
        im = Image.new("RGB", (GEN, YUK), ARKA)
        R = kam_k.dondur(az, EL0)
        aralik = derinlik_araligi(kam_k, R)
        for sira, (nok, kose, renk, ad) in enumerate(
                [(t_nok, t_kose, TAHMIN, "tahmin — yalnızca görüntüden"),
                 (g_nok, g_kose, GERCEK, "gerçek — optik izleyici")]):
            m = MIP(BOY_K, aralik)
            s = nok[::2].reshape(-1, 3)
            u, v, dd = kam_k.izdusur(s, R)
            val = parlak[::2].reshape(-1)
            g = val > esik
            m.ekle(u[g], v[g], dd[g], val[g])
            hac = panel(m.goruntu())
            dr = ImageDraw.Draw(hac)
            if sira == 0:
                # gercek yol tahmin panelinde de ince cizgiyle: suruklenme
                # iki cizginin arasindaki acilma olarak gorunur
                gu, gv, _ = kam_k.izdusur(g_kose[:, 4], R)
                dr.line(list(zip(gu.tolist(), gv.tolist())), fill=GERCEK, width=1)
            merkez = kose[:, 4]
            uu, vv, _ = kam_k.izdusur(merkez, R)
            dr.line(list(zip(uu.tolist(), vv.tolist())), fill=renk, width=2)
            x0 = 20 + sira * (BOY_K + 20)
            im.paste(hac, (x0, 70))
            ImageDraw.Draw(im).text((x0 + 8, 36), ad, fill=renk, font=orta)
        ImageDraw.Draw(im).text(
            (20, YUK - 30), f"{kimlik}  ·  {n} kare  ·  ortalama piksel hatası {gp_mm:.1f} mm",
            fill=SOLUK, font=kucuk)
        return im

    if a.onizleme:
        for i in range(n):
            u, v, d = kam_t.izdusur(t_nok[i], R0)
            g = parlak[i] > esik
            mip.ekle(u[g], v[g], d[g], parlak[i][g])
            if i in (n // 3, n - 1):
                olusum_karesi(i).save(cikti.with_name(f"onizleme_olusum_{i}.png"))
        karsilastirma_karesi(AZ0).save(cikti.with_name("onizleme_karsilastirma.png"))
        print(f"onizleme yazildi: {cikti.parent}")
        return 0

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{GEN}x{YUK}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(cikti)],
        stdin=subprocess.PIPE)

    eklenen = -1
    son = None
    for k in range(n_olusum):
        i = min(n - 1, int(k * n / n_olusum))
        while eklenen < i:
            eklenen += 1
            u, v, d = kam_t.izdusur(t_nok[eklenen], R0)
            g = parlak[eklenen] > esik
            mip.ekle(u[g], v[g], d[g], parlak[eklenen][g])
        son = olusum_karesi(i)
        ffmpeg.stdin.write(son.tobytes())
        if k % FPS == 0:
            print(f"\r  olusum {k // FPS:>2}/{n_olusum // FPS} sn", end="", flush=True)
    for _ in range(n_bekle):
        ffmpeg.stdin.write(son.tobytes())
    for k in range(n_don):
        ffmpeg.stdin.write(karsilastirma_karesi(AZ0 + 360 * k / n_don).tobytes())
        if k % FPS == 0:
            print(f"\r  karsilastirma {k // FPS:>2}/{n_don // FPS} sn   ", end="", flush=True)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"\nyazildi: {cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
