"""Donme temsilleri — agin donmeyi hangi sayilarla tahmin ettigi.

NEDEN AYRI BIR MODUL
  Faz 3 iki sey olctu: (1) model regresyon cokusunde, (2) cokmeyen bir modelde
  bile donme otelemeden belirgin sekilde kotu (oteleme olcegi 0,30-0,73, donme
  olcegi 0,09-0,29). Ikincisi yol haritasinin B hattini — "donme temsilini
  degistir" — olculmus bir gerekce haline getirdi.

SORUN: SUREKSIZLIK
  Euler acilari ve kuaterniyonlar SO(3)'u surekli temsil edemez. Euler'de
  gimbal kilidi ve sarma (pi -> -pi sicramasi), kuaterniyonda cift ortu
  (q ve -q ayni donme) vardir. Sinir agi surekli bir fonksiyon ogrenir;
  hedef fonksiyon sureksizse yakinsama bozulur. Zhou ve ark. (CVPR 2019)
  bunu gosterip 6 boyutlu surekli temsili onerdi: ag iki 3B vektor uretir,
  Gram-Schmidt ile ortonormallestirilip donme matrisi kurulur. Sureksizlik
  yok, ag her zaman gecerli bir donme uretir.

  | Temsil      | Boyut | Sureksiz mi | Cikti hep gecerli mi |
  |-------------|-------|-------------|----------------------|
  | euler ZYX   | 3     | EVET (sarma, gimbal) | evet         |
  | kuaterniyon | 4     | EVET (cift ortu)     | normlanirsa  |
  | matris 3x4  | 12    | hayir       | HAYIR (dik degil)    |
  | 6B surekli  | 6     | hayir       | evet (Gram-Schmidt)  |

NE DEGISMIYOR
  Etiket, kayip ve dort olcu ayni kaliyor. Degisen tek sey agin son katmaninin
  kac sayi urettigi ve o sayilarin 4x4 donusume nasil cevrildigi. Tek degisken
  kurali bu yuzden tutuyor.

  `euler`, `kuaterniyon`, `matris` icin ceviriyi REFERANSIN kodu yapiyor —
  Faz 2 sayilarinin birebir uretilebilmesi icin o yol hic degistirilmedi.
  Yalnizca `6b` bu modulde kuruluyor. `tests/test_temsil.py` euler yolunun
  referansla bit bit ayni kaldigini kilitliyor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"
for p in (str(KOK), str(REFERANS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.transform import PredictionTransform, Transforms  # noqa: E402

# temsil adi -> (cift basina cikti boyutu, referanstaki pred_type karsiligi)
TEMSILLER = {
    "euler":       (6,  "parameter"),
    "kuaterniyon": (7,  "quaternion"),
    "matris":      (12, "transform"),
    "6b":          (9,  None),          # 6 donme + 3 oteleme, referansta yok
}


def cikti_boyutu(temsil: str, num_pairs: int) -> int:
    """Agin son katmaninin kac sayi uretecegi.

    Referansin `type_dim` islevi burada KULLANILMIYOR: sozlugunu istekli
    kuruyor ve "point" girdisi num_points ile carpiliyor, o yuzden nokta
    sayisi verilmeden cagrilamiyor. Buradaki dort temsilin hicbiri nokta
    temsili degil; sayilar TEMSILLER sozlugunde zaten duruyor.
    """
    if temsil not in TEMSILLER:
        raise ValueError(f"bilinmeyen donme temsili: {temsil}")
    return TEMSILLER[temsil][0] * num_pairs


def altib_donusume(cikti: torch.Tensor) -> torch.Tensor:
    """(B,P,9) -> (B,P,4,4). Zhou ve ark. 6B surekli temsili.

    Ilk 6 sayi iki 3B vektor: Gram-Schmidt ile ortonormal taban kurulur.
      b1 = a1 / |a1|
      b2 = (a2 - <b1,a2> b1) normlanir      # a1 yonundeki bileseni cikarilir
      b3 = b1 x b2                          # sag el kurali, det = +1
    Son 3 sayi oteleme; hicbir kisitlamaya tabi degil.

    float32'ye zorlaniyor: normalize ve capraz carpim yari hassasiyette
    (AMP) kucuk normlarda sayisal olarak guvenilmez, egitimin ilk
    adimlarinda cikti sifira yakinken tam o bolgedeyiz.
    """
    x = cikti.float()
    a1, a2, t = x[..., 0:3], x[..., 3:6], x[..., 6:9]

    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_dik = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_dik, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack((b1, b2, b3), dim=-1)          # sutunlar taban vektorleri
    ust = torch.cat((R, t[..., None]), dim=-1)     # (B,P,3,4)
    alt = torch.zeros_like(ust[..., 0:1, :])
    alt[..., 0, 3] = 1.0
    return torch.cat((ust, alt), dim=-2)


def donusume_matrise(temsil: str, cikti: torch.Tensor, num_pairs: int,
                     **kalib) -> torch.Tensor:
    """Ag ciktisi -> (B, P, 4, 4) donusum matrisleri."""
    x = cikti.reshape(cikti.shape[0], num_pairs, -1)
    if temsil == "6b":
        return altib_donusume(x)
    if temsil == "kuaterniyon":
        # Transforms sinifi kuaterniyonu tanimiyor; PredictionTransform taniyor.
        return PredictionTransform(TEMSILLER[temsil][1], "transform",
                                   num_pairs=num_pairs, **kalib)(cikti)
    return Transforms(pred_type=TEMSILLER[temsil][1], num_pairs=num_pairs,
                      **kalib)(cikti)


class TahminDonusturucu:
    """Ag ciktisini kayip fonksiyonunun bekledigi uzaya cevirir.

    kayip_uzayi="nokta"     -> (B,P,3,4) goruntu kosesinin mm konumu
    kayip_uzayi="parametre" -> (B,P,6)   ZYX Euler acisi + oteleme
    """

    def __init__(self, temsil: str, kayip_uzayi: str, num_pairs: int, **kalib):
        self.temsil = temsil
        self.kayip_uzayi = kayip_uzayi
        self.num_pairs = num_pairs
        self.kalib = kalib
        etiket = {"nokta": "point", "parametre": "parameter"}[kayip_uzayi]
        self._referans = None
        if temsil != "6b":
            # Faz 2 yolu: ceviriyi referans yapiyor, tek bir satir bile degismiyor
            self._referans = PredictionTransform(
                TEMSILLER[temsil][1], etiket, num_pairs=num_pairs, **kalib)
        else:
            self._nokta = torch.matmul(kalib["tform_image_pixel_to_mm"],
                                       kalib["image_points"])

    def __call__(self, cikti: torch.Tensor) -> torch.Tensor:
        if self._referans is not None:
            return self._referans(cikti)
        T = altib_donusume(cikti.reshape(cikti.shape[0], self.num_pairs, -1))
        if self.kayip_uzayi == "nokta":
            return torch.matmul(T, self._nokta)[:, :, 0:3, :]
        import pytorch3d.transforms as p3d
        donme = p3d.matrix_to_euler_angles(T[:, :, 0:3, 0:3], "ZYX")
        return torch.cat((donme, T[:, :, 0:3, 3]), dim=2)


# ---------------------------------------------------------------------------
# BASLANGIC NOKTASI — temsiller arasi ADIL karsilastirmanin sarti
#
# Euler temsilinde agin son katmani sifir uretirse donusum BIRIM matristir:
# aci sifir, oteleme sifir. Rastgele baslatilmis bir ag yaklasik sifir uretir,
# yani egitim "hicbir hareket yok" tahmininden baslar — kare arasi gercek
# hareket 0,4 mm oldugundan bu cok iyi bir baslangic.
#
# 6B, kuaterniyon ve matris temsillerinde sifir cikti BIRIM DEGILDIR:
#   6b          -> Gram-Schmidt sifir vektorleri normlar, rastgele bir donme
#   kuaterniyon -> (0,0,0,0) normsuz, donme matrisi NaN
#   matris      -> sifir matris, hic donusum degil
# Olculdu: 6B ile ilk epok kaybi 10173 (euler'de 182), hata 131 mm (euler'de 16).
#
# Bu fark temsilin KENDISINDEN degil, baslangic noktasindan gelir. Duzeltmeden
# yapilan bir karsilastirma "6B daha kotu" derdi ve yanlis olurdu. Son katmanin
# YANLILIGI her temsilde birim donusumu kodlayacak sekilde kuruluyor; agirliklar
# rastgele kaliyor. Boylece dort temsil de ayni yerden yarisa basliyor ve
# olculen fark gercekten temsilin farki olur.
# ---------------------------------------------------------------------------

BIRIM_YANLILIK = {
    "euler":       [0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
    "kuaterniyon": [1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
    "matris":      [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0],
    "6b":          [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0],
}


def son_katmani_birime_ayarla(model, temsil: str, num_pairs: int) -> None:
    """Son katmanin yanliligini birim donusume kurar, agirliklara dokunmaz."""
    yanlilik = torch.tensor(BIRIM_YANLILIK[temsil] * num_pairs)
    katman = model.classifier[1]
    if katman.bias.shape != yanlilik.shape:
        raise RuntimeError(
            f"son katman {tuple(katman.bias.shape)} bekleniyordu "
            f"{tuple(yanlilik.shape)}")
    with torch.no_grad():
        katman.bias.copy_(yanlilik)
