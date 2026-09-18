"""Omurga agi — referansin build_model'i, uzerine ImageNet secenegi.

FAZ 3'UN TESHISI BURAYI ISARET EDIYOR
  Referans `build_model` agi `efficientnet_b1(weights=None)` ile kuruyor:
  6,5 milyon parametre TAMAMEN RASTGELE baslatiliyor. Faz 2'de bu agi
  128 bin pencereyle egittik ve model regresyon cokusune girdi — egitim
  kaybi 0,13'e indi, ki bu etiket varyansinin (0,386 mm)^2 = 0,149
  kadari: model goruntuye bakmayi hic ogrenmeden veri kumesinin ortalama
  hareketini soylemeyi ogrendi.

  ImageNet agirligi bu baslangici degistirir: kenar, doku ve yerel desen
  suzgecleri hazir gelir, ag sifirdan gormeyi ogrenmek zorunda kalmaz.

SIZINTI MI?
  Hayir — ve bu ayrim Faz 2'de acik bedelle ogrenildi. TUS-REC2024
  agirligi sizinti tasiyordu cunku ELIMIZDEKI VERIYLE egitilmisti; test
  kumemizdeki denekler o egitimin icindeydi. ImageNet'te ultrason yok,
  denek yok, hareket etiketi yok. Dogal goruntu istatistigi tasir,
  bu kumeye ait hicbir bilgi tasimaz.

ILK KATMAN NEDEN OZEL
  Referans ilk evrisimi 3 kanaldan `in_frames` kanala degistiriyor; o
  katmanin ImageNet agirligi bu yuzden dogrudan kullanilamaz. Rastgele
  birakmak ilk katmani yine sifirdan ogrenmeye birakirdi. Bunun yerine
  RGB suzgeclerinin kanal ortalamasi aliniyor ve her kare kanalina
  kopyalaniyor: gri seviyeli girdi icin dogru karsilik bu. Kanal sayisi
  degistigi icin toplam da degisir, o yuzden 3/in_frames ile olcekleniyor —
  ciktinin buyuklugu ImageNet'teki gibi kaliyor, sonraki katmanlarin
  BatchNorm istatistikleri anlamli baslar.
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

from utils.network import build_model  # noqa: E402


def ilk_katmani_uyarla(yeni: torch.nn.Conv2d, eski_agirlik: torch.Tensor) -> None:
    """RGB suzgeclerini gri seviyeli cok kareli girdiye tasir."""
    in_kanal = yeni.in_channels
    ortalama = eski_agirlik.mean(dim=1, keepdim=True)          # (cikti,1,k,k)
    with torch.no_grad():
        yeni.weight.copy_(ortalama.repeat(1, in_kanal, 1, 1) * (3.0 / in_kanal))


def kur(model_name: str, in_frames: int, pred_dim: int,
        onegitimli: bool = False) -> torch.nn.Module:
    """Referansin agi. onegitimli=True ise omurga ImageNet agirligiyla baslar."""
    model = build_model({"model_name": model_name}, in_frames=in_frames,
                        pred_dim=pred_dim)
    if not onegitimli:
        return model

    if model_name != "efficientnet_b1":
        raise ValueError(f"ImageNet uyarlamasi yalnizca efficientnet_b1 icin: {model_name}")
    from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

    kaynak = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
    ilk_agirlik = kaynak.features[0][0].weight.data.clone()

    # siniflandirici disinda her sey birebir ayni mimari; sadece o ikisi ayri
    durum = kaynak.state_dict()
    for anahtar in ("features.0.0.weight", "classifier.1.weight", "classifier.1.bias"):
        durum.pop(anahtar, None)
    eksik, fazla = model.load_state_dict(durum, strict=False)
    beklenen = {"features.0.0.weight", "classifier.1.weight", "classifier.1.bias"}
    if set(eksik) != beklenen or fazla:
        raise RuntimeError(f"ImageNet yuklemesi beklenmedik: eksik={eksik} fazla={fazla}")

    ilk_katmani_uyarla(model.features[0][0], ilk_agirlik)
    return model
