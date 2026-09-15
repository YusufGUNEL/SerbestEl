"""Koordinat sistemleri ve donusum matematigi — kendi uygulamamiz.

Bu modul referansi CAGIRMIYOR, matematigi sifirdan kuruyor. Sebebi:
`tests/test_geometri.py` bu uygulamayi referansin cikardigi sayilara karsi
dogruluyor. Referansi sarmak olsaydi test hicbir sey kanitlamazdi.

DORT KOORDINAT SISTEMI
  goruntu-piksel   satir/sutun, birimsiz.  x = sutun (1..640), y = satir (1..480)
  goruntu-mm       ayni duzlem, milimetre. piksel x olcek
  prob/arac        optik izleyicinin probun uzerine takili aracinin eksen takimi
  kamera/dunya     izleyicinin gordugu sabit cerceve

ZINCIR (sol carpim)
  nokta_arac   = T_rijit . T_olcek . nokta_piksel
  nokta_dunya  = tforms[i] . T_rijit . T_olcek . nokta_piksel

  Buradan kare i'den kare j'ye donusum (goruntu-mm -> goruntu-mm):

      T_{j<-i} = T_rijit^-1 . ( tforms[j]^-1 . tforms[i] ) . T_rijit

  Ortadaki parca arac_i -> arac_j donusumu ve DUNYADAN BAGIMSIZ. Kenarlardaki
  T_rijit ciftini kaldirmak sonucu bozar: o cift sonucu arac ekseninden
  goruntu duzlemine tasiyor.

NEDEN T_olcek ZINCIRIN ICINDE DEGIL
  T_{j<-i} yalnizca rijit parcayi tasiyor, yani **ortogonal** kaliyor. Ortogonal
  olmasi sart, cunku Euler acisina ancak ortogonal matris cevrilebilir.
  T_olcek (kosegen, 1 olmayan olcekler) matrisi ortogonallikten cikarir.
  Bu yuzden olcek zincire degil, NOKTAYA uygulanir:

      nokta_j_mm = T_{j<-i} . T_olcek . nokta_i_piksel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Kalibrasyon:
    """calib_matrix.csv icindeki iki matris ve bilesimleri.

    olcek : (4,4) goruntu-piksel -> goruntu-mm. Kosegen; z ve homojen 1.
    rijit : (4,4) goruntu-mm     -> arac.       Donme + oteleme, ortogonal donme.
    tam   : (4,4) goruntu-piksel -> arac.       = rijit @ olcek
    """

    def __init__(self, olcek: torch.Tensor, rijit: torch.Tensor):
        if olcek.shape != (4, 4) or rijit.shape != (4, 4):
            raise ValueError("iki matris de 4x4 olmali")
        self.olcek = olcek
        self.rijit = rijit
        self.tam = rijit @ olcek

    @classmethod
    def csvden(cls, yol: str | Path) -> "Kalibrasyon":
        """Dosya duzeni: baslik satiri, 4 satir matris, baslik satiri, 4 satir matris."""
        satirlar = Path(yol).read_text().strip().splitlines()
        sayisal = [s for s in satirlar if s and s[0].isdigit() or s.startswith("-")]
        if len(sayisal) != 8:
            raise ValueError(f"8 sayisal satir bekleniyordu, {len(sayisal)} bulundu")
        oku = lambda blok: torch.tensor(  # noqa: E731
            np.array([[float(x) for x in s.split(",")] for s in blok], dtype=np.float32)
        )
        return cls(olcek=oku(sayisal[0:4]), rijit=oku(sayisal[4:8]))

    def to(self, aygit) -> "Kalibrasyon":
        return Kalibrasyon(self.olcek.to(aygit), self.rijit.to(aygit))

    def __repr__(self) -> str:
        sx, sy = self.olcek[0, 0].item(), self.olcek[1, 1].item()
        return f"Kalibrasyon(piksel->mm: x={sx:.6f} y={sy:.6f} mm/piksel)"


def piksel_noktalari(h: int = 480, w: int = 640, yogunluk=None) -> torch.Tensor:
    """Goruntu duzlemindeki nokta izgarasi, homojen: (4, N).

    Satirlar: [x ; y ; 0 ; 1]
      x = sutun, 1'den w'ye   (kalibrasyonla tutarli olmasi icin 1'den basliyor)
      y = satir,  1'den h'ye
      z = 0       butun noktalar ayni duzlemde
    Duzlestirme sirasi SATIR ONCE: indeks = (y-1)*w + (x-1).
    Bu sira degistirilemez — DDF'lerin 307200 uzunlugundaki duzlestirmesi buna bagli.

    yogunluk=None ise her piksel (h*w nokta); (dy, dx) verilirse seyrek izgara.
    """
    if yogunluk is None:
        ny, nx = h, w
    elif isinstance(yogunluk, int):
        ny = nx = yogunluk
    else:
        ny, nx = yogunluk

    y = torch.linspace(1, h, ny)
    x = torch.linspace(1, w, nx)
    # satir once: y yavas, x hizli degisir
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    duz = torch.stack([xx.reshape(-1), yy.reshape(-1)])          # (2, N)
    return torch.cat([duz, torch.zeros(1, duz.shape[1]), torch.ones(1, duz.shape[1])])


def _arac_donusumleri(tforms: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """T_{arac_j <- arac_i} = tforms[j]^-1 @ tforms[i].  tforms: (N,4,4)."""
    return torch.linalg.inv(tforms[j]) @ tforms[i]


def goruntu_donusumleri(
    tforms: torch.Tensor, kalib: Kalibrasyon, i: torch.Tensor, j: torch.Tensor
) -> torch.Tensor:
    """T_{goruntu_j_mm <- goruntu_i_mm}, her (i,j) cifti icin. Cikti (K,4,4).

    Ortogonal donme parcasi korunur — Euler'e cevrilebilir olmasi icin sart.
    """
    arac = _arac_donusumleri(tforms, i, j)
    rijit_ters = torch.linalg.inv(kalib.rijit)
    return rijit_ters[None] @ arac @ kalib.rijit[None]


def kuresel_donusumler(
    tforms: torch.Tensor, kalib: Kalibrasyon, ilk_kare_dahil: bool = False
) -> torch.Tensor:
    """Her kareden ILK kareye donusum.

    ilk_kare_dahil=False (varsayilan): cikti (N-1,4,4), ilk kare haric.
      Referansin DDF gelenegi bu — DDF'ler ilk kareyi icermiyor. Olcum
      kodunun bu bicimi beklemesi yuzunden varsayilan bu.
    ilk_kare_dahil=True: cikti (N,4,4), ilk kare kimlik matrisi olarak basta.
      Hacim kurarken gerekli, cunku ilk kare de hacme yazilmali.

    NOT: bu donusum tforms[0] ve tforms[i]'den DOGRUDAN hesaplaniyor,
    ara kareler zincirlenmiyor. Bu yuzden gercek referansta suruklenme yok
    (tests/test_geometri.py A7 bunu kilitliyor). Suruklenme yalnizca modelin
    kare basina tahmin hatasi zincirlendiginde dogar.
    """
    n = tforms.shape[0]
    bas = 0 if ilk_kare_dahil else 1
    i = torch.arange(bas, n)
    j = torch.zeros(n - bas, dtype=torch.long)
    return goruntu_donusumleri(tforms, kalib, i, j)


def yerel_donusumler(tforms: torch.Tensor, kalib: Kalibrasyon) -> torch.Tensor:
    """Her kareden ONCEKI kareye donusum. Cikti (N-1,4,4)."""
    n = tforms.shape[0]
    i = torch.arange(1, n)
    j = torch.arange(0, n - 1)
    return goruntu_donusumleri(tforms, kalib, i, j)


def noktalari_tasi(
    donusumler: torch.Tensor, noktalar_piksel: torch.Tensor, kalib: Kalibrasyon
) -> torch.Tensor:
    """Piksel noktalarini hedef karenin mm koordinatina tasir.

    donusumler (K,4,4), noktalar_piksel (4,N) -> (K,3,N) milimetre.
    Olcek burada uygulanir, zincirin icinde degil (bkz. modul aciklamasi).
    """
    mm = kalib.olcek @ noktalar_piksel                    # (4,N)
    return (donusumler @ mm[None])[:, 0:3, :]


def yer_degistirme(
    donusumler: torch.Tensor, noktalar_piksel: torch.Tensor, kalib: Kalibrasyon
) -> torch.Tensor:
    """DDF: noktanin tasindiktan sonraki yeri eksi baslangic yeri, mm. (K,3,N).

    Referans tanimi birebir bu: yer degistirme, noktanin KENDI karesindeki
    mm koordinatina gore olculuyor — hedef karedeki karsiligina degil.
    """
    mm = kalib.olcek @ noktalar_piksel
    return noktalari_tasi(donusumler, noktalar_piksel, kalib) - mm[None, 0:3, :]


def isaret_ddf(
    ddf_tum: torch.Tensor, isaretler: torch.Tensor, h: int = 480, w: int = 640
) -> torch.Tensor:
    """Tum piksel DDF'sinden 100 isaret noktasinin DDF'sini secer. Cikti (3,100).

    isaretler (100,3): [kare indeksi, x, y].
    Kare indeksi 0'dan, x ve y 1'den basliyor (kalibrasyonla tutarlilik icin).
    ddf_tum ilk kareyi ICERMEDIGI icin kare f, dizide f-1'e karsilik geliyor.
    """
    d = ddf_tum.reshape(ddf_tum.shape[0], 3, h, w)
    kare = isaretler[:, 0].long() - 1      # ilk kare dizide yok
    sutun = isaretler[:, 1].long() - 1     # x
    satir = isaretler[:, 2].long() - 1     # y
    return d[kare, :, satir, sutun].T
