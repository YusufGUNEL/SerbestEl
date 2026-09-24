# FreeSweep — Sensörsüz 3B Ultrason Rekonstrüksiyonu

> English version: [README.md](README.md) · Önceki adı *SerbestEl*.

Elde gezdirilen ultrason probundan, **konum sensörü olmadan** 3B hacim çıkarma.
Model, ardışık kareler arasındaki rijit dönüşümü yalnızca görüntüden tahmin eder;
dönüşümler zincirleme çarpılarak her karenin ilk kareye göre konumu bulunur.

Kıyas noktası: **TUS-REC2025** (MICCAI 2025 / ASMUS, UCL). Yarışma Ekim 2025'te
kapandı; bu proje yayınlanmış sonuçlara karşı çalışır, sıralamaya girmez.

## Sonuç — tek bakışta

**Sürüklenme hatası (GP) referans sisteme göre %83 azaldı**: 86,90 mm → 14,40 mm,
dokunulmamış test kümesinde (10 denek, 240 tarama). Dört ölçünün dördünde de
yarışmanın kendi referansı geçildi. Tek bir 4 GB dizüstü ekran kartında,
sekiz saatlik eğitimle.

| | GP | GL | LP | LL |
|---|---|---|---|---|
| Referansın yeniden üretimi (Faz 2) | 86,90 | 83,65 | 0,3856 | 0,3850 |
| `U_uzun` (Faz 4) | 17,25 | 16,35 | 0,1565 | 0,1372 |
| **En iyi model (`U_uzun_C`, Faz 6)** | **14,40** | **12,52** | **0,1510** | **0,1313** |
| `U_uzun_C`, ikinci tohum | 15,27 | 13,76 | 0,1548 | 0,1353 |
| *`U_uzun_C`, 2 tohum ortalaması* | *14,84 ± 0,43* | *13,14 ± 0,62* | *0,1529* | *0,1333* |

Birim mm. GP/GL ilk kareye göre (birikmiş hata), LP/LL önceki kareye göre
(adım hatası); ayrıntı: [Ölçüm](#ölçüm).

**Ablasyon — hangi fikir ne getirdi.** Kümülatif merdiven, her satır bir
öncekinin üstüne tek şey ekliyor; dördü de eşit 180 dakika, doğrulama kümesi:

| Satır | Eklenen | GP | LP | \|r\| | Önceki satıra göre GP |
|---|---|---|---|---|---|
| `A1_referans` | referansın kendi ayarları | 93,18 | 0,4007 | 0,031 | (çıpa) |
| `A2_C` | parametre uzayında kayıp | 28,93 | 0,2106 | 0,371 | **%−69,0** |
| `A3_CB` | + 6B sürekli dönme temsili | 21,98 | 0,1953 | 0,366 | %−24,0 |
| `A4_CBG` | + ImageNet omurga | **20,91** | **0,1888** | **0,522** | %−4,9 |

`|r|`, tahmin edilen hareketle gerçek hareketin bağdaşımı: 0'a yakınsa model
görüntüye bakmıyor, ortalamayı söylüyor ("regresyon çöküşü"). Projenin ana
bulgusu bu sütunda: referans yapılandırması üç kat bütçede de çökmüş kalıyor
(0,031); çöküşten çıkaran bütçe değil **yapılandırma**, bütçe yalnızca
çıkışın görülmesini sağlıyor. Ayrıntı: [Faz 4](#faz-4--iyileştirme) ve
[Faz 5](#faz-5--ablasyon-ve-dürüst-rapor).

Yeniden üretim, tek komut (~23 saat, veri indirildikten sonra):

```bash
python scripts/yeniden_uret.py --kaynak D:/SerbestEl-veri/tusrec2024
```

60 saniyelik video (solda kareler, sağda tahminle oluşan hacim, sonda
gerçekle yan yana). Tarama elle seçilmiyor: test GP'si ortancaya en yakın
olan kullanılıyor, yani gösterilen şey tipik bir tarama, en iyisi değil.

```bash
python scripts/video.py --kosum results/faz6/U_uzun_C
```

## Bilinen sınırlar

Neyin çözülmediği de sonucun parçası:

- **Sıralama değil.** Test kümesi kendi denek bazlı bölmemiz (açık eğitim
  verisinden ayrılan 10 denek), yarışmanın kapalı test kümesi değil.
  Liderlik tablosuyla karşılaştırma [gösterge](#yayınlanmış-sonuçlara-göre-konum),
  sıra değil.
- **En iyi model iki tohum, geri kalanı tek tohum.** `U_uzun_C` ikinci tohumla
  yeniden koştu: test GP 14,40 / 15,27 (ortalama 14,84 ± 0,43, referansa göre
  %−82,9). Ana sonuç tutuyor ve iki tohum da `U_uzun`'u (17,25) geçiyor. Aynı
  çift **doğrulamada %31** (14,43 / 18,84), testte yalnız %6 farklı: 60
  taramalık doğrulama kümesi yakın yapılandırmaları sıralamaya yetmiyor.
  240 taramanın tamamında tohum farkı %4 (16,52 / 17,22) ve `U_uzun_CA1`'in
  görünen %11 kaybı beraberliğe dönüşüyor. Merdiven ve Faz 4 satırları 60
  taramada sıralandı: büyük basamaklar (%69, %24) geçerli, küçükler (son
  basamak, %4,9) belirsiz.
- **Sürüklenme bitmedi.** GP/LP oranı 225×'ten 95×'e indi ama hâlâ büyük:
  kalan hata taramaya özgü bir yanlılık. Tek bir sabit düzeltmeyle
  giderilemediği ölçüldü (Faz 3); taramaya uyum sağlayan bir yöntem denenmedi.
- **Kısa taramalarda eğitildi, uzunlarda sınanmadı.** TUS-REC2025 eğitim
  kümesine erişim talebi hâlâ bekliyor; model TUS-REC2024'te (ortanca 547
  kare) eğitilip ölçüldü. 2025'in 1570 karelik dönen taramalarında en iyi
  model ölçülmedi.
- **Bütçe referansın %4'ü.** 866 epok, referansın 20.000'ine karşı. Merdiven
  180 dakikayla sınırlı; en iyi yapılandırma uzun bütçede tek koşumla ölçüldü.
  `U_uzun_C`'nin `U_uzun`'a üstünlüğü (test GP %−16,5) tek tohumluk bir fark.
- **Tutarlılık kısıtı adil sınanmadı.** 60 dakikada 13 epok koşabildi;
  sonucu fikri değil uygulamanın maliyetini yargılıyor.
- **Tek anatomi, tek cihaz.** Veri kümesi önkol taramaları, tek prob ve tek
  protokol. Başka anatomiye ya da cihaza genelleme ölçülmedi.
- **Gerçek zamanlılık iddiası yok.** Çıkarım hızı sistem düzeyinde
  ölçülmedi; rekonstrüksiyon çevrimdışı yapılıyor.

## Belgeler

Tarayıcıda aç:

| Dosya | Ne var |
|---|---|
| `docs/ogrenme.html` | Büyüyen ders notu — her fazın konuları buraya eklenir |

## Kurulum

Windows + Python 3.10 + conda-forge. Referans depo 3.9.13 diyor, ama
conda-forge'da `pytorch3d`'nin **win-64 py39 derlemesi yok** (py310–py314 var).
Eğitim kodu düz PyTorch olduğu için 3.10 sonucu etkilemez.

```bash
conda env create -f environment.yml
conda activate serbestel
python scripts/ortam_dogrula.py     # 9 kontrol, hepsi geçmeli
```

`environment.yml` conda paketlerini, `requirements-lock.txt` pip paketlerini
kilitler. İkisi birlikte ortamın tamamını tanımlar.

### pytorch3d neden sorun değil

Referans depo `pytorch3d`'yi yalnızca `pytorch3d.transforms` için kullanır —
Euler/kuaterniyon ↔ matris çevrimleri, yani **saf Python**. Derlenmiş `_C`
eklentisi hiç içe aktarılmaz; `scripts/ortam_dogrula.py` bunu her çalıştırmada
doğrular. Bu yüzden CUDA derlemesi gerekmez, CPU derlemesi (`cpu_py310*`)
bilinçli olarak seçilmiştir.

## Dizin düzeni

```
src/         kendi kodumuz
configs/     deney ayarları
scripts/     tek seferlik araçlar, doğrulama
notebooks/   keşif ve görselleştirme
results/     ölçüm çıktıları (içerik .gitignore'da)
tests/       testler
docs/        proje sayfaları
data/        veri — .gitignore'da, depoya girmez
reference/   TUS-REC2025 referans sistemi (git submodule, sabit commit)
```

`reference/` **değiştirilmez**. Submodule olarak sabitlenmiştir;
`ortam_dogrula.py` hem commit'i hem de dokunulmamış olduğunu kontrol eder.
Bizim değişikliklerimiz `src/` altına yazılır.

## Veri

| Küme | Erişim | Boyut | Not |
|---|---|---|---|
| TUS-REC2025 eğitim | **kısıtlı** — talep bekliyor | ~22 GB | 50 denek, 100 tarama, ~164k kare |
| TUS-REC2025 doğrulama | **indirildi** | 1,23 GB | 3 denek (050–052), 6 dönen tarama, 1570 kare/tarama |
| TUS-REC2024 eğitim (1–2) | **indirildi ve açıldı** | 83,8 GB → 186 GB | 50 denek (000–049), 1200 tarama, 24/denek |
| TUS-REC2024 doğrulama | indirildi | 4,84 GB | |

Veriler CC-BY-NC-SA-4.0. TUS-REC2025 eğitim kümesi hâlâ kısıtlı erişimde,
bu yüzden **ana eğitim kaynağı TUS-REC2024**. İki kümenin denek numaraları
aynı kişiyi gösteriyor; 2024 eğitimi 000–049, 2025 doğrulaması 050–052, yani
2024'te eğitilen bir modeli 050–052'de ölçmek sızıntısız.

Üç farklı dizin düzeni var ve referans yükleyici yalnızca birini kaldırıyor;
`src/veri.py` üçünü de tanıyor:

| Küme | Düzen | Tarama/denek |
|---|---|---|
| TUS-REC2025 eğitim | `frames_transfs/<denek>/<tarama>.h5` | 2 |
| TUS-REC2025 doğrulama | `frames/` ve `transfs/` **ayrı** | 2 |
| TUS-REC2024 | `<denek>/<tarama>.h5`, sarmalayıcı yok | **24** |

## Ölçüm

Dört yer değiştirme alanı (mm), hepsi raporlanır:

| Kısaltma | Ne | Neden |
|---|---|---|
| `GP` | Tüm pikseller, ilk kareye göre | Sürüklenmeyi bu yakalar |
| `GL` | İşaret noktaları, ilk kareye göre | Klinik nokta doğruluğu, küresel |
| `LP` | Tüm pikseller, önceki kareye göre | Adım adım hareket |
| `LL` | İşaret noktaları, önceki kareye göre | Klinik nokta doğruluğu, yerel |

Tek sayı vermek yanıltıcıdır: yerelde iyi, küresel de kötü olmak bu problemin
klasik tuzağıdır.

### Ölçülen kıyas noktası

Ön eğitimli TUS-REC2024 modeli, TUS-REC2025 dönen doğrulama setinde
(050–052, 6 tarama, sızıntısız, tam 307.200 piksel ızgarası):

| GP | GL | LP | LL | GP/LP |
|---|---|---|---|---|
| 37,88 mm | 28,60 mm | 0,2513 mm | 0,2014 mm | **150,7×** |

`results/faz2_referans_2025val/olculer.json`. Kare başına 0,25 mm hata, 1500
karenin sonunda 38 mm'e çıkıyor. Faz 3 bu 151 katın nereden geldiğini ölçüyor.

## Faz 2 — geçilecek sayıyı üret

Veri indi ve açıldı: **TUS-REC2024, 50 denek, 1200 tarama** (denek başına 24),
`D:\SerbestEl-veri\tusrec2024\acilmis`. Açılmış boyut 186 GB. Tarama uzunluğu
en az 250, ortanca 547, en çok 710 kare — TUS-REC2025'in 1570'inden çok kısa.

Tek komutla yeniden üretim:

```bash
python scripts/faz2_hazirla.py --kaynak D:/SerbestEl-veri/tusrec2024
```

Bu betik zip'leri **paralel** açar (96,5 GB / 7,4 dk), veri düzenini çözer,
denek bazlı bölmeyi üretir, dört test takımını koşar ve eğitimin gerçekten
başladığını 2 epokla doğrular. Tekrar çalıştırmak güvenli: açılmış zip'i
atlar, mevcut bölmeye dokunmaz.

Sonra:

```bash
# eğitim — SIFIRDAN, süre bütçesiyle (aşağıdaki sızıntı notuna bak)
python src/egit.py --veri D:/SerbestEl-veri/tusrec2024/acilmis \
    --on-egitimli yok --isci 6 --sure-siniri 210 --epok 100000
python src/egit.py --devam                   # kesilirse kaldığı yerden

# dört ölçü — projenin omurgası olan tablo
python src/olcum.py --veri D:/SerbestEl-veri/tusrec2024/acilmis \
    --agirlik results/faz2_referans/en_iyi_model.pt --kume test \
    --cikti results/faz2_referans/olculer.json
```

**Test kümesine eğitim sırasında dokunulmaz.** `src/egit.py` bölmeyi yükler,
test deneklerini ayırır ve eğitimde kullanmaz.

### Faz 2 çıktısı — dört sayı

Kendi eğittiğimiz model (sıfırdan, 178 epok / ~90 dk, en iyi doğrulama
mesafesi 0,4022 mm @ epok 150), **2024 test kümesi**: 10 denek, 240 tarama,
tam 307.200 piksel ızgarası.

| Model | Test kümesi | GP | GL | LP | LL | GP/LP |
|---|---|---|---|---|---|---|
| **Bizim (sıfırdan)** | 2024 test, 240 tarama | **86,90** | **83,65** | **0,3856** | **0,3850** | **225×** |
| Ön eğitimli 2024 | 2025 dönen doğrulama, 6 tarama | 37,88 | 28,60 | 0,2513 | 0,2014 | 151× |

Birim mm. `results/faz2_referans/olculer.json`,
`results/faz2_referans_2025val/olculer.json`.

**İki satır doğrudan karşılaştırılamaz** — farklı protokol, farklı tarama
uzunluğu, farklı eğitim bütçesi. Yan yana durmalarının sebebi ölçek vermek:
kare başına 0,39 mm hata, 550 karenin sonunda 87 mm'e çıkıyor.

Eğitim epok 55'ten sonra 0,40–0,44 mm bandında düzleşti ve 27 epok boyunca
iyileşmedi; koşum orada kesildi (`results/faz2_referans/egitim_egrisi.png`).
Referansın 20.000 epokluk bütçesinin yanında bu **%0,9**'luk bir bütçe —
model eksik eğitilmiş, sayı bu şartla okunmalı.

Tarama tipine göre kırılım şaşırtıcı biçimde düz: sol/sağ kol 88,4 / 85,4 mm,
paralel/dik prob 85,7 / 88,1 mm, C/L/S yörünge 86,0 / 88,4 / 86,4 mm.
Kare sayısıyla GP bağdaşımı yalnızca **0,24**, LP ile GP bağdaşımı **0,55** —
yani sürüklenme ne tarama uzunluğunun ne de kare başına hatanın basit bir
sonucu. Faz 3 farkın nereden geldiğini ölçüyor.

### Bölme sabit

`configs/bolme.json` — 30 eğitim / 10 doğrulama / 10 test denek, tohum
20260915. **Bu dosya bir daha değişmez.** Değişirse önceki bütün ölçümler
karşılaştırılamaz hâle gelir; `src/bolme.py` üzerine yazmak için açıkça
`--yeniden-uret` ister.

### Neden sıfırdan eğitiliyor — ön eğitimli ağırlık da sızıntıdır

Referans `train.py` eğitime sıfırdan başlamıyor: `efficientnet_b1(weights=None)`
kuruyor, sonra hemen üzerine `TUS-REC2024_model/model_weights` yüklüyor. O
ağırlıklar **TUS-REC2024 eğitim kümesinde** eğitilmiş — yani elimizdeki 50
deneğin hepsinde.

Bölmeyi denek bazında yapmak bunu çözmez: test deneklerimizi eğitimden
ayırsak bile yüklediğimiz ağırlık onları zaten görmüştür. Skor şişer, hata
mesajı çıkmaz. Bu yüzden Faz 2 eğitimi `--on-egitimli yok` ile yapılıyor.

Bedeli açık: referansın 20.000 epokluk bütçesi bu donanımda günler sürer,
dolayısıyla modelimiz eksik eğitilmiş oluyor. `--sure-siniri` deneyi epok
yerine **süreyle** sabitliyor, böylece bütçe raporlanabilir bir sayı oluyor.

Ön eğitimli ağırlık yine de ölçülüyor, ama rolü açıkça etiketli:

| Model | Test kümesi | Temiz mi | Rolü |
|---|---|---|---|
| Ön eğitimli 2024 | 2024 test denekleri | **hayır — sızıntılı** | Üst sınır |
| Ön eğitimli 2024 | 2025 dönen doğrulama (050–052) | evet | Gerçek kıyas noktası |
| Bizim (sıfırdan) | 2024 test denekleri | evet | Faz 2'nin asıl sayısı |

050–052'nin temiz olduğu varsayılmadı, kontrol edildi: yarışma belgesi
*"patient IDs are consistent across datasets"* diyor, TUS-REC2024 eğitim
kümesi 000–049 — kesişim boş.

### Neden kendi eğitim döngümüz var

Referans `train.py` Windows'ta koşmuyor: bütün kod modül seviyesinde,
`if __name__ == "__main__"` koruması yok, ama DataLoader `num_workers=8` ile
kuruluyor. Windows `spawn` kullandığı için her işçi süreç betiği baştan
çalıştırıp yeni süreç doğurmaya kalkıyor. Linux'ta `fork` olduğu için orada
hiç görünmeyen bir kusur.

`src/egit.py` ağı, kaybı, etiket dönüşümlerini ve veri yükleyiciyi
referanstan **doğrudan ithal ediyor** — matematiğe dokunulmadı. Değişen
yalnızca çalıştırma biçimi: koruma, AMP, gradyan biriktirme, denek bazlı
bölme, sürdürülebilirlik.

Gradyan biriktirme matematiği değiştirmez: 8'lik iki yığından gelen gradyanlar
toplanıp tek adımda uygulanıyor, 16'lık tek yığın gibi. Referans ayarının
(`MINIBATCH_SIZE=16`) 4 GiB'deki tam karşılığı budur.

### Ölçümde referanstan iki ayrılış — matematik aynı

| | Referans | Bizde | Kanıt |
|---|---|---|---|
| İleri geçiş | Kare kare, yığın 1 | Pencereler toplu | `test_olcum.py` T1: yığın 1'de fark **tam sıfır**; T1b: dört sayıya etkisi %0,04'ün altında |
| Ölçüm | Dört DDF'yi bellekte kurar (~11,6 GB) | DDF kurmadan, blokla (~88 MB) | T2: dört sayı referansla aynı (bağıl fark 4·10⁻⁷) |

İkincisi bir cebir sadeleşmesine dayanıyor: ölçülen şey iki DDF'nin farkı ve
`DDF_gerçek − DDF_tahmin = T_gerçek·p − T_tahmin·p` — noktanın kendi konumu
sadeleşiyor, DDF'leri kurmaya hiç gerek kalmıyor.

## Faz 3 — hata analizi

Yol haritası bu fazdan **tek bir cümle** istiyor: "hata şu durumda, şu sebepten
birikiyor." Ölçüm: `python src/analiz.py`, 240 test taraması, 48 taramalık
ayrı bir yanlılık kümesi (`results/faz3_bizim_2024test/`).

### Teşhis

> **Model kareye özgü hareketi tahmin etmiyor; girdiden bağımsız, neredeyse
> sabit bir dönüşüm üretiyor. Her taramanın kendi ortalama hareketi bu
> sabitten farklı olduğu için fark tarama boyunca aynı yönde kalıyor,
> zincirde doğrusal birikiyor ve küresel hatanın %83'ünü oluşturuyor.**

Dört bağımsız ölçüm aynı yeri gösteriyor:

**1. Tahmin gerçek hareketle ilişkili değil.** Bileşen bileşen
`tahmin ≈ a·gerçek + b` uydurması, altı bileşenin hepsinde:

| | öteleme x | öteleme y | öteleme z | dönme x | dönme y | dönme z |
|---|---|---|---|---|---|---|
| a | −0,0001 | −0,0001 | −0,0001 | 0,0001 | 0,0081 | −0,0001 |
| r | −0,015 | −0,008 | −0,020 | +0,000 | +0,023 | −0,000 |

Tarama başına bakıldığında da aynı: `|r|` ortancası 0,06–0,14.

**Kod bunu uydurmuyor — kontrol koşuldu.** Ön eğitimli referans modeli
**aynı** veride, **aynı** test kümesinde, **aynı** kod yolundan geçirdik
(`results/faz3_onegitimli_2024test_SIZINTILI/`):

| Bileşen | öteleme x | öteleme y | öteleme z | dönme x | dönme y | dönme z |
|---|---|---|---|---|---|---|
| Ön eğitimli, a | 0,732 | 0,301 | 0,580 | 0,089 | 0,123 | 0,289 |
| Ön eğitimli, r | **+0,844** | **+0,529** | **+0,772** | +0,268 | +0,309 | +0,492 |
| Bizim, r | −0,015 | −0,008 | −0,020 | +0,000 | +0,023 | −0,000 |

Ölçüm, bağdaşım varken onu görüyor. Bizim modelimizde yok.

> **Uyarı:** ön eğitimli satır **sızıntılı** — o ağırlıklar bu 10 test
> deneğini de içeren kümede eğitildi. GP/LP değerleri iyimser, doğruluk
> kıyaslaması için kullanılamaz. Burada kullanılan şey hatanın *yapısı*:
> bağdaşımın varlığı sızıntıdan etkilenmez, çünkü sızıntı bağdaşımı
> yaratmaz — yalnızca büyütür.

**2. Büyüme doğrusal, karekök değil.** Doğrusal artık **0,00063**, karekök
artık **0,02843** — 45 kat daha iyi uyum. Yansız gürültü `√N` ile büyürdü;
`N` ile büyümek yanlılığın imzası (`suruklenme_egrileri.png`).

**3. Yanlılığı geri almak sürüklenmenin %83'ünü siliyor.**

| | GP | Değişim | Kaç taramada iyileşti |
|---|---|---|---|
| Ham | 86,93 mm | — | — |
| Kehanet (taramanın kendi yanlılığı) | **14,41 mm** | **−%83,4** | **240 / 240** |
| Dürüst (başka deneklerde ölçülen tek yanlılık) | 87,86 mm | +%1,1 | 118 / 240 |

**Bu tablodaki asıl bilgi ikinci satır değil, üçüncü satır.** Kehanet
sürümü *her* taramada iyileştiriyor; dürüst sürüm yazı tura. Demek ki
yanlılık **taramaya özgü**, modelin sabit bir kusuru değil — tek bir global
düzeltmeyle giderilemez.

Ve bu, çökmüş modele özgü bir durum **değil**. Ön eğitimli model aynı test
kümesinde aynı yapıyı gösteriyor:

| Model | GP | Kehanet | Dürüst | Kehanet kaç taramada iyileştirdi |
|---|---|---|---|---|
| Bizim (temiz) | 86,93 mm | 14,41 mm (−%83) | 87,86 mm (+%1) | 240 / 240 |
| Ön eğitimli (sızıntılı) | 10,34 mm | 4,91 mm (−%53) | 10,54 mm (+%2) | 231 / 240 |

İki model arasında sekiz kat doğruluk farkı var, ama **ikisinde de**: büyüme
doğrusal, tarama başına yanlılığı gidermek sürüklenmenin yarısından fazlasını
siliyor, ve tek bir global yanlılık hiçbir şey kazandırmıyor. Yani *tarama
başına yanlılık* bu yöntemin yapısal özelliği, bizim modelimizin arızası
değil.

**4. Görsel doğruluyor.** En kötü taramaların yörünge çizimlerinde gerçek
yol 210 mm boyunca kıvrılıp geri dönerken tahmin **kısa ve dümdüz bir
çizgi** (`results/faz3_bizim_2024test/en_kotu/`). Sabit bir dönüşümü
zincirlemek tam olarak düz bir çizgi verir.

Yan ölçümler: kare başına hata hızla bağdaşımı **+0,90** (en yavaş %10
karede 0,19 mm, en hızlı %10'da 0,64 mm). Sıçrama yalnızca 38/240 taramada
var — sürüklenme düzgün, ani olaylardan gelmiyor.

### Bunun Faz 4 için anlamı

**Birinci öncelik — çöküşten çıkmak.** Bizim modelimiz bu bütçede regresyon
çöküşünde: veri kümesinin ortalama hareketini öğrenmiş, kareye özgü
hareketi değil. Sürüklenmeye özgü hiçbir fikir (tutarlılık kısıtı, uzun
zamansal bağlam, dönme temsili) bu haldeki bir modelde anlamlı ölçülemez —
ölçtüğün şey fikrin katkısı değil, çöküşün derinliği olur. Daha uzun
eğitim, öğrenme oranı azaltması, ve **her koşumda bağdaşım kontrolü**.

**İkinci öncelik — dönme.** Ön eğitimli model bile dönmeyi öteleme kadar
iyi görmüyor: öteleme ölçeği 0,30–0,73 iken dönme ölçeği **0,09–0,29**.
Dönme sistematik olarak küçümseniyor. Yol haritasının B maddesi (dönme
temsilini değiştirmek) buraya doğrudan oturuyor ve artık **ölçülmüş bir
gerekçesi** var.

**Hedef olmayan — global sürüklenme düzeltmesi.** Test edildi ve iki modelde
de işe yaramadı (+%1 ve +%2). Yanlılık gerçek ve büyük ama taramaya özgü;
tek bir sabit düzeltmeyle giderilemez.

### Ölçülen tuzak: tek kapı yetmedi

Ölçek düzeltmesine önce yalnızca `|r| ≥ 0,5` kapısı kondu. Bütün
bileşenlerde `a ≈ 0,0001` iken 240 taramanın birkaçında bir bileşenin `r`'si
şans eseri eşiği geçti (tarama başına en büyük `|r|` = 0,761); 0,0001'e
bölünce o taramaların GP'si patladı ve **küme ortalaması 12.669 mm'ye
çıktı**. Tek bir geçersiz tarama 240'lık ortalamayı tek başına bozuyor.

Kapı artık `a`'yı da sınırlıyor (`0,2 ≤ a ≤ 5`); `tests/test_analiz.py` A6
beş ayrı geçersiz durumu kilitliyor. Düzeltilmiş kapıyla aynı koşum:
**86,934 → 86,934 mm, %0,0** — hiçbir bileşen kapıdan geçmiyor.

Yorumlanabilir cevap bu. 12.669 mm "düzeltme sonucu" değil, sıfıra
bölmenin gürültüsüydü; %0,0 ise gerçek bilgiyi veriyor: **düzeltilecek bir
ölçek yok, çünkü ortada sinyal yok.** Bir teşhis aracının "hiçbir şey
yapamıyorum" diyebilmesi, uydurma bir sayı üretmesinden iyidir.

## Faz 4 — iyileştirme

Yol haritası bu fazdan iki şey istiyor: **referansı geçen en az bir yöntem** ve
**her denemenin kaydı — çalışmayanlar dahil**. İkincisi birincisi kadar önemli;
"şu fikir şu kadar getirdi, şu fikir hiçbir şey getirmedi" diyebilmek Faz 5'in
ablasyon tablosunun tek hammaddesi.

Tek komutla yeniden üretim:

```bash
python scripts/faz4_deney.py --liste                 # deneyleri gör
python scripts/faz4_deney.py taban G_imagenet B_6b A_baglam \
    D_tutarlilik C_parametre L_plato --sure 60       # hepsini sırayla koş
python scripts/faz4_tablo.py                         # tabloyu üret
```

### Protokol — dört kural

**1. Tek değişken.** Her deney tabandan yalnızca kendi bayrağıyla ayrılıyor.
Deneyler `scripts/faz4_deney.py` içinde adlarıyla duruyor; hangi bayrağın
hangi deneye ait olduğu koddan okunuyor, elle yazılmıyor.

**2. Bütçe süre, epok değil.** Referansın 20.000 epokluk bütçesi bu donanımda
günler sürer. Epok saymak ayrıca yanıltıcı: bağlam uzunluğu ya da tutarlılık
kısıtı değişince bir epoğun maliyeti değişir. Herkese **60 dakika** veriliyor;
"aynı donanımda aynı süre verildiğinde hangisi daha iyi" sorusu, bir fikrin
pratikte işe yarayıp yaramadığını soran doğru sorudur. Bir fikrin gizli bedeli
tabloda **epok** sütununda görünür.

**3. Sıralama doğrulama kümesinde, test kümesine bir kez dokunulur.** Yedi
deney `dogrulama` kümesinde (denek başına 6 tarama, 60 tarama) sıralanıyor.
Yalnızca kazanan yöntem, bir kere, 240 taramalık test kümesinde ölçülüp Faz
2'nin dört sayısıyla karşılaştırılıyor. Yedi deneyin en iyisini test skoruna
göre seçmek test kümesini eğitim kümesine çevirirdi.

**4. Her koşumda bağdaşım kontrolü.** Faz 3'ün teşhisi buydu: kayıp eğrisi
düzgün inerken model tamamen çökmüş olabilir. Çökmüş modelde hiçbir fikrin
katkısı anlamlı ölçülemez. Tabloda `|r|` sütunu bu yüzden dört ölçünün yanında
duruyor — önce oraya bakılır.

### Ölçülen tuzak: temsiller aynı yerden başlamıyor

Euler temsilinde ağın son katmanı sıfır üretirse dönüşüm **birim matristir**:
açı sıfır, öteleme sıfır. Rastgele başlatılmış bir ağ yaklaşık sıfır üretir,
yani eğitim "hiç hareket yok" tahmininden başlar — kare arası gerçek hareket
0,4 mm olduğundan bu çok iyi bir başlangıç. 6B, kuaterniyon ve matris
temsillerinde sıfır çıktı birim **değildir**: 6B'de Gram-Schmidt sıfır
vektörleri normlayıp rastgele bir dönme verir, kuaterniyonda (0,0,0,0) normsuz
kalır, matriste sıfır matris hiç dönüşüm değildir.

Ölçüldü — ilk epok, aynı veri, aynı bütçe:

| Temsil | İlk epok kaybı | İlk epok hatası |
|---|---|---|
| euler | 182 | 16,0 mm |
| 6B (düzeltmeden) | **10.173** | **131,4 mm** |
| 6B (birimden başlatılmış) | 240 | 15,3 mm |

Bu fark temsilin kendisinden değil, başlangıç noktasından geliyor. Düzeltmeden
yapılan bir karşılaştırma "6B daha kötü" derdi ve yanlış olurdu. Son katmanın
**yanlılığı** her temsilde birim dönüşümü kodlayacak biçimde kuruluyor,
ağırlıklar rastgele kalıyor (`src/temsil.py`, `son_katmani_birime_ayarla`).
`tests/test_temsil.py` hem bunu hem de euler yolunun referansla **bit bit aynı**
kaldığını kilitliyor — o yol kırılırsa Faz 2 ile Faz 4 karşılaştırılamaz.

### Denenen dokuz yol

İlk yedisi aynı 60 dakikalık bütçeyle, tek değişkenle koştu. Son ikisi o
yedisinin **sonucundan doğdu** — biri katkıların toplanıp toplanmadığını,
diğeri geriye kalan tek ekseni ölçüyor.

| Deney | Hat | Ne değişti | Neden |
|---|---|---|---|
| `taban` | — | (çıpa) | Faz 2 koşumu, Faz 4 koşullarında yeniden |
| `G_imagenet` | G | Omurga ImageNet ağırlığıyla başlıyor | Faz 3'ün ana teşhisi regresyon çöküşüydü; 6,5M parametre rastgeleden başlıyor |
| `B_6b` | B | 6B sürekli dönme temsili | Faz 3 ölçtü: dönme ötelemeden belirgin kötü |
| `A_baglam` | A | 2 kare yerine 5 kare, 10 çift | Tek karelik gürültü 0,4 mm'lik sinyalin yanında büyük |
| `D_tutarlilik` | D | İleri/geri tutarlılık kısıtı | Ek etiket istemez, doğrudan hata birikmesini hedefler |
| `C_parametre` | C | Kayıp mm yerine ham parametrede | C hattı tabanda zaten var; bu onun **negatif kontrolü** |
| `L_plato` | L | Doğrulama düzelmeyince lr yarıya | Faz 3'ün birinci önceliği "çöküşten çıkmak" |
| `S_ezber` | tanı | Eğitim kümesi 720 → 24 tarama | Yöntem değil **tanı**: aynı bütçe, otuz kat yoğun tekrar. Ağ burada da ortalamayı söylerse sorun veri ölçeğinde değil |
| `K_birlesik` | B+G | 6B + ImageNet birlikte | B ve G ayrı ayrı benzer kazanç verdi; **katkılar toplanıyor mu?** |
| `U_uzun` | ölçek | Kazanan yapılandırma, sekiz kat bütçe | Beş eksen elendikten sonra geriye kalan tek eksen: görülen pencere sayısı |

`S_ezber` bölmesi eğitim ve doğrulamayı **bilerek** aynı deneğe koyuyor;
genelleme ölçmez, bu yüzden sonuçları yöntem tablosunda değil tanı satırında
durur. Sızıntı kontrolü kapatılmadı — `src/bolme.py` yalnızca bölme dosyasının
içinde açıkça işaretlenmiş tanı bölmelerini muaf tutuyor ve her koşumda uyarı
basıyor.

`U_uzun` karşılaştırma tablosunun **dışındadır**: sekiz kat bütçeyle koştuğu
için diğer sekiziyle aynı ölçekte değil. Onun işi bir yöntemi diğerlerine
karşı sıralamak değil, bütçe ekseninin kendisini ölçmek.

**C hattı neden negatif kontrol:** yol haritası "kayıp fonksiyonunu nokta
tabanlı yap" diyor, ama referans kaybı zaten öyle — `pred_type=parameter` ile
`label_type=point` birleşince tahmin edilen dönüşüm görüntü köşelerine
uygulanıyor ve hata **milimetre** cinsinden alınıyor. Fikrin değeri ancak
kaldırılınca sayıya döner.

### Faz 4 çıktısı — dokuz deneyin kaydı

Doğrulama kümesi, denek başına 6 tarama (60 tarama). Test kümesine bu tabloyu
üretirken **dokunulmadı**.

| Deney | Hat | GP | GL | LP | LL | GP/LP | \|r\| | Epok | GP iyileşmesi |
|---|---|---|---|---|---|---|---|---|---|
| `taban` | — | 93,963 | 93,108 | 0,4131 | 0,4231 | 227× | 0,035 | 108 | (çıpa) |
| `U_uzun` | ölçek | **15,377** | **13,587** | **0,1724** | **0,1578** | **89×** | **0,545** | 912 | **%+83,6** |
| `C_parametre` | C | 58,429 | 46,904 | 0,3014 | 0,2956 | 194× | 0,183 | 129 | %+37,8 |
| `S_ezber` | tanı | *75,221* | *81,681* | *0,4958* | *0,5247* | *152×* | *0,066* | 2882 | *(sızıntılı)* |
| `K_birlesik` | B+G | 86,814 | 87,105 | 0,4043 | 0,4185 | 215× | 0,013 | 119 | %+7,6 |
| `B_6b` | B | 88,124 | 89,226 | 0,4092 | 0,4219 | 215× | 0,041 | 115 | %+6,2 |
| `G_imagenet` | G | 88,628 | 89,293 | 0,4095 | 0,4190 | 216× | 0,020 | 112 | %+5,7 |
| `A_baglam` | A | 91,753 | 92,402 | 0,3988 | 0,4129 | 230× | 0,026 | 113 | %+2,4 |
| `L_plato` | L | 93,963 | 93,108 | 0,4131 | 0,4231 | 227× | 0,035 | 130 | %+0,0 |
| `D_tutarlilik` | D | 125,815 | 106,988 | 1,0441 | 0,9815 | 120× | 0,019 | **13** | %−33,9 |

`U_uzun` sekiz kat bütçeyle koştu, diğerleri eşit 60 dakikayla; aynı sütunda
durması karşılaştırma değil **ölçek eksenini** göstermek için.
`S_ezber` bölmesi bilerek sızıntılı, sayıları eğik yazılı.

Tablo elle yazılmıyor: `python scripts/faz4_tablo.py` her deneyin kendi
dosyalarından üretiyor, atlamıyor, başarısız deneyi de satır olarak yazıyor.

### Faz 4'ün ana bulgusu: çöküş bir faz geçişiyle bitiyor

Beş bağımsız eksen — başlangıç ağırlığı, dönme temsili, bağlam uzunluğu,
kayıp uzayı, veri yoğunluğu — tek tek denendi. Hepsi GP'yi oynattı, **hiçbiri
`r`'yi kıpırdatmadı**. Geriye kalan tek ekseni, görülen pencere sayısını,
ölçmek için kazanan yapılandırma sekiz kat bütçeyle koşuldu ve her 50 epokta
ağırlıklar ayrı dosyaya yazıldı. 19 anlık görüntünün bağdaşımı:

| Epok | Görülen pencere | \|r\| ortanca | \|r\| en büyük |
|---|---|---|---|
| 0 | 0 | 0,052 | 0,140 |
| 100 | 72.000 | 0,024 | 0,197 |
| 150 | 108.000 | 0,020 | 0,236 |
| **200** | **144.000** | **0,025** | **0,297** |
| **250** | **180.000** | **0,173** | **0,929** |
| 300 | 216.000 | 0,224 | 0,952 |
| 500 | 360.000 | 0,490 | 0,966 |
| 900 | 648.000 | **0,572** | **0,972** |

Grafik: `results/faz4/U_uzun/bagdasim_egrisi.png`

**Bu kademeli bir iyileşme değil, bir geçiş.** Model 200 epok boyunca
bağdaşım sıfıra yakın halde duruyor, sonra epok 200–250 arasında aniden
çıkıyor: `|r|` en büyük 0,297 → 0,929.

Ve buradan Faz 4'ün tamamı açıklanıyor: **60 dakikalık bütçe epok ~115'te
bitiyor — çıkış noktasının yarısında.** Dokuz deneyin dokuzu da ölçümünü
çöküş havuzunun *içinde* yaptı. Havuzun dibinde hangi fikir denenirse
denensin, ölçülen şey fikrin katkısı değil havuzun derinliği oluyor. Beş
bağımsız eksenin aynı küçük kazancı vermesi ve katkıların toplanmaması
(6,2 + 5,7 → 11,9 değil 7,6) bunun ikinci kanıtı: hepsi aynı şeyi, sabitin
biraz iyileşmesini, farklı yoldan yapıyordu.

### Kazanan yöntem — test kümesi, tek seferlik ölçüm

`U_uzun`, 240 taramalık dokunulmamış test kümesinde, Faz 2'nin dört sayısına
karşı:

| | Faz 2 (referans) | Faz 4 (`U_uzun`) | Değişim |
|---|---|---|---|
| GP | 86,9035 mm | **17,2520 mm** | **%−80,1** |
| GL | 83,6517 mm | **16,3516 mm** | **%−80,5** |
| LP | 0,3856 mm | **0,1565 mm** | %−59,4 |
| LL | 0,3850 mm | **0,1372 mm** | %−64,4 |
| GP/LP | 225× | **110×** | |

Kıyas noktası Faz 3'ten geliyor: **sızıntılı** ön eğitimli referans modeli
aynı test kümesinde LP 0,1381 mm veriyordu. Bizim modelimiz 0,1565 —
test deneklerini eğitiminde görmüş bir modelin %13 yakınında, **sızıntısız**.
Bağdaşımda da aynı tablo: o modelin `r` değerleri 0,844 / 0,529 / 0,772,
bizimkinin en büyüğü 0,972.

### Çalışmayanlar — ve neden çalışmadıkları

**`L_plato` hiçbir şey yapmadı, tam olarak sıfır.** Dört ölçü de ondalığına
kadar tabanla aynı. Sebep `olcumler.csv`'nin `lr` sütununda: öğrenme oranı
60 dakika boyunca `1,00e-04`'te kaldı, plato tetikleyicisi **bir kez bile**
devreye girmedi — doğrulama hep azıcık iyileşmeye devam etti, sabrı hiç
tüketmedi. Faz 3'ün "çöküşten çıkmak için lr azalt" önerisi, azaltılacak bir
plato olmadığı için boşa çıktı. Yan ürün: koşumların deterministik olduğu
kanıtlandı.

**`D_tutarlilik`'in sayısı fikri değil uygulamayı yargılıyor.** GP 125,8 ile
tablonun en kötüsü, ama epok sütunu asıl hikâyeyi anlatıyor: 60 dakikada
sadece **13 epok**. Tahmin edilen bedel ikinci ileri geçişten gelen ~2 kattı;
ölçülen ~8 kat. Model daha eğitimin en başındayken bütçe doldu. Bu satır
"tutarlılık kısıtı işe yaramıyor" demiyor, "bu uygulamayla eşit süre
bütçesinde şansı yok" diyor — ve fark önemli, çünkü kısıtın kendisi
`tests/test_temsil.py` T6'da doğru çalıştığı kanıtlanmış durumda. Kısıtı
adil sınamak için önce maliyetinin nereden geldiği bulunmalı.

**`A_baglam` yerelde kazandı, küresele taşıyamadı.** LP 0,3988 ile 60
dakikalık koşumların en iyi yerel doğruluğu, ama GP kazancı en düşük (%2,4)
ve GP/LP oranı tabandan **kötü** (230× vs 227×). Faz 1'in bulgusunun
tekrarı: sürüklenmeyi hatanın büyüklüğü değil yanlılığı belirliyor.

### Beklentiyi tersine çeviren: C hattı

Yol haritası "kayıp fonksiyonunu nokta tabanlı yap" diyor. Referans kaybı
zaten öyle olduğu için bu deney hattın **negatif kontrolü** olarak kondu:
kaldırılınca ne kaybedildiği ölçülsün diye. Kaybedilmedi, kazanıldı —
`C_parametre` eşit bütçedeki en iyi sonuç:

| | `taban` (nokta, mm) | `C_parametre` (ham parametre) |
|---|---|---|
| GP | 93,963 | **58,429** (%−37,8) |
| LP | 0,4131 | **0,3014** |
| \|r\| | 0,035 | **0,183** (beş katı) |

Sebep ölçülebilir: mm cinsinden nokta kaybı **ötelemeye ağırlık veriyor**,
çünkü büyüklük orada. Parametre kaybı altı bileşeni daha dengeli tartıyor ve
dönmeye düşen pay artıyor — Faz 3'ün "dönme ötelemeden belirgin kötü
öğreniliyor" bulgusunun doğrudan karşılığı.

Bu, yol haritasının tavsiyesinin **bu bütçede tersine işlediği** anlamına
geliyor. Nokta tabanlı kaybın kendi gerekçesi hâlâ geçerli (hata,
değerlendirme ölçüsüyle aynı birimde olur); ama çöküş havuzundan çıkma
hızını belirleyen şey birim değil, bileşenler arası ağırlık dağılımı.

### Faz 5 için bırakılanlar

1. **Parametre kaybı + uzun bütçe birlikte denenmedi.** `U_uzun` nokta
   tabanlı kayıpla koştu; `C_parametre` eşit bütçede onu %37,8 geçti. İkisinin
   bileşimi Faz 5'in ablasyon tablosunun ilk satırı — ve Faz 4'ün en olası
   açık kazancı.
2. **`D_tutarlilik`'in maliyeti araştırılmalı.** 8 kat yavaşlama ikinci ileri
   geçişle açıklanmıyor; muhtemel kaynak her yığında yeniden kurulan dönüşüm
   nesneleri (her biri üç 4×4 matris tersliyor ve GPU'yu eşitliyor).
   Düzeltilirse kısıt adil sınanabilir.
3. **Geçiş noktası bir kez ölçüldü.** Faz geçişinin epok 200–250'de olması
   bu yapılandırmaya mı özgü, yoksa bütün yapılandırmalarda benzer bir yerde
   mi olduğu bilinmiyor. `C_parametre`'nin 129 epokta `r` = 0,183'e ulaşması,
   onun geçişe **daha erken** girdiğini düşündürüyor.

## Faz 5 — ablasyon ve dürüst rapor

Yol haritası bu fazdan üç şey istiyor: her fikrin katkısını **kümülatif** olarak
ölçen bir tablo, çalışmayanların gerekçeli listesi, ve yayınlanmış sonuçlara
göre konum.

```bash
python scripts/faz4_deney.py A1_referans A2_C A3_CB A4_CBG \
    --sure 180 --kok results/faz5
python scripts/faz4_tablo.py --kok results/faz5
```

### Neden ablasyon 60 dakikada yapılamazdı

Faz 4'ün ana bulgusu bu fazın tasarımını belirledi: çöküş kademeli bitmiyor,
epok 200–250 arasında bir **faz geçişiyle** bitiyor, ve 60 dakikalık bütçe
epok ~115'te — geçişin yarısında — bitiyor.

Bunun doğrudan sonucu: **çöküş havuzunun içinde yapılan bir ablasyon hiçbir
şey ölçmez.** Her satır, fikrin katkısını değil havuzun derinliğini raporlar.
Faz 4'ün tablosu tam olarak bunu yaşadı — beş bağımsız eksen aynı küçük
kazancı verdi ve katkılar toplanmadı.

Bu yüzden merdivenin dört satırı da **180 dakika** ile koştu: en yavaş
yapılandırmada bile ~300 epok, yani bütün satırlar geçişin ötesinde. Faz 4'ün
60 dakikalık tablosuyla yan yana okunmamalı; bu yüzden ayrı dizinde
(`results/faz5/`) duruyor.

### Kümülatif merdiven — hangi fikir kaç puan getirdi

Basamak sırası Faz 4'ün eşit bütçeli sonuçlarına göre büyükten küçüğe:
C (%37,8) → B (%6,2) → G (%5,7).

| Satır | Eklenen | GP | GL | LP | LL | GP/LP | \|r\| | Epok | Bir önceki satıra göre GP |
|---|---|---|---|---|---|---|---|---|---|
| `A1_referans` | — (referansın ayarları) | 93,182 | 93,984 | 0,4007 | 0,4128 | 233× | **0,031** | 360 | (çıpa) |
| `A2_C` | parametre uzayında kayıp | 28,928 | 27,830 | 0,2106 | 0,2029 | 137× | **0,371** | 332 | **%−69,0** |
| `A3_CB` | + 6B sürekli dönme temsili | 21,984 | 20,353 | 0,1953 | 0,1868 | 113× | 0,366 | 322 | %−24,0 |
| `A4_CBG` | + ImageNet omurga | **20,909** | **18,894** | **0,1888** | **0,1681** | **111×** | **0,522** | 412 | %−4,9 |

Toplam: GP 93,18 → 20,91 (**%−77,6**), doğrulama kümesi, dördü de 180 dakika.

**Katkılar azalan sırada ve üst üste biniyor.** Kayıp uzayı tek başına
mesafenin %69'unu kapatıyor; 6B temsili kalanın dörtte birini; ImageNet
omurga son %5'i. Üçü de aynı yöne — dönmenin daha iyi öğrenilmesine —
çalıştığı için azalan getiri beklenen davranış, ve ölçüldü.

### Merdivenin ilk satırı Faz 4'ün sonucunu düzeltti

`A1_referans` 180 dakikada **360 epok** koştu — `U_uzun`'un faz geçişini
yaptığı epok 200–250 aralığının çok ötesi — ve **hâlâ çökmüş**: `|r|` 0,031,
yani Faz 4'teki 60 dakikalık `taban` koşumunun (0,035) aynısı.

Faz 4 "çöküşü bütçe kırıyor" demişti. Doğrusu daha keskin:

> **Geçiş hem yeterli bütçe hem doğru yapılandırma istiyor. Referansın kendi
> ayarları, kendilerine üç kat bütçe verilse bile çöküşten çıkmıyor.**

`U_uzun`'un çıkabilmesinin sebebi bütçe değil, 6B + ImageNet birleşimiydi;
bütçe yalnızca çıkışın *görülebilmesi* için gerekiyordu. Ve `A2_C`, başka
hiçbir şeyi değiştirmeden, sadece kayıp uzayıyla aynı çıkışı üç saatte
yapıyor.

### Neden bu tablo Faz 4'ünkinden farklı sonuç veriyor

Aynı fikirler, iki farklı bütçede ölçüldüğünde:

| Fikir | Faz 4 (60 dk, havuzun içinde) | Faz 5 (180 dk, geçişin ötesinde) |
|---|---|---|
| Parametre kaybı (C) | %−37,8 | **%−69,0** |
| 6B temsil (B) | %−6,2 | **%−24,0** |
| ImageNet (G) | %−5,7 | %−4,9 |

C ve B'nin gerçek katkısı, kısa bütçede olduğundan **iki-dört kat** büyük.
Sebebi Faz 4'ün faz geçişi bulgusu: 60 dakikalık ölçüm çöküş havuzunun
içinde yapılıyordu ve orada her satır fikrin katkısını değil havuzun
derinliğini raporluyordu.

**Bu, Faz 5'in kendi başına bir bulgusu:** yanlış bütçede yapılan bir
ablasyon, fikirlerin gerçek katkısını gizler — ve bunu iddia olarak değil,
aynı üç fikrin iki bütçedeki ölçümüyle gösteriyoruz. G'nin katkısının
değişmemesi de tutarlı: o, diğer ikisinden farklı bir şeyi (nereden
başlandığını) değiştiriyor.

### Eldeki en iyi model hâlâ `U_uzun`

Merdiven 180 dakikayla sınırlı; `U_uzun` 480 dakika koştu ve doğrulamada
GP 15,38 ile `A4_CBG`'nin 20,91'inin altında. İkisi farklı bütçede olduğu
için merdiven satırı olarak yan yana konamaz.

Açık kalan tek kutu da burada: **parametre kaybı + uzun bütçe birlikte
denenmedi.** `U_uzun` nokta tabanlı kayıpla koştu; merdiven kayıp uzayının
tek başına %69 getirdiğini gösterdi. İkisinin bileşimi bu donanımda
ulaşılabilecek en iyi sonuç olmaya aday ve tek bir sekiz saatlik koşum
mesafesinde. Faz 6'da koşuldu: `U_uzun_C`, test GP 14,40 — ayrıntı
[Faz 6](#faz-6--paketleme-ve-son-koşum).

### Çalışmayanlar — ve neden çalışmadıkları

Negatif sonuç da sonuçtur. Dokuz Faz 4 deneyinin dördü hiçbir şey getirmedi
ya da zarar verdi; hepsinin gerekçesi ölçüldü.

| Fikir | Sonuç | Neden |
|---|---|---|
| **Öğrenme oranı planı** (`L_plato`) | Dört ölçüde de tabanla **ondalığına kadar aynı** | `lr` 60 dakika boyunca `1,00e-04`'te kaldı: plato tetikleyicisi **bir kez bile** devreye girmedi, doğrulama hep azıcık iyileşip sabrı tüketmedi. Faz 3'ün "çöküşten çıkmak için lr azalt" önerisi, azaltılacak bir plato olmadığı için boşa çıktı |
| **Tutarlılık kısıtı** (`D_tutarlilik`) | Tablonun en kötüsü, GP %−33,9 | Sayı fikri değil **uygulamayı** yargılıyor: 60 dakikada sadece 13 epok (taban 108). Tahmin edilen bedel ~2 kat, ölçülen ~8 kat. Kısıtın matematiği `tests/test_temsil.py` T6'da doğru çalıştığı kanıtlı — adil sınanmadı |
| **Uzun zamansal bağlam** (`A_baglam`) | Yerelde en iyi (LP 0,3988), küresel kazanç en düşük (%2,4) | GP/LP oranı tabandan **kötü** (230× vs 227×). Kare başına tahmini iyileştirdi, sürüklenmeyi iyileştirmedi — Faz 1'in bulgusunun tekrarı: sürüklenmeyi hatanın büyüklüğü değil **yanlılığı** belirliyor |
| **Global sürüklenme düzeltmesi** (Faz 3) | İki modelde de zarar (+%1 ve +%2) | Yanlılık gerçek ve büyük ama **taramaya özgü**. Kehanet sürümü GP'nin %83'ünü siliyor, dürüst sürüm kaybettiriyor. Tek bir sabit düzeltmeyle giderilemez |
| **Veri yoğunlaştırma** (`S_ezber`, tanı) | Bağdaşım yine kıpırdamadı (\|r\| 0,066) | 24 taramada 2400 epok, kendi eğittiği denekte ölçüm. GP'deki 94→75 iyileşmesi öğrenmeden değil, tek deneğe daralınca yanlılığın küçülmesinden — **yerel doğruluğun kötüleşmesi** (0,413→0,496) bunu ele veriyor |
| **En iyi modelin üstüne 5 karelik bağlam** (`U_uzun_CA`, 8 sa) | GP 26,35 / 14,43, LP 0,196 / 0,173 (doğrulama) | Dört ölçüde de kötü, GP/LP 134× / 84×. Bu kez bütçe değil: 1122 epok, faz geçişinin ötesinde, \|r\| 0,51. Sınanmamış hipotez: kayıpta 10 kare çifti var ve parametre kaybını aralıklı çiftler domine ediyor, ölçüm ise yalnız komşu çiftleri zincirliyor |
| **5 kare, yalnız yan yana çiftler** (`U_uzun_CA1`, 8 sa) | GP 16,07 / 14,43, LP 0,1732 / 0,1726 (doğrulama) | Yukarıdaki hipotezi tek değişkenle sınıyor (`--tek-aralik 1`: 10 karışık çift yerine yan yana 4 çift). Hipotez doğrulandı — GP 26,35 → 16,07 (%−39) — ama 5 kare yine 2 kareyi geçemiyor: GP/LP 93× / 84×, yerel doğruluk aynı. **Doğrulamanın 240 taramasının tamamında yeniden ölçüldü: GP 16,31 / 16,87 (`U_uzun_C`'nin iki tohum ortalaması) — berabere** (eşleşik fark −0,56 ± 0,49 mm, taramaların %52'sinde iyi). 60 taramalık sıralama bunu kayıp saymıştı; 5 kare ne kazandırıyor ne kaybettiriyor |

Ayrıca **beklentiyi tersine çeviren** bir sonuç: `C_parametre` negatif kontrol
olarak konmuştu (yol haritası "kaybı nokta tabanlı yap" diyor, referans kaybı
zaten öyleydi, kaldırınca ne kaybedildiği ölçülecekti). Kaybedilmedi —
parametre uzayında kayıp eşit bütçedeki **en iyi** sonuç oldu. Ayrıntı:
"Beklentiyi tersine çeviren: C hattı" bölümü.

### Yayınlanmış sonuçlara göre konum

TUS-REC2024'ün resmî liderlik tablosu (yarışmanın kendi kapalı test kümesi):

| Sıra | Takım | GPE | GLE | LPE | LLE | Skor |
|---|---|---|---|---|---|---|
| 1 | ImFusion | 9,249 | 6,986 | 0,153 | 0,120 | 0,622 |
| 2 | MUSIC Lab | 9,851 | 8,116 | **0,138** | **0,116** | 0,620 |
| 3 | COCHE | 13,892 | 10,705 | 0,172 | 0,143 | 0,503 |
| 4 | AGH-MedApp | 18,605 | 15,955 | 0,179 | 0,153 | 0,383 |
| 5 | AMI-Lab | 21,800 | 19,645 | 0,177 | 0,157 | 0,301 |
| 6 | QBME | 25,708 | 21,728 | 0,216 | 0,183 | 0,188 |
| 7 | **Yarışmanın kendi referansı** | 26,110 | 23,681 | 0,214 | 0,181 | 0,163 |

Bizim sayılarımız (kendi ayırdığımız 10 denek, 240 tarama):

| | GP | GL | LP | LL |
|---|---|---|---|---|
| Faz 2 — referansın yeniden üretimi (178 epok) | 86,90 | 83,65 | 0,3856 | 0,3850 |
| Faz 4 — `U_uzun` (912 epok) | 17,25 | 16,35 | 0,1565 | 0,1372 |
| **Faz 6 — `U_uzun_C` (866 epok)** | **14,40** | **12,52** | **0,1510** | **0,1313** |

**Bu bir sıralama değil — test kümeleri farklı.** Liderlik tablosu yarışmanın
kapalı test kümesinde (85 deneklik kohortun ayrılmış kısmı), bizimki kendi
denek bazlı bölmemizde (açık eğitim verisinden ayrılan 10 denek) ölçüldü.
Aynı protokol ve aynı cihaz, ama aynı kümede değil. Sayılar **gösterge**,
sıralama değil.

Bu kaydıyla okunduğunda:

- **Yerel doğruluk üst sıralarla yarışıyor.** LP 0,1510 ve LL 0,1313;
  3.–7. sıradaki bütün takımlardan iyi. LP'de 1. sıranın (0,153) önünde,
  yalnızca 2. sıranın (0,138) gerisinde; LL'de ilk ikinin (0,120 ve 0,116)
  gerisinde.
- **Küresel doğruluk 3.–4. sıra arasında.** GP 14,40 ve GL 12,52, COCHE
  (13,9 / 10,7) ile AGH-MedApp (18,6 / 16,0) arasında, COCHE'ye daha yakın.
- **Yarışmanın kendi referansı geçildi** — dört ölçünün dördünde de
  (26,11 → 14,40; 23,68 → 12,52; 0,214 → 0,151; 0,181 → 0,131). Bu referans
  20.000 epokluk bütçeyle eğitilmişti; bizimki 866 epok, tek bir 4 GB
  dizüstü ekran kartında sekiz saat.

Bu son satırın pratik karşılığı: kazancı getiren şey yeni bir mimari değil,
**çöküş havuzundan çıkaran yapılandırma** (6B temsil, ImageNet omurga,
parametre uzayında kayıp) ve çıkışın görülmesine yetecek bütçe. Faz 5
bütçenin tek başına yetmediğini ölçtü: referansın kendi ayarları üç kat
bütçede de çökmüş kalıyor. Çıkıldıktan sonra ise referansın kendi mimarisi,
referansın kendi skorunun belirgin biçimde üstüne çıkıyor.

Kaynak: [TUS-REC2024 liderlik tablosu](https://github.com/UCL/tus-rec-challenge/blob/main/leaderboard.md),
[yarışma raporu (arXiv:2506.21765)](https://arxiv.org/abs/2506.21765).

## Faz 6 — paketleme ve son koşum

Yol haritası bu fazdan video, tek komutla yeniden üretim, README'nin başına
ablasyon tablosu ve bilinen sınırlar istiyor; hepsi yukarıda. Faz, Faz 5'in
açık bıraktığı tek kutuyla başladı: parametre kaybı ile uzun bütçe hiç
birlikte denenmemişti.

```bash
python scripts/faz4_deney.py U_uzun_C --sure 480 --kok results/faz6
python scripts/faz4_deney.py U_uzun_C --kok results/faz6 --test-olcumu
```

`U_uzun_C`, `U_uzun`'dan **tek değişkenle** ayrılıyor: kayıp nokta yerine
parametre uzayında. Bütçe, tohum, temsil ve omurga aynı.

### Sonuç

| | Küme | GP | GL | LP | LL | GP/LP | \|r\| |
|---|---|---|---|---|---|---|---|
| `U_uzun` | doğrulama (60 tarama) | 15,377 | 13,587 | 0,1724 | 0,1578 | 89× | 0,545 |
| `U_uzun_C` | doğrulama (60 tarama) | **14,425** | **12,611** | 0,1726 | **0,1551** | **84×** | **0,557** |
| `U_uzun` | test (240 tarama) | 17,252 | 16,352 | 0,1565 | 0,1372 | 110× | |
| `U_uzun_C` | test (240 tarama) | **14,404** | **12,516** | **0,1510** | **0,1313** | **95×** | |

Test kümesinde GP %−16,5, GL %−23,5; taramaların %59'unda `U_uzun_C` daha
iyi. Referansa (Faz 2) göre toplam: GP %−83,4, GL %−85,0, LP %−60,8,
LL %−65,9. Test ölçümü yalnız doğrulamada kazanan için, bir kez yapıldı.

### Kayıp uzayının katkısı uzun bütçede küçülüyor

| Bütçe | Nokta kaybı → parametre kaybı, GP (doğrulama) |
|---|---|
| 60 dk (Faz 4, havuzun içinde) | %−37,8 |
| 180 dk (Faz 5, merdivenin ilk basamağı) | %−69,0 |
| 480 dk (Faz 6, 6B + ImageNet'in üstüne) | %−6,2 |

Kare başına doğrulama mesafesi bunun nasıl olduğunu gösteriyor: epok 230'da
`U_uzun_C` 0,229 mm, `U_uzun` 0,305 mm; epok 460'ta 0,203 ve 0,208. Parametre
kaybı çöküşten **daha erken** çıkarıyor, ama iki koşum da çıktıktan sonra
aradaki fark kapanıyor. Merdivendeki %69, büyük ölçüde "çıktı mı çıkmadı mı"
farkıydı; burada ikisi de çıkmış durumda. Kalan kazanç yerelde değil
(LP aynı) küreselde — GP/LP 89×'ten 84×'e: parametre kaybı, sürüklenmeyi
büyüten yanlılığı biraz daha az üretiyor.

Doğrulamadaki %6,2 ile testteki %16,5 arasındaki fark da kayda değer:
doğrulama denek başına 6 tarama (60), test 240 tarama. Tek tohumla bu iki
sayının hangisinin gerçek etkiye daha yakın olduğu söylenemez; ikisi de aynı
yönde.

### Merdivenin bağdaşım eğrileri — çıkış yapılandırmaya bağlı

Faz 5'in tablosu her satırın **son** noktasını veriyordu. Ara anlık
görüntülerden çizilen eğriler (doğrulama, denek başına 2 tarama) çıkışın
**ne zaman** olduğunu gösteriyor:

```bash
python scripts/faz4_bagdasim_egrisi.py --kosum results/faz5/A2_C --tarama 2
```

![Bağdaşım eğrisi, ablasyon merdiveni](docs/figures/coherence_ladder.png)

| Epok | 0 | 50 | 100 | 200 | 300 | son |
|---|---|---|---|---|---|---|
| `A1_referans` | 0,092 | 0,036 | 0,041 | 0,042 | 0,021 | 0,036 (350) |
| `A2_C` | 0,030 | 0,147 | 0,256 | 0,333 | 0,402 | 0,402 (300) |
| `A3_CB` | 0,045 | 0,126 | 0,222 | 0,313 | 0,429 | 0,429 (300) |
| `A4_CBG` | 0,056 | **0,391** | 0,434 | 0,481 | 0,508 | **0,538** (400) |

Üç şey okunuyor:

- **`A1` düz.** 350 epok boyunca 0,02–0,06 bandında; faz geçişinin olduğu
  epok 200–250'de hiçbir kıpırtı yok. Referans yapılandırması için daha uzun
  eğitim çıkış getirmez — Faz 5'in "bütçe değil yapılandırma" sonucunun
  görsel kanıtı.
- **`A2` ve `A3` kademeli yükseliyor.** Parametre kaybı çıkışı başlatıyor;
  6B temsil eğriyi pek değiştirmiyor ama son GP'yi %24 iyileştiriyor.
- **ImageNet (`A4`) çıkışı öne çekiyor.** Epok 50'de 0,39'a sıçrıyor —
  `A2`'nin 200–250 epokta vardığı düzeye. GP tablosunda G'nin katkısı en küçüğüydü
  (%4,9); eğri, katkısının **son noktada değil hızda** olduğunu gösteriyor.
  Kısa bütçede bu fark belirleyici olur.

## Donanım

RTX 3050 Ti Laptop (4 GiB) + 64 GiB RAM. Referans README'nin andığı 30 GB GPU
**yalnızca DDF üretimini GPU'da yapmak isteyenler için**; varsayılan yol
işlemcide üretmektir ve orada belirleyici olan RAM'dir. Eğitimde darboğaz
4 GiB VRAM: referans ayar `MINIBATCH_SIZE=16` @ 480×640, düşürülmesi gerekecek.
