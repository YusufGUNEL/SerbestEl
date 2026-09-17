# SerbestEl — Sensörsüz 3B Ultrason Rekonstrüksiyonu

Elde gezdirilen ultrason probundan, **konum sensörü olmadan** 3B hacim çıkarma.
Model, ardışık kareler arasındaki rijit dönüşümü yalnızca görüntüden tahmin eder;
dönüşümler zincirleme çarpılarak her karenin ilk kareye göre konumu bulunur.

Kıyas noktası: **TUS-REC2025** (MICCAI 2025 / ASMUS, UCL). Yarışma Ekim 2025'te
kapandı; bu proje yayınlanmış sonuçlara karşı çalışır, sıralamaya girmez.

## Belgeler

Tarayıcıda aç:

| Dosya | Ne var |
|---|---|
| `docs/index.html` | Proje özeti |
| `docs/yol-haritasi.html` | 7 fazlık plan, faz başına adımlar ve çıktılar |
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
beş ayrı geçersiz durumu kilitliyor.

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

## Donanım

RTX 3050 Ti Laptop (4 GiB) + 64 GiB RAM. Referans README'nin andığı 30 GB GPU
**yalnızca DDF üretimini GPU'da yapmak isteyenler için**; varsayılan yol
işlemcide üretmektir ve orada belirleyici olan RAM'dir. Eğitimde darboğaz
4 GiB VRAM: referans ayar `MINIBATCH_SIZE=16` @ 480×640, düşürülmesi gerekecek.
