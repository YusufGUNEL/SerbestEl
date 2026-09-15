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
| TUS-REC2025 eğitim | **kısıtlı** — Zenodo'dan talep gerekir | ~22 GB | 50 denek, 100 tarama, ~164k kare |
| TUS-REC2025 doğrulama | açık | 1,23 GB | 3 denek, 6 tarama, ~9k kare |
| TUS-REC2024 eğitim (1–3) | açık | ~84 GB | Aynı kohort, döndürmesiz protokol |
| TUS-REC2024 doğrulama | açık | 4,84 GB | |

Veriler CC-BY-NC-SA-4.0. Eğitim kümesi kısıtlı olduğu için erişim talebi
projenin kritik yolundadır — reddedilirse TUS-REC2024'ün açık ~84 GB'ı
ana eğitim kaynağı olur.

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

## Faz 2'ye başlarken — sırayla bunlar

Faz 0 ve Faz 1 bitti. Eğitim verisi gelir gelmez aşağıdaki sıra izlenir.

```bash
# 0) ortam ve doğrulamalar hâlâ sağlam mı (hepsi geçmeli)
python scripts/ortam_dogrula.py              #  9 kontrol
python tests/test_geometri.py                # 11 kontrol
python tests/test_hacim.py                   #  3 kontrol
python tests/test_olcum.py                   #  4 kontrol

# 1) veriyi indir  (2025 eğitim seti kısıtlı, elle talep gerekir)
python scripts/veri_indir.py --kume tusrec2024 --hedef D:/SerbestEl-veri/tusrec2024
#    kesilirse aynı komutu tekrar çalıştır: kaldığı yerden devam eder

# 2) denek bazlı bölmeyi ÜRET ve SABİTLE — bir kere
python src/bolme.py --veri <veri>/frames_transfs

# 3) referansı eğit (değiştirmeden; 4 GiB karta uyarlanmış çalıştırma)
python src/egit.py --veri <veri> --epok 20000
python src/egit.py --devam                   # kesilirse kaldığı yerden

# 4) dört ölçüyü hesapla — projenin omurgası olan tablo
python src/olcum.py --veri <veri> --agirlik results/faz2_referans/en_iyi_model.pt \
                    --kume test --cikti results/faz2_referans/olculer.json
```

**Test kümesine 3. adımda dokunulmaz.** `src/egit.py` bölmeyi yükler, test
deneklerini ayırır ve eğitimde kullanmaz.

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
