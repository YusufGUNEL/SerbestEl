"""Faz 4 deney kosucusu — bir deneyi ucdan uca kosar ve kaydini birakir.

DISIPLIN KURALI
  Yol haritasinin Faz 4 maddesi tek bir kurala dayaniyor: HER DENEMEDE TEK
  DEGISKEN. Iki sey birden degisirse hangisinin ise yaradigi bilinemez ve
  Faz 5'in ablasyon tablosu yazilamaz. Bu betik o kurali kodun icine koyuyor:
  deneyler asagidaki sozlukte adlariyla duruyor, her biri TABANDAN yalnizca
  kendi bayragiyla ayriliyor, ve ayni betik hepsini ayni butceyle kosuyor.

BUTCE NEDEN SURE
  Referansin 20000 epokluk butcesi bu donanimda gunler surer. Epok saymak
  ayrica yaniltici: baglam uzunlugu ya da tutarlilik kisiti degisince bir
  epogun maliyeti degisir. Sabit SURE hem tekrar edilebilir hem durust —
  "ayni donanimda ayni sure verildiginde hangisi daha iyi" sorusu, bir
  fikrin pratikte ise yarayip yaramadigini soran dogru sorudur.

UC ADIM, HER DENEYDE AYNI
  1. egitim       — sabit sure butcesi, en iyi dogrulama modeli saklanir
  2. dort olcu    — DOGRULAMA kumesinde (test kumesine dokunulmaz)
  3. bagdasim     — model goruntuye mi bakiyor yoksa ortalamayi mi soyluyor

  Ucuncu adim Faz 3'ten geliyor ve atlanamaz: kayip egrisi duzgun inerken
  model tamamen cokmus olabilir. Cokmus modelde hicbir fikrin katkisi
  anlamli olculemez, o yuzden her kosumun ilk kontrolu bagdasimdir.

TEST KUMESI
  Faz 4 boyunca test kumesine DOKUNULMUYOR. Deneyler dogrulama kumesinde
  siralaniyor; yalnizca kazanan yontem, bir kere, test kumesinde olculup
  Faz 2'nin dort sayisiyla karsilastiriliyor. Aksi halde 7 deneyin en
  iyisini test skoruna gore secmek test kumesini egitim kumesine cevirirdi.

KULLANIM
  python scripts/faz4_deney.py --liste
  python scripts/faz4_deney.py taban G_imagenet --veri D:/SerbestEl-veri/...
  python scripts/faz4_deney.py B_6b --sure 60 --atla egitim   # yalniz olcum
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Deney kaydi. "egitim" bayraklari egit.py'ye, "mimari" bayraklari HEM
# egitime HEM olcume gider (ag mimarisini degistirdikleri icin olcum de ayni
# ayarla kosmak zorunda; yanlis ayarla olcum sessizce sacma sayi uretir).
DENEYLER: dict[str, dict] = {
    "taban": {
        "hat": "-",
        "baslik": "Faz 2 kosumu, Faz 4 kosullarinda",
        "neden": "Butun karsilastirmalarin cipasi. Faz 2'nin kendi kosumu 90 "
                 "dakikaydi ve son katman yanliligi rastgeleydi; Faz 4'te "
                 "butun temsiller birimden basliyor ve butce ortak. Taban "
                 "ayni kosullarda yeniden uretiliyor ki fark gercekten "
                 "degiskenin farki olsun.",
        "egitim": [],
        "mimari": [],
    },
    "G_imagenet": {
        "hat": "G",
        "baslik": "Omurga ImageNet agirligiyla basliyor",
        "neden": "Faz 3'un ana teshisi regresyon cokusuydu: 6,5 milyon "
                 "parametre rastgeleden basliyor ve 128 bin pencereyle "
                 "goruntuye bakmayi hic ogrenemiyor. ImageNet sizinti "
                 "tasimaz (ultrason yok, denek yok, hareket etiketi yok) "
                 "ama kenar ve doku suzgeclerini hazir verir.",
        "egitim": ["--omurga-onegitimli"],
        "mimari": [],
    },
    "B_6b": {
        "hat": "B",
        "baslik": "6B surekli donme temsili",
        "neden": "Yol haritasi B hatti. Faz 3 olctu: cokmeyen bir modelde "
                 "bile donme otelemeden belirgin kotu (donme olcegi "
                 "0,09-0,29, oteleme 0,30-0,73). Euler acilari SO(3)'u "
                 "surekli temsil edemez; 6B temsil edebilir.",
        "egitim": ["--donme-temsili", "6b"],
        "mimari": ["--donme-temsili", "6b"],
    },
    "A_baglam": {
        "hat": "A",
        "baslik": "Daha uzun zamansal baglam (2 -> 5 kare)",
        "neden": "Yol haritasi A hatti. Iki kare arasi hareket 0,4 mm; tek "
                 "karelik gurultu bu sinyalin yaninda buyuk. Bes kare on "
                 "cift uretir, yani AYNI ileri gecisten on kat denetim.",
        "egitim": ["--num-samples", "5", "--sample-range", "5", "--num-pred", "4"],
        "mimari": ["--num-samples", "5", "--num-pred", "4"],
    },
    "D_tutarlilik": {
        "hat": "D",
        "baslik": "Ileri/geri tutarlilik kisiti",
        "neden": "Yol haritasi D hatti. Kare sirasi ters verilince ag "
                 "T(0->1) tahmin eder; T(1->0) ile bilesimi birim olmali. "
                 "Ek etiket istemez ve dogrudan hata BIRIKMESINI hedefler. "
                 "Bedeli acik: ikinci ileri gecis adim suresini iki katina "
                 "cikarir, yani ayni surede yari pencere gorulur.",
        "egitim": ["--tutarlilik", "1.0"],
        "mimari": [],
    },
    "C_parametre": {
        "hat": "C",
        "baslik": "Kayip nokta yerine parametre uzayinda",
        "neden": "Yol haritasi C hatti TABANDA ZATEN VAR: referans kaybi "
                 "mm cinsinden nokta mesafesi. O yuzden buradaki deney "
                 "hattin kendisi degil NEGATIF KONTROLU — kayip ham "
                 "parametreye alinirsa ne kaybedildigi olculuyor. Fikrin "
                 "degeri ancak kaldirilinca sayiya doner.",
        "egitim": ["--kayip-uzayi", "parametre"],
        "mimari": [],
    },
    "S_ezber": {
        "hat": "tani",
        "baslik": "Ezber tanisi — tek denek, 24 tarama",
        "neden": "Yontem degil TANI. ImageNet cokusu kirmayinca soru ikiye "
                 "ayrildi: ag hareketi ogrenemiyor mu, yoksa bu butcede "
                 "yeterince pencere goremiyor mu? Egitim kumesi 720 taramadan "
                 "24'e indiriliyor, butce ayni kaliyor — ayni sayida pencere, "
                 "otuz kat daha yogun. Ag BURADA da ortalamayi soyluyorsa "
                 "sorun veri olceginde degil, daha temelde. Ogreniyorsa "
                 "darbogaz olcek demektir ve cevap daha uzun egitimdir.",
        "egitim": [],
        "mimari": [],
        "bolme": "configs/bolme_ezber.json",
    },
    "K_birlesik": {
        "hat": "B+G",
        "baslik": "6B temsil + ImageNet omurga birlikte",
        "neden": "TEK DEGISKEN KURALININ BILINCLI ISTISNASI, ve yerini "
                 "hak ediyor: B ile G tek tek olculdu, ikisi de tabani "
                 "yaklasik %6 gecti, ve degistirdikleri seyler birbirinden "
                 "bagimsiz (biri agin son katmaninin ne urettigi, digeri "
                 "omurganin nereden basladigi). Soru acik: kazanclar "
                 "topluyor mu? Toplamiyorsa ikisi de ayni seyi — sabitin "
                 "biraz iyilesmesini — farkli yoldan yapiyor demektir. "
                 "Bu bir katki olcumu; Faz 5'in ablasyon tablosunun ilk "
                 "satiri buradan cikiyor.",
        # "mimari" bayraklari zaten hem egitime hem olcume gidiyor, o yuzden
        # 6b burada bir kere yaziliyor; "egitim" yalnizca omurgayi tasiyor.
        "egitim": ["--omurga-onegitimli"],
        "mimari": ["--donme-temsili", "6b"],
    },
    "U_uzun": {
        "hat": "olcek",
        "baslik": "Kazanan yapilandirma, sekiz kat butce",
        "neden": "Faz 4'un butun deneyleri ayni yere cikti: dort bagimsiz "
                 "eksen (baslangic agirligi, donme temsili, baglam uzunlugu, "
                 "veri yogunlugu) dort olcuyu %2-6 oynatiyor, hicbiri "
                 "bagdasimi kipirdatmiyor. Geriye tek eksen kaldi — gorulen "
                 "pencere sayisi. Referans 14,4 milyon pencere goruyor, 60 "
                 "dakikalik butce 78 bin: binde bes. Ve o eksenin ust ucunda "
                 "olculmus bir nokta var: Faz 3'un kontrol kosumunda ayni "
                 "mimari, ayni veri, r = 0,84. Yani mimari bu isi "
                 "ogrenebiliyor; ogrenemeyen sey 78 bin pencereyle egitilmis "
                 "hali.\n"
                 "Bu kosum tek bir sayi degil BIR EGRI uretiyor: ara anlik "
                 "goruntulerden bagdasim, egitim olcegine karsi cizdiriliyor. "
                 "Duz cizgi 'cokus butceye bagli degil' der ve bu donanimda "
                 "durulacak yeri gosterir; yukselen egri 'bagli ve su hizla' "
                 "der ve Faz 5'e tasinacak sayiyi verir.",
        "egitim": ["--omurga-onegitimli", "--ara-kayit", "50"],
        "mimari": ["--donme-temsili", "6b"],
    },
    # ---------------------------------------------------------------------
    # FAZ 5 — ABLASYON MERDIVENI
    #
    # Faz 4 tek degiskenli deneyler yapti: her biri TABANDAN ayriliyordu.
    # Yol haritasinin Faz 5 maddesi baska bir sey istiyor: KUMULATIF bir
    # merdiven (referans -> +C -> +C+B -> +C+B+G), cunku sorulan soru
    # "bu fikir tek basina ne getirir" degil, "oncekilerin USTUNE ne ekler".
    # Ikisi ayni sey degil: Faz 4 olctu ki B ve G tek tek %6 getiriyor ama
    # birlikte %12 degil %7,6 getiriyor.
    #
    # BUTCE NEDEN 180 DAKIKA
    # Faz 4'un ana bulgusu, cokusun epok 200-250 arasinda bir FAZ GECISIYLE
    # bittigi. 60 dakikalik butce epok ~115'te bitiyordu, yani gecisin
    # yarisinda — o tablonun her satiri havuz derinligini olcuyordu, fikrin
    # katkisini degil. Havuzun icinde yapilan bir ablasyon HICBIR SEY olcmez.
    # 180 dakika en yavas yapilandirmada bile ~300 epok veriyor; butun
    # satirlar gecisin otesinde.
    #
    # SIRA NEDEN C -> B -> G
    # Faz 4'un esit butceli sonuclarina gore buyukten kucuge: C %37,8,
    # B %6,2, G %5,7. Boylece her basamak bir oncekinin uzerine ne
    # ekledigini gosterir ve azalan getiri gorulur.
    # ---------------------------------------------------------------------
    "A1_referans": {
        "hat": "referans",
        "baslik": "Merdivenin cipasi — referansin kendi ayarlari",
        "neden": "Nokta tabanli kayip, Euler temsili, rastgele omurga. "
                 "Faz 2'nin yapilandirmasi, Faz 5 butcesiyle. Ayrica kendi "
                 "basina bir soruyu cevapliyor: taban, 180 dakikada faz "
                 "gecisini tek basina yapabiliyor mu?",
        "egitim": ["--ara-kayit", "50"],
        "mimari": [],
    },
    "A2_C": {
        "hat": "+C",
        "baslik": "+ parametre uzayinda kayip",
        "neden": "Faz 4'un en buyuk tek degiskenli kazanci (%37,8). Merdivenin "
                 "ilk basamagi; mm cinsinden nokta kaybi otelemeye agirlik "
                 "verirken parametre kaybi alti bileseni daha dengeli tartiyor.",
        "egitim": ["--kayip-uzayi", "parametre", "--ara-kayit", "50"],
        "mimari": [],
    },
    "A3_CB": {
        "hat": "+C+B",
        "baslik": "+ 6B surekli donme temsili",
        "neden": "Ikinci basamak. C donmeye dusen payi artirdiysa, B'nin "
                 "katkisi kucumelidir — ikisi de ayni sorunu, donmenin kotu "
                 "ogrenilmesini, hedefliyor. Azalan getiri beklentisi burada "
                 "sinaniyor.",
        "egitim": ["--kayip-uzayi", "parametre", "--ara-kayit", "50"],
        "mimari": ["--donme-temsili", "6b"],
    },
    "A4_CBG": {
        "hat": "+C+B+G",
        "baslik": "+ ImageNet omurga",
        "neden": "Ucuncu basamak ve merdivenin tepesi. G digerlerinden farkli "
                 "bir seyi degistiriyor (nereden baslandigini, ne uretildigini "
                 "degil), o yuzden katkisinin toplanmasi en olasi olan bu.",
        "egitim": ["--kayip-uzayi", "parametre", "--omurga-onegitimli",
                   "--ara-kayit", "50"],
        "mimari": ["--donme-temsili", "6b"],
    },
    # FAZ 6 — Faz 5'in acik biraktigi tek kutu. Merdiven, parametre
    # kaybinin tek basina %69 getirdigini gosterdi; U_uzun ise nokta
    # kaybiyla kosmustu. Bu kosum ikisini birlestiriyor. Tek degisken
    # kuralina uyuyor: U_uzun'dan yalnizca kayip uzayi ile ayriliyor.
    "U_uzun_C": {
        "hat": "olcek+C",
        "baslik": "U_uzun + parametre uzayinda kayip",
        "neden": "Merdivenin tepesi (A4_CBG) U_uzun'un butcesiyle. U_uzun'dan "
                 "tek farki kayip uzayi; merdiven o farkin 180 dakikada "
                 "%69 oldugunu olctu. Soru: faz gecisinin otesinde, uzun "
                 "butcede de kazanc duruyor mu, yoksa U_uzun zaten ayni "
                 "yere mi vardi?",
        "egitim": ["--kayip-uzayi", "parametre", "--omurga-onegitimli",
                   "--ara-kayit", "50"],
        "mimari": ["--donme-temsili", "6b"],
    },
    # FAZ 7 — uzun kare dizisi. A_baglam (5 kare) Faz 4'te yalniz 60 dakika
    # kostu, yani cokus havuzunun icinde olculdu: fikir adil sinanmadi.
    # Burada en iyi yapilandirmanin ustune, faz gecisinin otesinde deneniyor.
    # U_uzun_C'den tek farki baglam uzunlugu.
    "U_uzun_CA": {
        "hat": "olcek+C+A",
        "baslik": "U_uzun_C + 5 karelik baglam",
        "neden": "TUS-REC'in ilk iki sirasi (GP 9-10 mm) uzun kare dizisine "
                 "bakan modellerle o farki kapatiyor. Bizim tek 5 kare denememiz "
                 "(A_baglam) 60 dakikada, cokusun icinde olculdu: yerelde en "
                 "iyi, kuresele tasiyamadi — ama o butcede hicbir fikir "
                 "kuresele tasiyamiyordu. Epok maliyeti tabanla ayni (113 vs "
                 "108 epok / 60 dk), yani ayni 480 dakika adil bir kiyas.",
        "egitim": ["--kayip-uzayi", "parametre", "--omurga-onegitimli",
                   "--ara-kayit", "50", "--sample-range", "5"],
        "mimari": ["--donme-temsili", "6b", "--num-samples", "5", "--num-pred", "4"],
    },
    "L_plato": {
        "hat": "L",
        "baslik": "Dogrulama duzelmeyince ogrenme hizi yariya iner",
        "neden": "Faz 3'un birinci onceligi 'cokusten cikmak' ve ilk akla "
                 "gelen carey bu. Ayni zamanda bir kontrol: cokus "
                 "eniyilemenin mi yoksa baslangicin/temsilin mi sorunu? "
                 "Yalniz basina ise yaramazsa bu sorunun cevabi 'degil'.",
        "egitim": ["--lr-plan", "plato"],
        "mimari": [],
    },
}


def utf8_kur() -> None:
    """Kendi ciktimizi UTF-8 yapar, alt sureclere de oyle yazdirir.

    OLCULEN COKME: Windows konsolu cp1254 kullaniyor. Alt betigin ciktisinda
    o kod sayfasinda karsiligi olmayan tek bir karakter olunca
    `sys.stdout.write` UnicodeEncodeError atiyor — ve bunu yakalayan except
    blogu hatayi YAZDIRMAYA calisinca ayni hatayi tekrar aliyor, boylece
    butun dizi oluyor. Altmis dakikalik egitimini bitirmis, olcumunu ve
    bagdasimini yazmis bir kosum, yalnizca bir karakter yuzunden kalan alti
    deneyi hic baslatmadan durdu.

    Iki yerden birden kapatiliyor: kendi akisimiz UTF-8'e ve "hata olursa
    yerine koy"a ayarlaniyor, alt surecler de PYTHONIOENCODING ile UTF-8
    yazmaya zorlaniyor. Gunluk dosyasi zaten UTF-8 aciliyordu, bu yuzden
    diske yazilan kayit hicbir zaman etkilenmemisti.
    """
    for akis in (sys.stdout, sys.stderr):
        try:
            akis.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def kos(etiket: str, komut: list[str], gunluk: Path) -> None:
    """Alt betigi kosar, ciktisini hem ekrana hem gunluge yazar.

    `-u` SART: tamponlanmis ciktida saatler suren bir egitim dakikalarca
    sessiz kalir ve kosumun donup donmedigi anlasilmaz (Faz 2'de olculdu).
    """
    print(f"\n{'-' * 70}\n>>> {etiket}\n    {' '.join(komut)}\n{'-' * 70}",
          flush=True)
    ortam = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(gunluk, "a", encoding="utf-8") as g:
        g.write(f"\n=== {etiket}\n{' '.join(komut)}\n")
        g.flush()
        p = subprocess.Popen(komut, cwd=KOK, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1,
                             env=ortam)
        for satir in p.stdout:
            sys.stdout.write(satir)
            sys.stdout.flush()
            g.write(satir)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"{etiket} basarisiz (cikis {p.returncode})")


def deneyi_kos(ad: str, a) -> dict:
    d = DENEYLER[ad]
    # Deneyin kendi bolmesi varsa o kullanilir. Yalnizca tani deneyleri icin;
    # yontem deneylerinin hepsi AYNI bolmeyi paylasmak zorunda, yoksa
    # karsilastirma anlamini yitirir.
    bolme = KOK / d["bolme"] if "bolme" in d else a.bolme
    dizin = a.kok / ad
    dizin.mkdir(parents=True, exist_ok=True)
    gunluk = dizin / "gunluk.txt"
    bas = time.time()

    if "egitim" not in a.atla:
        kos(f"{ad} — egitim ({a.sure:.0f} dk)", [
            PYTHON, "-u", "src/egit.py",
            "--veri", str(a.veri), "--bolme", str(bolme),
            "--kayit", str(dizin),
            "--epok", "100000", "--sure-siniri", str(a.sure),
            "--dogrulama-sikligi", "10", "--kayit-sikligi", "25",
            "--isci", str(a.isci),
            "--on-egitimli", "yok",          # sizintili agirlik Faz 4'te de yasak
            *d["egitim"], *d["mimari"],
        ] + (["--devam"] if a.devam else []), gunluk)

    agirlik = dizin / "en_iyi_model.pt"
    if not agirlik.exists():
        raise SystemExit(f"{ad}: egitilmis agirlik yok ({agirlik})")

    if "olcum" not in a.atla:
        kos(f"{ad} — dort olcu (dogrulama, denek basina {a.tarama} tarama)", [
            PYTHON, "-u", "src/olcum.py",
            "--veri", str(a.veri), "--bolme", str(bolme),
            "--agirlik", str(agirlik), "--kume", "dogrulama",
            "--tarama-basina", str(a.tarama),
            "--cikti", str(dizin / "dogrulama_olculer.json"),
            *d["mimari"],
        ], gunluk)

    if "bagdasim" not in a.atla:
        kos(f"{ad} — bagdasim kontrolu", [
            PYTHON, "-u", "src/analiz.py",
            "--veri", str(a.veri), "--bolme", str(bolme),
            "--agirlik", str(agirlik), "--kume", "dogrulama",
            "--cikti", str(dizin / "bagdasim"),
            "--yanlilik-kumesi", "yok",      # burada yalniz r'ye bakiliyor
            "--sinir", str(a.bagdasim_tarama), "--gorsel", "0",
            *d["mimari"],
        ], gunluk)

    ozet = {"ad": ad, "hat": d["hat"], "baslik": d["baslik"],
            "neden": d["neden"], "bayraklar": d["egitim"] + d["mimari"],
            "sure_dk": a.sure, "gecen_dk": round((time.time() - bas) / 60, 1)}
    (dizin / "deney.json").write_text(json.dumps(ozet, indent=2,
                                                 ensure_ascii=False),
                                      encoding="utf-8")
    return ozet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("deneyler", nargs="*", help="kosulacak deney adlari")
    ap.add_argument("--liste", action="store_true", help="deneyleri yazdir, cik")
    ap.add_argument("--veri", type=Path, default=Path("D:/SerbestEl-veri/tusrec2024/acilmis"))
    ap.add_argument("--bolme", type=Path, default=KOK / "configs" / "bolme.json")
    ap.add_argument("--kok", type=Path, default=KOK / "results" / "faz4",
                    metavar="DIZIN",
                    help="sonuclarin yazilacagi kok. Faz 5 ablasyon merdiveni "
                         "results/faz5 altina yaziliyor ki Faz 4'un tek "
                         "degiskenli deneyleriyle karismasin — iki tablonun "
                         "butceleri farkli, yan yana okunurlarsa yaniltir")
    ap.add_argument("--sure", type=float, default=60.0, metavar="DAKIKA",
                    help="deney basina egitim butcesi — HEPSINDE AYNI olmali")
    ap.add_argument("--isci", type=int, default=6)
    ap.add_argument("--tarama", type=int, default=6, metavar="N",
                    help="dogrulama olcumunde denek basina tarama sayisi")
    ap.add_argument("--bagdasim-tarama", type=int, default=12)
    ap.add_argument("--atla", nargs="*", default=[],
                    choices=["egitim", "olcum", "bagdasim"])
    ap.add_argument("--devam", action="store_true",
                    help="egitimi son checkpoint'ten surdur")
    ap.add_argument("--test-olcumu", action="store_true",
                    help="SON ADIM: verilen deneyi TEST kumesinde, butun 240 "
                         "taramada olcer ve Faz 2'nin dort sayisiyla "
                         "karsilastirilabilir hale getirir. Yalnizca dogrulama "
                         "kumesinde secilmis KAZANAN icin, bir kere kosulur.")
    a = ap.parse_args()
    utf8_kur()

    if a.liste or not a.deneyler:
        print(f"{'ad':<16} {'hat':<4} baslik")
        print("-" * 70)
        for ad, d in DENEYLER.items():
            print(f"{ad:<16} {d['hat']:<4} {d['baslik']}")
        return 0

    bilinmeyen = [x for x in a.deneyler if x not in DENEYLER]
    if bilinmeyen:
        raise SystemExit(f"bilinmeyen deney: {bilinmeyen}")

    if a.test_olcumu:
        for ad in a.deneyler:
            d = DENEYLER[ad]
            bolme = KOK / d["bolme"] if "bolme" in d else a.bolme
            dizin = a.kok / ad
            kos(f"{ad} — TEST kumesi, 240 tarama", [
                PYTHON, "-u", "src/olcum.py",
                "--veri", str(a.veri), "--bolme", str(bolme),
                "--agirlik", str(dizin / "en_iyi_model.pt"), "--kume", "test",
                "--cikti", str(dizin / "test_olculer.json"),
                *d["mimari"],
            ], dizin / "gunluk.txt")
        return 0

    # Bir deneyin cokmesi kalanlari dusurmuyor: gece boyu kosan bir dizide
    # ucuncu deneydeki bir hata kalan dordunu de iptal ederdi. Hata
    # kaydediliyor, sira devam ediyor, sonunda topluca raporlaniyor.
    hatali = []
    for ad in a.deneyler:
        try:
            ozet = deneyi_kos(ad, a)
            print(f"\n### {ad} bitti — {ozet['gecen_dk']} dk", flush=True)
        # SystemExit de yakalaniyor: `kos` basarisiz alt sureci onunla
        # bildiriyor ve SystemExit BaseException'dan turedigi icin duz bir
        # `except Exception` onu KACIRIR. Olculdu — bir deneyin coktugu yerde
        # butun sira, hic baslamamis deneyler dahil, duruyordu.
        except (Exception, SystemExit) as e:         # noqa: BLE001
            hatali.append((ad, f"{type(e).__name__}: {e}"))
            print(f"\n### {ad} BASARISIZ — {e}", flush=True)
    if hatali:
        print("\nbasarisiz deneyler:")
        for ad, hata in hatali:
            print(f"  {ad}: {hata}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
