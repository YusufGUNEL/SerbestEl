"""Faz 0 kabul testi.

Ortamin gercekten calistigini kanitlar. Tahmin yok, her sey olculur.
Kullanim:  python scripts/ortam_dogrula.py
"""

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
REFERANS = KOK / "reference" / "TUS-REC2025-Challenge_baseline"
BEKLENEN_COMMIT = "88413de5d0afffd35a08cd0196388d30faf32d31"

sonuclar = []


def kontrol(ad, fn):
    """fn -> (bool gecti, str detay)"""
    try:
        gecti, detay = fn()
    except Exception as exc:  # noqa: BLE001 - her hatayi rapora yaz
        gecti, detay = False, f"{type(exc).__name__}: {exc}"
    sonuclar.append((ad, gecti, detay))
    print(f"[{'OK ' if gecti else 'HATA'}] {ad}: {detay}")
    return gecti


# --- 1. yorumlayici -------------------------------------------------------
def _python():
    v = sys.version_info
    return (v[:2] == (3, 10), f"{platform.python_version()} @ {sys.prefix}")


# --- 2. paket surumleri ---------------------------------------------------
BEKLENEN = {
    "torch": "2.1.0+cu121",
    "torchvision": "0.16.0",
    "h5py": "3.8.0",
    "numpy": "1.26.2",
    "matplotlib": "3.8.2",
}


def _paketler():
    eksik, yanlis, tamam = [], [], []
    for ad, beklenen in BEKLENEN.items():
        try:
            m = importlib.import_module(ad)
        except ImportError:
            eksik.append(ad)
            continue
        bulunan = getattr(m, "__version__", "?")
        # "0.16.0+cu121" ile "0.16.0" ayni surum: yerel derleme etiketini yok say
        esit = bulunan == beklenen or bulunan.split("+")[0] == beklenen.split("+")[0]
        (tamam if esit else yanlis).append(f"{ad}={bulunan}")
    detay = ", ".join(tamam)
    if yanlis:
        detay += " | SURUM FARKLI: " + ", ".join(yanlis)
    if eksik:
        detay += " | EKSIK: " + ", ".join(eksik)
    return (not eksik and not yanlis, detay)


# --- 3. CUDA gercekten calisiyor mu -------------------------------------
def _cuda():
    import torch

    if not torch.cuda.is_available():
        return (False, "torch.cuda.is_available() False")
    ad = torch.cuda.get_device_name(0)
    toplam = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # sadece "available" demek yeterli degil: cekirdek gercekten kossun
    a = torch.randn(512, 512, device="cuda")
    beklenen = torch.eye(512, device="cuda")
    fark = (a @ beklenen - a).abs().max().item()
    if fark > 1e-3:
        return (False, f"GPU matmul yanlis sonuc verdi (fark={fark})")
    return (True, f"{ad}, {toplam:.1f} GiB, cuda {torch.version.cuda}, matmul dogru")


# --- 4. pytorch3d.transforms: derlenmis _C'ye ihtiyac duymuyor ----------
def _pytorch3d():
    import pytorch3d.transforms as t3

    if "pytorch3d._C" in sys.modules:
        return (False, "pytorch3d._C yuklendi - derlenmis eklentiye bagimlilik var")
    return (True, f"pytorch3d {importlib.import_module('pytorch3d').__version__}, _C yuklenmedi")


# --- 5. donme donusumu gidis-donus dogrulugu ----------------------------
def _donme_roundtrip():
    """Deponun kullandigi dort fonksiyon: ZYX euler <-> matris, kuaterniyon -> matris."""
    import torch
    import pytorch3d.transforms as t3

    torch.manual_seed(0)
    # ZYX Euler'de ikinci aci +-pi/2'ye yakinsa tekillik var; guvenli bantta kal
    aci = (torch.rand(64, 3) - 0.5) * torch.tensor([2.8, 1.2, 2.8])
    M = t3.euler_angles_to_matrix(aci, "ZYX")
    geri = t3.matrix_to_euler_angles(M, "ZYX")
    hata_aci = (geri - aci).abs().max().item()

    # matris gercekten ortogonal mi
    I = M @ M.transpose(-1, -2)
    hata_ort = (I - torch.eye(3)).abs().max().item()

    q = torch.nn.functional.normalize(torch.randn(64, 4), dim=-1)
    Mq = t3.quaternion_to_matrix(q)
    hata_q = (Mq @ Mq.transpose(-1, -2) - torch.eye(3)).abs().max().item()

    ok = hata_aci < 1e-5 and hata_ort < 1e-5 and hata_q < 1e-5
    return (ok, f"euler gidis-donus {hata_aci:.2e}, ortogonallik {hata_ort:.2e}, kuaterniyon {hata_q:.2e}")


# --- 6. referans depo sabit commit'te -----------------------------------
def _referans():
    if not REFERANS.exists():
        return (False, f"yok: {REFERANS}")
    commit = subprocess.run(
        ["git", "-C", str(REFERANS), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    kirli = subprocess.run(
        ["git", "-C", str(REFERANS), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if commit != BEKLENEN_COMMIT:
        return (False, f"commit {commit[:8]}, beklenen {BEKLENEN_COMMIT[:8]}")
    if kirli:
        return (False, "referans depo DEGISTIRILMIS - degisiklikler kendi depona tasinmali")
    return (True, f"commit {commit[:8]}, dokunulmamis")


# --- 7. deponun kendi modulleri yuklenebiliyor mu -----------------------
def _referans_import():
    sys.path.insert(0, str(REFERANS))
    try:
        for mod in ("utils.transform", "utils.network", "utils.loss",
                    "utils.metrics", "utils.Transf2DDFs", "utils.plot_functions"):
            importlib.import_module(mod)
    finally:
        sys.path.pop(0)
    return (True, "utils.* modulleri yuklendi (transform, network, loss, metrics, Transf2DDFs, plot_functions)")


# --- 8. EfficientNet-B1 agirliklari alinabiliyor mu ---------------------
def _omurga():
    import torch
    from torchvision.models import efficientnet_b1

    m = efficientnet_b1(weights=None)
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        y = m(x)
    p = sum(q.numel() for q in m.parameters()) / 1e6
    return (True, f"efficientnet_b1 ileri gecis tamam, cikis {tuple(y.shape)}, {p:.1f}M parametre")


# --- 9. sistem belleği (DDF uretimi islemcide, RAM kritik) --------------
def _ram():
    bayt = None
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, check=True,
            ).stdout
            bayt = int(out.strip())
        except Exception:
            bayt = None
    else:
        try:
            bayt = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, AttributeError, OSError):
            bayt = None
    if bayt is None:
        return (True, "okunamadi (kritik degil)")
    gb = bayt / 1024**3
    # README: DDF uretimi islemcide yapiliyor, bellek yetmezse "killed"
    return (gb >= 32, f"{gb:.1f} GiB (DDF uretimi islemcide, >=32 GiB onerilir)")


kontrol("Python 3.10", _python)
kontrol("Paket surumleri", _paketler)
kontrol("CUDA", _cuda)
kontrol("pytorch3d.transforms saf Python", _pytorch3d)
kontrol("Donme donusumu dogrulugu", _donme_roundtrip)
kontrol("Referans depo sabit", _referans)
kontrol("Referans modulleri", _referans_import)
kontrol("EfficientNet-B1 omurga", _omurga)
kontrol("Sistem bellegi", _ram)

gecen = sum(1 for _, g, _ in sonuclar if g)
print(f"\n{gecen}/{len(sonuclar)} kontrol gecti")
sys.exit(0 if gecen == len(sonuclar) else 1)
