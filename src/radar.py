"""Radar görüntüleri ve oyun koordinatı → radar pikseli dönüşümü (Faz 3, 2D harita).

Radar görüntüsü ve koordinat bilgisi oyunun kendi dosyalarından çıkarılır (awpy harita verisi indirilemiyor: CDN 403):
    pak01_dir.vpk:  panorama/images/overheadmaps/<harita>[_<kat>]_radar_psd.vtex_c   → PNG
                    resource/overviews/<harita>.txt   (pos_x, pos_y, scale, verticalsections)
Çıktı data/maps/radar/ (git DIŞI: Valve'ın telifli dosyaları; .tri'ler gibi yerelde üretilir):
    <harita>.json   {"pos_x", "pos_y", "scale", "size", "levels": [{"name", "image", "z_min", "z_max"}]}
    <harita>[_<kat>].png

    python src/radar.py build            (9 haritanın hepsi)
    python src/radar.py build de_inferno

Dönüşüm (HLTV overview tanımı; pos_x/pos_y = radarın SOL ÜST köşesinin dünya koordinatı):
    piksel_x = (X - pos_x) / scale          (sağa doğru artar)
    piksel_y = (pos_y - Y) / scale          (aşağı doğru artar; oyunda Y kuzeye doğru artar)
Çok katlı haritalarda (ör. nuke, vertigo) kat, Z'nin hangi [z_min, z_max) aralığında olduğuna göre seçilir.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RADAR_DIR = Path(os.environ.get("RADAR_DIR", "data/maps/radar"))
MAPS = ["cs_office", "de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_train", "de_vertigo"]


# ---------------------------------------------------------------- araç ve oyun yolları (README "Harita dosyaları")
# Sıra: (1) ortam değişkeni, (2) otomatik keşif, (3) genel yedek. Makineye özgü yol YOK. Yalnız harita dosyası
# üretirken gerekir (hazır analizleri açmak için gerekmez). Tembel: arayüz açılırken kayıt defteri / disk taranmaz.
VRF_NAME = "Source2Viewer-CLI"
CSGO_SUBDIR = Path("steamapps/common/Counter-Strike Global Offensive/game/csgo")
DEFAULT_STEAM = Path("C:/Program Files (x86)/Steam")


def vrf_path() -> Path:
    """Source2Viewer-CLI: VRF_CLI → PATH → <proje>/tools/** ve ~/tools/** → çıplak ad (bulunamadı)."""
    if os.environ.get("VRF_CLI"):
        return Path(os.environ["VRF_CLI"])
    hit = shutil.which(VRF_NAME)
    if hit:
        return Path(hit)
    for base in (Path(__file__).resolve().parent.parent / "tools", Path.home() / "tools"):
        if base.is_dir():
            found = sorted(base.glob(f"**/{VRF_NAME}.exe")) + sorted(base.glob(f"**/{VRF_NAME}"))
            if found:
                return found[0]
    return Path(VRF_NAME + (".exe" if os.name == "nt" else ""))


def steam_libraries() -> list[Path]:
    """Steam kütüphane klasörleri: kayıt defteri (Windows) + bilinen kurulum yerleri + libraryfolders.vdf."""
    roots = []
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
                roots.append(Path(winreg.QueryValueEx(k, "SteamPath")[0]))
        except OSError:
            pass
    roots += [DEFAULT_STEAM, Path.home() / ".steam/steam", Path.home() / ".local/share/Steam"]
    libs = []
    for r in roots:
        vdf = r / "steamapps" / "libraryfolders.vdf"
        found = re.findall(r'"path"\s*"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")) if vdf.exists() else []
        for lib in [r] + [Path(f.replace("\\\\", "\\")) for f in found]:
            if lib not in libs:
                libs.append(lib)
    return libs


def csgo_dir() -> Path:
    """CS2 'game/csgo' klasörü: CS2_CSGO_DIR → pak01_dir.vpk bulunan ilk Steam kütüphanesi → varsayılan Steam yeri."""
    if os.environ.get("CS2_CSGO_DIR"):
        return Path(os.environ["CS2_CSGO_DIR"])
    for lib in steam_libraries():
        if (lib / CSGO_SUBDIR / "pak01_dir.vpk").exists():
            return lib / CSGO_SUBDIR
    return DEFAULT_STEAM / CSGO_SUBDIR


def game_vpk() -> Path:
    return csgo_dir() / "pak01_dir.vpk"


# ---------------------------------------------------------------- overview (.txt, Valve KeyValues) okuma
def parse_kv(text: str) -> dict:
    """Basit KeyValues ayrıştırıcı: "anahtar" "değer" ve "anahtar" { ... } blokları; // yorumları atılır."""
    tokens = re.findall(r'"([^"]*)"|([{}])', re.sub(r"//[^\n]*", "", text))
    tokens = [a if a else b for a, b in tokens]

    def block(i):
        out = {}
        while i < len(tokens):
            t = tokens[i]
            if t == "}":
                return out, i + 1
            key = t
            nxt = tokens[i + 1]
            if nxt == "{":
                val, i = block(i + 2)
            else:
                val, i = nxt, i + 2
            out[key] = val
        return out, i

    root, _ = block(0)
    return root


def overview_info(text: str, map_name: str, images: dict[str, str], size: int) -> dict:
    """overview .txt → dönüşüm bilgisi. images: kat adı → png dosya adı ('default' ana görüntü)."""
    kv = parse_kv(text)
    body = kv.get(map_name) or next(iter(kv.values()))
    levels = []
    vs = body.get("verticalsections") or {}
    if vs:
        for name, sec in vs.items():
            levels.append(dict(name=name, image=images.get(name), z_min=float(sec["AltitudeMin"]),
                               z_max=float(sec["AltitudeMax"])))
    else:
        levels.append(dict(name="default", image=images.get("default"), z_min=-1e9, z_max=1e9))
    return dict(map=map_name, pos_x=float(body["pos_x"]), pos_y=float(body["pos_y"]), scale=float(body["scale"]),
                size=size, levels=levels, multi_level=len(levels) > 1)


# ---------------------------------------------------------------- dönüşüm
def world_to_pixel(info: dict, x, y):
    """Oyun (X, Y) → radar pikseli (sol üst 0,0). Diziler veya sayılar."""
    return (x - info["pos_x"]) / info["scale"], (info["pos_y"] - y) / info["scale"]


def pixel_to_world(info: dict, px, py):
    return px * info["scale"] + info["pos_x"], info["pos_y"] - py * info["scale"]


def level_of(info: dict, z: float) -> str:
    """Z yüksekliğine göre kat adı ('default' tek katlı haritalarda her zaman)."""
    for lv in info["levels"]:
        if lv["z_min"] <= z < lv["z_max"]:
            return lv["name"]
    return info["levels"][0]["name"]


def load_info(map_name: str, radar_dir: Path = RADAR_DIR) -> dict | None:
    f = Path(radar_dir) / f"{map_name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def fallback_info(xs, ys, size: int = 1024, margin: float = 0.06) -> dict | None:
    """Radar dosyası yoksa (başka bilgisayar, Valve dosyaları dağıtılmıyor): konumların sınırlarından kare, boş zemin.
    Aynı dönüşüm (world_to_pixel) kullanılır; görüntü yok, tek kat."""
    xs, ys = [float(v) for v in xs if v is not None], [float(v) for v in ys if v is not None]
    if not xs or not ys:
        return None
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * (1 + 2 * margin)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return dict(pos_x=cx - span / 2, pos_y=cy + span / 2, scale=span / size, size=size,
                levels=[dict(name="default", image=None, z_min=-1e9, z_max=1e9)], multi_level=False,
                image_missing=True, fallback=True)


# ---------------------------------------------------------------- oyun dosyalarından çıkarma
def build(map_name: str | None = None):
    from PIL import Image
    maps = [map_name] if map_name else MAPS
    RADAR_DIR.mkdir(parents=True, exist_ok=True)
    vrf, vpk = vrf_path(), game_vpk()
    listing = subprocess.run([str(vrf), "-i", str(vpk), "--vpk_list", "-f", "panorama/images/overheadmaps/"],
                             capture_output=True, text=True).stdout
    for m in maps:
        files = sorted(set(re.findall(rf"panorama/images/overheadmaps/({m}(?:_[a-z0-9]+)?)_radar_psd\.vtex_c", listing)))
        with tempfile.TemporaryDirectory() as tmp:
            wanted = [f"resource/overviews/{m}.txt"] + [f"panorama/images/overheadmaps/{f}_radar_psd.vtex_c" for f in files]
            subprocess.run([str(vrf), "-i", str(vpk), "-f", ",".join(wanted), "-o", tmp, "-d"],
                           capture_output=True, text=True, check=True)
            images = {}
            for f in files:
                level = "default" if f == m else f[len(m) + 1:]
                if level in ("vanity", "preview"):
                    continue
                png = Path(tmp) / "panorama/images/overheadmaps" / f"{f}_radar_psd.png"
                shutil.copy(png, RADAR_DIR / f"{f}.png")
                images[level] = f"{f}.png"
            size = Image.open(RADAR_DIR / images["default"]).size[0]
            info = overview_info((Path(tmp) / "resource/overviews" / f"{m}.txt").read_text(encoding="utf-8"), m, images, size)
        (RADAR_DIR / f"{m}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"{m}: {size}px, katlar {[lv['name'] for lv in info['levels']]}, görüntüler {images}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
