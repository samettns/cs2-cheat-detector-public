"""Harita dosyalarını kullanıcının KENDİ CS2 kurulumundan üretir ve kontrol eder (Valve dosyaları dağıtılmaz).

    python src/setup_maps.py check     → hangi haritada radar görüntüsü / .tri var
    python src/setup_maps.py build     → 9 haritanın radar görüntüsü + .tri'si (Source2Viewer-CLI gerekir)
    python src/setup_maps.py build de_dust2

Gerekenler (README "Harita dosyaları"):
    VRF_CLI      = Source2Viewer-CLI.exe yolu (https://github.com/ValveResourceFormat/ValveResourceFormat/releases)
    CS2_CSGO_DIR = <Steam>/steamapps/common/Counter-Strike Global Offensive/game/csgo
    Ayarlanmazsa otomatik aranır: Source2Viewer-CLI PATH'te, <proje>/tools/ veya ~/tools/ altında; CS2 Steam
    kütüphanelerinde (kayıt defteri + libraryfolders.vdf). `check` bulunan yolları gösterir.
Neden: radar görüntüsü yalnız 2D gösterim içindir (yoksa arayüz boş zemin çizer). .tri ray-cast görünürlük içindir:
.tri yoksa YENİ demo analizinde görünürlük hesaplanamaz ve model skoru üretilmez (hazır analizler yine açılır).
Not: .tri oyunun o anki harita sürümünden üretilir; raporlanan sonuçlar geliştirme makinesindeki dosyalarla alındı.
"""
import sys
from pathlib import Path

import radar as R
import visibility as VIS

MAPS = R.MAPS


def status() -> list[tuple[str, bool, bool]]:
    return [(m, (R.RADAR_DIR / f"{m}.json").exists() and (R.RADAR_DIR / f"{m}.png").exists(),
             (Path(VIS.MAPS_DIR) / f"{m}.tri").exists()) for m in MAPS]


def check() -> None:
    print(f"{'harita':<12} radar  .tri")
    for m, r, t in status():
        print(f"{m:<12} {'var' if r else 'YOK':<6} {'var' if t else 'YOK'}")
    print()
    vrf, csgo = R.vrf_path(), R.csgo_dir()
    print(f"VRF_CLI: {vrf} ({'bulundu' if vrf.exists() else 'BULUNAMADI'})")
    print(f"CS2_CSGO_DIR: {csgo} ({'bulundu' if R.game_vpk().exists() else 'pak01_dir.vpk BULUNAMADI'})")


def build(map_name: str | None = None) -> None:
    if not R.vrf_path().exists():
        sys.exit(f"Source2Viewer-CLI bulunamadı: {R.vrf_path()} — VRF_CLI ortam değişkenini ayarlayın (README).")
    if not R.game_vpk().exists():
        sys.exit(f"CS2 dosyaları bulunamadı: {R.game_vpk()} — CS2_CSGO_DIR ortam değişkenini ayarlayın (README).")
    for m in [map_name] if map_name else MAPS:
        R.build(m)
        VIS.build_tri(m)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check"]:
        check()
    elif a[:1] == ["build"]:
        build(a[1] if len(a) > 1 else None)
    else:
        print(__doc__)
