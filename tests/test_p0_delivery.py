"""Faz 3.5 P0 (teslim paketi) testleri.

- Radar görüntüsü / JSON yoksa arayüz çökmez: aynı dönüşüm ya da konumlardan boş zemin (viewer_data.radar_info)
- Örnek analiz paketi (samples/analysis/): gizlilik denetimi, arayüz listesi, radar olmadan yükleme
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import radar as R  # noqa: E402
import sample_pack as SP  # noqa: E402
import viewer_data as V  # noqa: E402


def test_fallback_info_square_and_contains_points():
    info = R.fallback_info([-1000, 500, 200], [300, -2000, 0])
    assert info["fallback"] and info["image_missing"] and info["levels"][0]["image"] is None
    px, py = R.world_to_pixel(info, pl.Series([-1000.0, 500.0]), pl.Series([300.0, -2000.0]))
    assert all(0 <= v <= info["size"] for v in list(px) + list(py))
    assert R.fallback_info([], []) is None


def _work_with_context(tmp: Path) -> Path:
    w = tmp / "w"
    w.mkdir()
    pl.DataFrame({"X": [-500.0, 800.0, None], "Y": [100.0, -900.0, 5.0]}).write_parquet(w / "map_context.parquet")
    return w


def test_radar_info_without_any_radar_files(tmp_path):
    info = V.radar_info(_work_with_context(tmp_path), "de_dust2", radar_dir=tmp_path / "bos")
    assert info is not None and info["image_missing"] and info["levels"][0]["image"] is None


def test_radar_info_json_without_png_keeps_transform(tmp_path):
    rd = tmp_path / "radar"
    rd.mkdir()
    j = dict(pos_x=-2476, pos_y=3239, scale=4.4, size=1024, multi_level=False,
             levels=[dict(name="default", image="de_dust2.png", z_min=-1e9, z_max=1e9)])
    (rd / "de_dust2.json").write_text(json.dumps(j), encoding="utf-8")
    info = V.radar_info(_work_with_context(tmp_path), "de_dust2", radar_dir=rd)
    assert info["pos_x"] == -2476 and info["scale"] == 4.4 and info["image_missing"]
    assert info["levels"][0]["image"] is None


@pytest.mark.skipif(not SP.DST.exists(), reason="örnek paket yok")
def test_sample_pack_private_and_complete():
    dirs = [d for d in SP.DST.iterdir() if d.is_dir()]
    assert {d.name for d in dirs} >= set(SP.DEFAULT)
    for d in dirs:
        assert SP.privacy_problems(d) == [], d.name
        assert not (d / "_stages").exists()
    roles = {e["jury"]["role"] for e in V.list_analyses(SP.DST) if e.get("jury")}
    assert roles == {"dogrulanmis_pozitif", "hilecisiz", "yanlis_alarm"}


def test_privacy_check_catches_steamid_and_names(tmp_path):
    w = tmp_path / "x"
    w.mkdir()
    fake_id = "7656119" + "8" * 10      # sahte desen, parçalı yazıldı (pre-commit kancası gerçek ID arar)
    (w / "metadata.json").write_text(json.dumps({"note": fake_id}), encoding="utf-8")
    pl.DataFrame({"attacker": ["GercekIsim"]}).write_parquet(w / "kills.parquet")
    probs = SP.privacy_problems(w)
    assert any("Steam ID" in p for p in probs) and any("anonim olmayan" in p for p in probs)


@pytest.mark.skipif(not SP.DST.exists(), reason="örnek paket yok")
def test_samples_load_without_radar(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_info", lambda m, radar_dir=None: None)
    for e in V.list_analyses(SP.DST):
        d = V.load(e)
        assert d["radar"] is not None and d["radar"]["image_missing"]
        assert d["players"].height > 0


def test_local_analysis_wins_over_sample(tmp_path, monkeypatch):
    for root in ("local", "samples"):
        w = tmp_path / root / "abc"
        w.mkdir(parents=True)
        (w / "metadata.json").write_text(json.dumps(dict(demo_id="abc", map="de_dust2", source=root)), encoding="utf-8")
        pl.DataFrame({"x": [1]}).write_parquet(w / "player_scores.parquet")
    monkeypatch.setattr(V, "ANALYSIS_ROOT", tmp_path / "local")
    monkeypatch.setattr(V, "SAMPLES_ROOT", tmp_path / "samples")
    E = V.list_analyses()
    assert len(E) == 1 and E[0]["source"] == "local"


# ---------------------------------------------------------------- taşınabilirlik: araç / oyun yolları
def test_source_has_no_machine_specific_paths():
    """Kaynak kodda geliştirme makinesine özgü kullanıcı klasörü yolu yok (C:/Users/<ad>/..., D:/SteamLibrary)."""
    import re
    pat = re.compile(r"(?i)[a-z]:[\/]+users[\/]+\w|steamlibrary")
    root = Path(__file__).parent.parent
    bad = [f"{f.relative_to(root)}:{i}" for d in ("src", "app") for f in (root / d).rglob("*.py")
           for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    assert bad == []


def test_tool_paths_env_first_then_discovery_then_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("VRF_CLI", str(tmp_path / "x.exe"))
    monkeypatch.setenv("CS2_CSGO_DIR", str(tmp_path / "csgo"))
    assert R.vrf_path() == tmp_path / "x.exe" and R.game_vpk() == tmp_path / "csgo" / "pak01_dir.vpk"
    monkeypatch.delenv("VRF_CLI")
    monkeypatch.delenv("CS2_CSGO_DIR")
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(R.shutil, "which", lambda name: None)
    monkeypatch.setattr(R, "DEFAULT_STEAM", tmp_path / "steam")
    monkeypatch.setattr(R, "__file__", str(tmp_path / "proje" / "src" / "radar.py"))
    if R.os.name == "nt":                                            # gerçek kayıt defteri okunmaz
        import winreg
        monkeypatch.setattr(winreg, "OpenKey", lambda *a: (_ for _ in ()).throw(OSError()))
    assert R.vrf_path().name.startswith(R.VRF_NAME) and not R.vrf_path().exists()      # genel yedek: çıplak ad
    assert R.csgo_dir() == tmp_path / "steam" / R.CSGO_SUBDIR
    exe = home / "tools" / "vrf-9" / f"{R.VRF_NAME}.exe"                                 # keşif: ~/tools/**
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert R.vrf_path() == exe
    lib = tmp_path / "Oyunlar" / "Steam Kutuphanesi"                                    # keşif: libraryfolders.vdf
    (lib / R.CSGO_SUBDIR).mkdir(parents=True)
    (lib / R.CSGO_SUBDIR / "pak01_dir.vpk").write_text("")
    (tmp_path / "steam" / "steamapps").mkdir(parents=True)
    vdf = '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n' % str(lib).replace("\\", "\\\\")
    (tmp_path / "steam" / "steamapps" / "libraryfolders.vdf").write_text(vdf, encoding="utf-8")
    assert R.csgo_dir() == lib / R.CSGO_SUBDIR
