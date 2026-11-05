"""Radar dönüşümü (src/radar.py) testleri: overview ayrıştırma, dünya ↔ piksel, kat seçimi."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import radar as R  # noqa: E402

INFERNO = '''// HLTV overview
"de_inferno"
{
	"material"	"overviews/de_inferno"	// texture file
	"pos_x"		"-2087"	// upper left world coordinate
	"pos_y"		"3870"
	"scale"		"4.9"
}'''
NUKE = '''"de_nuke"
{
	"pos_x"		"-3453"
	"pos_y"		"2887"
	"scale"		"7"
	"verticalsections"
	{
		"default" { "AltitudeMax" "10000" "AltitudeMin" "-495" }
		"lower" { "AltitudeMax" "-495" "AltitudeMin" "-10000" }
	}
}'''


def test_parse_single_level():
    info = R.overview_info(INFERNO, "de_inferno", {"default": "de_inferno.png"}, 1024)
    assert (info["pos_x"], info["pos_y"], info["scale"]) == (-2087.0, 3870.0, 4.9)
    assert not info["multi_level"] and info["levels"][0]["image"] == "de_inferno.png"


def test_parse_multi_level_and_level_of():
    info = R.overview_info(NUKE, "de_nuke", {"default": "de_nuke.png", "lower": "de_nuke_lower.png"}, 1024)
    assert info["multi_level"] and [lv["name"] for lv in info["levels"]] == ["default", "lower"]
    assert R.level_of(info, -400) == "default" and R.level_of(info, -600) == "lower"
    assert R.level_of(info, -495) == "default"          # sınır: [z_min, z_max)


def test_world_pixel_roundtrip_and_axes():
    info = R.overview_info(INFERNO, "de_inferno", {"default": "x.png"}, 1024)
    assert R.world_to_pixel(info, -2087, 3870) == (0.0, 0.0)          # sol üst köşe
    px, py = R.world_to_pixel(info, -2087 + 4.9 * 100, 3870 - 4.9 * 50)
    assert px == pytest.approx(100) and py == pytest.approx(50)       # X sağa, Y yukarı (piksel aşağı)
    x, y = R.pixel_to_world(info, 512, 300)
    assert R.world_to_pixel(info, x, y) == (pytest.approx(512), pytest.approx(300))


@pytest.mark.skipif(not (R.RADAR_DIR / "de_inferno.json").exists(), reason="radar dosyaları yok (yerelde üretilir)")
def test_extracted_inferno_matches_game_overview():
    info = R.load_info("de_inferno")
    assert (info["pos_x"], info["pos_y"], info["scale"], info["size"]) == (-2087.0, 3870.0, 4.9, 1024)
    assert (R.RADAR_DIR / info["levels"][0]["image"]).exists()
