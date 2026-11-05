"""Gizlilik: dönüştürülmüş demolar ve uyumluluk raporları gerçek Steam ID / isim içermemeli.  python -m pytest tests"""
import json
import re
from pathlib import Path

import polars as pl
import pytest

CONVERTED = Path("data/raw/current_converted")
REPORTS = Path("reports/demo_compat")
STEAMID64 = re.compile(r"7656119\d{10}")


@pytest.mark.skipif(not CONVERTED.exists(), reason="dönüştürülmüş demo yok")
def test_converted_files_are_anonymous():
    for pq in CONVERTED.glob("*/*.parquet"):
        t = pl.read_parquet(pq)
        assert "name" not in t.columns
        assert t["steamid"].str.starts_with("Player_").all(), pq
    for js in CONVERTED.glob("*/*.json"):
        if js.name.endswith(".ids.json"):
            continue                                  # yerel eşleme dosyası (git dışı) — kasıtlı
        text = js.read_text(encoding="utf-8")
        assert not STEAMID64.search(text), js
        ev = json.loads(text)
        for name, rows in ev.items():
            if name in ("demo_meta", "CSstats_info"):     # demo/harita bilgisi (ör. demo_version_name), oyuncu değil
                continue
            if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                assert not [k for k in rows[0] if k.endswith("_name")], (js, name)


@pytest.mark.skipif(not REPORTS.exists(), reason="rapor yok")
def test_reports_are_anonymous():
    for f in REPORTS.glob("*"):
        assert not STEAMID64.search(f.read_text(encoding="utf-8")), f


def test_converted_dir_is_gitignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "data/raw/current_converted/x.parquet"])
    assert r.returncode == 0
    r = subprocess.run(["git", "check-ignore", "-q", "data/raw/current/x.dem"])
    assert r.returncode == 0


ANALYSIS_DIRS = [Path("reports/analysis"), Path("data/processed/analysis_test"), Path("data/processed/demo_cache"),
                 Path("data/raw/current/upload")]          # yükleme klasörü (iş durumu dosyaları)


@pytest.mark.parametrize("root", ANALYSIS_DIRS, ids=[str(d) for d in ANALYSIS_DIRS])
def test_analysis_outputs_are_anonymous(root):
    """Faz 1 çıktıları: Steam ID yok, isim sütunu yok, oyuncu sütunları Player_N (.ids.json yerel eşleme hariç)."""
    if not root.exists():
        pytest.skip("çıktı yok")
    for f in root.rglob("*"):
        if not f.is_file() or f.name.endswith(".ids.json"):
            continue
        if f.suffix == ".parquet":
            t = pl.read_parquet(f)
            assert not [c for c in t.columns if c in ("name", "player_name") or c.endswith("_name")
                        and c not in ("last_place_name", "active_weapon_name")], f
            for c in ("player", "steamid", "attacker", "victim"):
                if c in t.columns and t[c].dtype == pl.Utf8:
                    v = t[c].drop_nulls().filter(t[c].drop_nulls() != "")   # boş = kimlik yok (ör. CS2CD bomba ölümü)
                    assert v.str.starts_with("Player_").all() or c == "player" and v.str.contains("Player_").all(), (f, c)
            for c in [c for c in t.columns if t[c].dtype == pl.Utf8]:
                assert not t[c].drop_nulls().str.contains(r"7656119\d{10}").any(), (f, c)
        elif f.suffix in (".json", ".md"):
            assert not STEAMID64.search(f.read_text(encoding="utf-8")), f


def test_analysis_dirs_are_gitignored():
    import subprocess
    for p in ["reports/analysis/x/features.parquet", "data/processed/demo_cache/x/x.ids.json"]:
        assert subprocess.run(["git", "check-ignore", "-q", p]).returncode == 0, p


def test_raw_demo_filenames_not_in_outputs():
    """Ham .dem dosya adı (kişisel bilgi/maç kimliği taşıyabilir) rapor ve çıktılara girmez; demo_id = içerik hash'i."""
    stems = [p.stem for p in Path("data/raw/current").glob("*/*.dem")]
    if not stems:
        pytest.skip("yerel demo yok")
    roots = [Path("reports"), Path("notes"), Path("data/processed/analysis_test")]
    for root in roots:
        for f in root.rglob("*"):
            if f.suffix in (".md", ".json", ".txt", ".csv") and f.is_file():
                text = f.read_text(encoding="utf-8", errors="ignore")
                assert not any(s in text for s in stems), f
            elif f.suffix == ".parquet":
                t = pl.read_parquet(f)
                for c in [c for c in t.columns if t[c].dtype == pl.Utf8]:
                    v = t[c].drop_nulls()
                    assert not any(v.str.contains(s, literal=True).any() for s in stems), (f, c)


def test_upload_dir_has_no_identity_map_or_original_names():
    """yükleme klasöründe kimlik eşlemesi olmaz; dosyalar yalnız içerik hash'iyle adlandırılır."""
    up = Path("data/raw/current/upload")
    if not up.exists():
        pytest.skip("yükleme yok")
    import re
    for f in up.iterdir():
        assert not f.name.endswith(".ids.json"), f
        assert re.fullmatch(r"[0-9a-f]{12}\.(dem|job\.json|log)|analysis\.lock|incoming_[0-9a-f]{32}\.part", f.name), f
