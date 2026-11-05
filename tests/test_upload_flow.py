"""arayüzden .dem yükleme akışı (src/upload_flow.py). Ağır hat yerine sahte alt süreç komutları kullanılır;
gerçek demo ile uçtan uca deneme geliştirme sırasında elle yapıldı."""
import io
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import upload_flow as U  # noqa: E402

DEM = U.MAGIC + b"\x00" * (U.MIN_BYTES + 100)


def _dirs(tmp_path):
    return dict(upload_dir=tmp_path / "up", analysis_root=tmp_path / "ana", cache_root=tmp_path / "cache")


# ---------------------------------------------------------------- dosya kontrolü
def test_rejects_non_dem_and_size_limits():
    with pytest.raises(U.UploadError, match=".dem"):
        U.check_name_and_size("mac.zip", 5 * U.MIN_BYTES)
    with pytest.raises(U.UploadError, match="büyük"):
        U.check_name_and_size("mac.dem", U.MAX_BYTES + 1)
    with pytest.raises(U.UploadError, match="küçük"):
        U.check_name_and_size("mac.dem", 1000)
    U.check_name_and_size("MAC.DEM", 5 * U.MIN_BYTES)


def test_rejects_wrong_header_and_leaves_nothing(tmp_path):
    bad = b"HL2DEMO\x00" + b"\x00" * U.MIN_BYTES
    with pytest.raises(U.UploadError, match="PBDEMS2"):
        U.save_upload(io.BytesIO(bad), "x.dem", len(bad), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_oversize_stream_rejected_even_if_declared_small(tmp_path):
    with pytest.raises(U.UploadError):
        U.save_upload(io.BytesIO(DEM), "x.dem", U.MIN_BYTES, tmp_path, max_bytes=U.MIN_BYTES + 10)
    assert list(tmp_path.iterdir()) == []


def test_demo_id_is_content_hash_like_analyze_demo_and_name_not_stored(tmp_path):
    import analyze_demo as AD
    did, path = U.save_upload(io.BytesIO(DEM), "Gercek_Takim_vs_Rakip_finali.dem", len(DEM), tmp_path)
    assert did == AD.sha12(path) and path.name == f"{did}.dem"
    assert [p.name for p in tmp_path.iterdir()] == [f"{did}.dem"]           # orijinal ad yok, .part kalmadı
    did2, _ = U.save_upload(io.BytesIO(DEM), "baska_ad.dem", len(DEM), tmp_path)   # aynı içerik → aynı kimlik
    assert did2 == did and len(list(tmp_path.iterdir())) == 1


# ---------------------------------------------------------------- kilit
def test_single_analysis_lock_and_stale_lock(tmp_path):
    assert U.acquire_lock("aaa", tmp_path)
    assert not U.acquire_lock("bbb", tmp_path)          # aynı anda ikinci analiz yok (aynı hash dahil)
    assert not U.acquire_lock("aaa", tmp_path)
    assert U.running_job(tmp_path) == "aaa"
    U.release_lock("aaa", tmp_path)
    assert U.running_job(tmp_path) is None
    (tmp_path / U.LOCK_NAME).write_text(json.dumps(dict(did="ccc", pid=999999999)), encoding="utf-8")
    assert U.running_job(tmp_path) is None              # sahibi ölü → bayat
    assert U.acquire_lock("ddd", tmp_path)              # bayat kilit geri alınır


# ---------------------------------------------------------------- koşucu (sahte alt süreçler)
FAKE_FAZ1 = r'''
import json, sys
from pathlib import Path
did, ana, cache = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
(cache / did).mkdir(parents=True); (cache / did / f"{did}.ids.json").write_text("{}")
(cache / did / "adapter_meta.json").write_text("{}")
w = ana / did; (w / "_stages").mkdir(parents=True)
import polars as pl
for f in ["features", "kills", "kill_windows", "map_context", "utility", "_stages/aim", "_stages/visibility_features"]:
    pl.DataFrame({"a": [1]}).write_parquet(w / f"{f}.parquet")
(w / "qc.json").write_text("{}"); (w / "qc.md").write_text("x")
(w / "metadata.json").write_text(json.dumps({"demo_id": did}))
'''
FAKE_FAZ2 = r'''
import json, sys
from pathlib import Path
import polars as pl
w = Path(sys.argv[1])
for f in ["player_scores", "event_scores", "map_events"]:
    pl.DataFrame({"player": ["Player_1"]}).write_parquet(w / f"{f}.parquet")
(w / "model_result.json").write_text(json.dumps({"status": "skor üretildi"}, ensure_ascii=False), encoding="utf-8")
(w / "model_report.md").write_text("x")
'''


def _setup(tmp_path, faz1=FAKE_FAZ1, faz2=FAKE_FAZ2):
    D = _dirs(tmp_path)
    did, dem = U.save_upload(io.BytesIO(DEM), "x.dem", len(DEM), D["upload_dir"])
    s1, s2 = tmp_path / "f1.py", tmp_path / "f2.py"
    s1.write_text(faz1, encoding="utf-8")
    s2.write_text(faz2, encoding="utf-8")
    cmds = dict(faz1=[sys.executable, str(s1), did, str(D["analysis_root"]), str(D["cache_root"])],
                faz2=[sys.executable, str(s2), str(D["analysis_root"] / did)])
    U.acquire_lock(did, D["upload_dir"])
    return D, did, dem, cmds


def test_success_verifies_then_deletes_raw_demo_and_identity_map(tmp_path):
    D, did, dem, cmds = _setup(tmp_path)
    job = U.run(did, "valve", commands=cmds, **D)
    assert job["state"] == "done", job
    assert not dem.exists()                                        # ham demo silindi (varsayılan)
    assert not (D["cache_root"] / did).exists()                    # .ids.json dahil önbellek silindi
    assert not list(D["upload_dir"].glob("*.ids.json")) and not list(D["upload_dir"].glob("*.log"))
    assert U.verify_outputs(did, D["analysis_root"]) == [] and U.is_done(did, D["analysis_root"])
    assert U.running_job(D["upload_dir"]) is None                  # kilit bırakıldı
    assert [s[2] for s in U.stage_status(did, D["analysis_root"], D["cache_root"])] == [True] * 5
    assert U.read_job(did, D["upload_dir"])["peak_ram_mb"]["faz1"] > 0


def test_keep_raw_option_keeps_only_dem(tmp_path):
    D, did, dem, cmds = _setup(tmp_path)
    job = U.run(did, "valve", keep_raw=True, commands=cmds, **D)
    assert job["state"] == "done" and dem.exists()
    assert not (D["cache_root"] / did).exists()                    # kimlik eşlemesi yine silinir


def test_failure_keeps_everything_for_debug_and_resume(tmp_path):
    D, did, dem, cmds = _setup(tmp_path, faz1=FAKE_FAZ1 + '\nprint("HATA: demo ayrıştırılamadı (x)", file=sys.stderr)\nsys.exit(2)\n')
    job = U.run(did, "valve", commands=cmds, **D)
    assert job["state"] == "failed" and job["failed_step"] == "faz1"
    assert "ayrıştırılamadı" in job["message"]
    assert dem.exists() and (D["cache_root"] / did).exists()        # otomatik silme YOK
    assert (D["analysis_root"] / did / "_stages" / "aim.parquet").exists()   # --resume için kalır
    assert U.running_job(D["upload_dir"]) is None
    U.discard(did, **D)                                             # kullanıcı seçerse temizlenir
    assert not dem.exists() and not (D["cache_root"] / did).exists() and not (D["analysis_root"] / did).exists()


def test_incomplete_outputs_do_not_delete_raw(tmp_path):
    D, did, dem, cmds = _setup(tmp_path, faz2=FAKE_FAZ2.replace('"model_report.md"', '"baska.md"'))
    job = U.run(did, "valve", commands=cmds, **D)
    assert job["state"] == "failed" and job["failed_step"] == "dogrulama"
    assert dem.exists() and (D["cache_root"] / did).exists()


def test_not_applicable_reasons_shown(tmp_path):
    f2 = FAKE_FAZ2.replace('{"status": "skor üretildi"}', '{"status": "uygulanamaz", "model_v1_reasons": ["harita desteklenmiyor"]}')
    D, did, dem, cmds = _setup(tmp_path, faz2=f2)
    job = U.run(did, "valve", commands=cmds, **D)
    assert job["state"] == "failed" and "harita desteklenmiyor" in job["message"] and dem.exists()


def test_friendly_errors():
    assert "64" in U.friendly_error("HATA: tick hızı 128.0, beklenen 64")
    assert "Bellek" in U.friendly_error("numpy.core._exceptions._ArrayMemoryError: Unable to allocate")
    assert U.friendly_error("HATA: bir şey oldu").startswith("Analiz durdu: bir şey oldu")
    assert "beklenmedik" in U.friendly_error("Traceback ...", 1)


def test_upload_dir_is_git_ignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", str(U.UPLOAD_DIR / "abc.dem")], cwd=U.ROOT)
    assert r.returncode == 0
    r = subprocess.run(["git", "check-ignore", "-q", str(U.CACHE_ROOT / "abc" / "abc.ids.json")], cwd=U.ROOT)
    assert r.returncode == 0
