"""Arayüzden .dem yükleme → Faz 1 + Faz 2 analizi. Model/özellik/eşik/skor mantığı DEĞİŞMEZ:
mevcut komutlar (src/analyze_demo.py, src/score_demo.py) AYRI SÜREÇLERDE, olduğu gibi çalıştırılır.

Akış:
  1. save_upload: uzantı (.dem) + boyut (≤ 1 GB) + CS2 başlığı (PBDEMS2) kontrolü; dosya parça parça
     data/raw/current/upload/ altına yazılır (git dışı) ve aynı anda SHA-256 hesaplanır → demo_id = ilk 12 hane
     (analyze_demo.sha12 ile aynı). Orijinal dosya adı HİÇBİR YERE yazılmaz.
  2. Aynı demo_id daha önce başarıyla analiz edildiyse yeniden çalıştırılmaz (is_done).
  3. start_job: tek analiz kilidi (O_EXCL ile atomik; ölü süreçten kalan kilit geri alınır) → ayrı bir koşucu süreç
     (`python src/upload_flow.py run <demo_id>`) analyze_demo.py'yi, sonra score_demo.py'yi alt süreç olarak çalıştırır;
     durum <demo_id>.job.json'a yazılır (aşama, süreler, tepe RAM, hata).
  4. Başarıda çıktılar doğrulanır (verify_outputs); ancak ondan sonra ham .dem ve kimlik eşlemesini içeren dönüştürme
     önbelleği (data/processed/demo_cache/<demo_id>/, .ids.json dahil) silinir. "Ham demoyu sakla" seçilirse yalnız
     .dem kalır; kimlik eşlemesi yine silinir.
  5. Hata olursa HİÇBİR ŞEY otomatik silinmez: _stages/ ve önbellek --resume için kalır; arayüz "kaldığı yerden
     yeniden dene" veya "ham demoyu ve geçici dosyaları sil" seçeneklerini sunar.

    python src/upload_flow.py run <demo_id> [--source valve] [--keep-raw] [--resume]     (arayüz çağırır)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path("data/raw/current/upload")          # git dışı (.gitignore: data/raw/*)
ANALYSIS_ROOT = Path("reports/analysis")
CACHE_ROOT = Path("data/processed/demo_cache")
LOCK_NAME = "analysis.lock"
MAX_BYTES = 1024 ** 3                                  # ~1 GB (Streamlit server.maxUploadSize = 1024 ile aynı)
MIN_BYTES = 1024 * 1024                                # 1 MB altı gerçek maç demosu olamaz
MAGIC = b"PBDEMS2"
SOURCES = ["valve", "faceit", "hltv", "other"]
STAGES = [   # (anahtar, arayüz adı, bitti sayılma koşulu)
    ("donusturme", "Demo dönüştürme (anonim Player_N)", "cache_meta"),
    ("nisan", "Nişan özellikleri", "_stages/aim.parquet"),
    ("gorunurluk", "Görünürlük (ray-cast, en uzun adım)", "_stages/visibility_features.parquet"),
    ("baglam", "Bağlam + kalite kontrol", "metadata.json"),
    ("skor", "Model skorları (Faz 2)", "model_result.json"),
]
REQUIRED_OUTPUTS = ["features.parquet", "kills.parquet", "kill_windows.parquet", "map_context.parquet",
                    "utility.parquet", "qc.json", "qc.md", "metadata.json", "player_scores.parquet",
                    "event_scores.parquet", "map_events.parquet", "model_result.json", "model_report.md"]


class UploadError(Exception):
    """Kullanıcıya olduğu gibi gösterilebilen, anlaşılır yükleme hatası."""


# ---------------------------------------------------------------- 1) dosya kontrolü + kaydetme
def check_name_and_size(name: str, size: int, max_bytes: int = MAX_BYTES) -> None:
    if not str(name).lower().endswith(".dem"):
        raise UploadError("Yalnız .dem uzantılı CS2 demo dosyaları kabul edilir (.zip/.gz/.bz2 ise önce açın).")
    if size > max_bytes:
        raise UploadError(f"Dosya çok büyük ({size / 1e6:.0f} MB); sınır {max_bytes / 1e6:.0f} MB.")
    if size < MIN_BYTES:
        raise UploadError(f"Dosya çok küçük ({size / 1e3:.0f} KB); tam bir maç demosu değil.")


def save_upload(fileobj, name: str, size: int, upload_dir: Path = UPLOAD_DIR, max_bytes: int = MAX_BYTES) -> tuple[str, Path]:
    """Yüklenen dosyayı parça parça diske yazar, SHA-256 hesaplar → (demo_id, yol). Ad diske yazılmaz."""
    check_name_and_size(name, size, max_bytes)
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    part = upload_dir / f"incoming_{uuid.uuid4().hex}.part"
    h, n, head = hashlib.sha256(), 0, b""
    try:
        with open(part, "wb") as f:
            while chunk := fileobj.read(1 << 20):
                if not head:
                    head = chunk[:8]
                    if not head.startswith(MAGIC):
                        raise UploadError("Bu dosya bir CS2 demosu değil ya da bozuk (PBDEMS2 başlığı yok). "
                                          "CS:GO demoları ve sıkıştırılmış dosyalar desteklenmez.")
                n += len(chunk)
                if n > max_bytes:
                    raise UploadError(f"Dosya {max_bytes / 1e6:.0f} MB sınırını aşıyor.")
                h.update(chunk)
                f.write(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    did = h.hexdigest()[:12]
    dest = upload_dir / f"{did}.dem"
    if dest.exists():               # aynı içerik zaten var (ad = içerik hash'i); analiz okuyor olabilir → dokunma
        part.unlink(missing_ok=True)
    else:
        try:
            os.rename(part, dest)   # atomik; iki oturum aynı anda yazarsa ikincisi FileExistsError alır
        except FileExistsError:
            part.unlink(missing_ok=True)
    return did, dest


def is_done(did: str, analysis_root: Path = ANALYSIS_ROOT) -> bool:
    """Bu demo daha önce başarıyla analiz + skorlandı mı?"""
    w = Path(analysis_root) / did
    if not (w / "model_result.json").exists() or not (w / "player_scores.parquet").exists():
        return False
    return json.loads((w / "model_result.json").read_text(encoding="utf-8")).get("status") == "skor üretildi"


# ---------------------------------------------------------------- 2) tek analiz kilidi
def _pid_alive(pid: int) -> bool:
    import psutil
    return psutil.pid_exists(pid)


def acquire_lock(did: str, upload_dir: Path = UPLOAD_DIR, pid: int | None = None) -> bool:
    """Aynı anda tek analiz. O_EXCL ile atomik oluşturma; sahibi ölmüş kilit geri alınır."""
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    lock = upload_dir / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            cur = read_lock(upload_dir)
            if cur and _pid_alive(cur.get("pid", -1)):
                return False
            lock.unlink(missing_ok=True)       # bayat kilit (süreç ölmüş)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(did=did, pid=pid or os.getpid(), started=time.time()), f)
        return True
    return False


def read_lock(upload_dir: Path = UPLOAD_DIR) -> dict | None:
    try:
        return json.loads((Path(upload_dir) / LOCK_NAME).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def set_lock_pid(pid: int, upload_dir: Path = UPLOAD_DIR) -> None:
    lk = read_lock(upload_dir) or {}
    lk["pid"] = pid
    (Path(upload_dir) / LOCK_NAME).write_text(json.dumps(lk), encoding="utf-8")


def release_lock(did: str, upload_dir: Path = UPLOAD_DIR) -> None:
    cur = read_lock(upload_dir)
    if cur and cur.get("did") == did:
        (Path(upload_dir) / LOCK_NAME).unlink(missing_ok=True)


def running_job(upload_dir: Path = UPLOAD_DIR) -> str | None:
    """Şu anda çalışan analizin demo_id'si (kilit sahibi yaşıyorsa)."""
    cur = read_lock(upload_dir)
    return cur["did"] if cur and _pid_alive(cur.get("pid", -1)) else None


# ---------------------------------------------------------------- 3) iş durumu
def job_file(did: str, upload_dir: Path = UPLOAD_DIR) -> Path:
    return Path(upload_dir) / f"{did}.job.json"


def read_job(did: str, upload_dir: Path = UPLOAD_DIR) -> dict | None:
    try:
        return json.loads(job_file(did, upload_dir).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def write_job(job: dict, upload_dir: Path = UPLOAD_DIR) -> None:
    f = job_file(job["did"], upload_dir)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, f)          # okuyucu yarım dosya görmez


def stage_status(did: str, analysis_root: Path = ANALYSIS_ROOT, cache_root: Path = CACHE_ROOT) -> list[tuple[str, str, bool]]:
    """Dosya düzeninden aşamalar: (anahtar, ad, bitti mi). Analiz kodu değiştirilmeden ilerleme okunur."""
    w, c = Path(analysis_root) / did, Path(cache_root) / did
    out = []
    for key, label, marker in STAGES:
        done = (c / "adapter_meta.json").exists() or (w / "metadata.json").exists() if marker == "cache_meta" \
            else (w / marker).exists()
        out.append((key, label, done))
    # sıralı: sonraki aşama bittiyse öncekiler de bitmiştir (ör. önbellek silinmiş olabilir)
    last = max((i for i, s in enumerate(out) if s[2]), default=-1)
    return [(k, lbl, i <= last) for i, (k, lbl, _) in enumerate(out)]


def start_job(did: str, source: str, keep_raw: bool = False, resume: bool = False,
              upload_dir: Path = UPLOAD_DIR) -> bool:
    """Kilidi alıp koşucu süreci başlatır. Kilit alınamazsa (başka analiz sürüyor) False."""
    if not acquire_lock(did, upload_dir):
        return False
    write_job(dict(did=did, state="running", source=source, keep_raw=keep_raw, started=time.time(),
                   stage_times={}, message=None), upload_dir)
    cmd = [sys.executable, str(ROOT / "src" / "upload_flow.py"), "run", did, "--source", source]
    cmd += ["--keep-raw"] if keep_raw else []
    cmd += ["--resume"] if resume else []
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    try:
        p = subprocess.Popen(cmd, cwd=ROOT, creationflags=flags, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"}, start_new_session=os.name != "nt")
    except OSError:
        release_lock(did, upload_dir)
        raise
    set_lock_pid(p.pid, upload_dir)      # kilit artık koşucuya ait (arayüz kapansa da analiz sürer)
    return True


# ---------------------------------------------------------------- 4) doğrulama, temizlik, hata metni
def verify_outputs(did: str, analysis_root: Path = ANALYSIS_ROOT) -> list[str]:
    """Ham demo silinmeden önce: gerekli bütün çıktılar var ve okunabilir mi? Sorun listesi (boşsa geçerli)."""
    import polars as pl
    w = Path(analysis_root) / did
    problems = [f"eksik: {f}" for f in REQUIRED_OUTPUTS if not (w / f).exists()]
    if problems:
        return problems
    for f in REQUIRED_OUTPUTS:
        try:
            if f.endswith(".parquet"):
                pl.read_parquet_schema(w / f)
            elif f.endswith(".json"):
                json.loads((w / f).read_text(encoding="utf-8"))
        except Exception as e:      # noqa: BLE001 — hangi dosyanın bozuk olduğu raporlanır
            problems.append(f"okunamadı: {f} ({type(e).__name__})")
    meta = json.loads((w / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("demo_id") != did:
        problems.append(f"metadata demo_id uyuşmuyor: {meta.get('demo_id')}")
    if json.loads((w / "model_result.json").read_text(encoding="utf-8")).get("status") != "skor üretildi":
        problems.append("skor üretilmedi")
    if pl.read_parquet(w / "player_scores.parquet").height == 0:
        problems.append("oyuncu skoru yok")
    return problems


def cleanup_after_success(did: str, keep_raw: bool, upload_dir: Path = UPLOAD_DIR,
                          cache_root: Path = CACHE_ROOT) -> list[str]:
    """Doğrulanmış başarıdan SONRA: kimlik eşlemesini içeren önbellek her zaman, ham .dem varsayılan olarak silinir."""
    removed = []
    c = Path(cache_root) / did
    if c.exists():
        shutil.rmtree(c)
        removed.append(str(c))
    if not keep_raw:
        dem = Path(upload_dir) / f"{did}.dem"
        if dem.exists():
            dem.unlink()
            removed.append(str(dem))
    log = Path(upload_dir) / f"{did}.log"
    log.unlink(missing_ok=True)
    return removed


def discard(did: str, upload_dir: Path = UPLOAD_DIR, cache_root: Path = CACHE_ROOT,
            analysis_root: Path = ANALYSIS_ROOT) -> None:
    """Başarısız analizden sonra kullanıcı isterse: ham demo, önbellek (kimlik eşlemesi), yarım çıktılar, günlük."""
    if running_job(upload_dir) == did:
        raise UploadError("Analiz hâlâ çalışıyor; bitmeden silinemez.")
    for p in (Path(upload_dir) / f"{did}.dem", Path(upload_dir) / f"{did}.log", job_file(did, upload_dir)):
        p.unlink(missing_ok=True)
    for d in (Path(cache_root) / did,):
        if d.exists():
            shutil.rmtree(d)
    w = Path(analysis_root) / did
    if w.exists() and not is_done(did, analysis_root):
        shutil.rmtree(w)


def friendly_error(stderr: str, returncode: int | None = None) -> str:
    """Alt süreç çıktısı → kullanıcıya anlaşılır Türkçe mesaj."""
    s = stderr or ""
    rules = [
        ("PBDEMS2", "Dosya bir CS2 demosu değil ya da bozuk."),
        ("demo ayrıştırılamadı", "Demo ayrıştırılamadı: dosya bozuk, yarım indirilmiş ya da desteklenmeyen bir sürüm."),
        ("tick hızı", "Bu demonun tick hızı 64 değil; Model v1 64 tick verisiyle eğitildi, bu demoya uygulanamaz."),
        ("zorunlu", "Demoda analiz için gereken alanlar/olaylar eksik (ör. kısaltılmış veya GOTV olmayan demo)."),
        ("MemoryError", "Bellek yetmedi. Diğer programları kapatıp kaldığı yerden yeniden deneyin."),
        ("ArrayMemoryError", "Bellek yetmedi. Diğer programları kapatıp kaldığı yerden yeniden deneyin."),
        ("No space left", "Diskte yer kalmadı."),
    ]
    for key, msg in rules:
        if key in s:
            return msg
    hata = [ln[len("HATA:"):].strip() for ln in s.splitlines() if ln.startswith("HATA:")]
    if hata:
        return f"Analiz durdu: {hata[-1]}"
    return f"Analiz beklenmedik şekilde durdu (çıkış kodu {returncode}). Ayrıntı teknik günlükte."


# ---------------------------------------------------------------- 5) koşucu (ayrı süreç)
def _run_step(cmd: list[str], log, job: dict, upload_dir: Path) -> tuple[int, str, float]:
    """Alt süreci çalıştırır, tepe RAM'i (alt süreç + çocukları) 0.5 sn'de bir ölçer. → (kod, stderr, tepe MB)."""
    import psutil
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}, text=True, encoding="utf-8",
                         errors="replace")
    peak = 0.0
    import threading
    out_buf, err_buf = [], []
    t1 = threading.Thread(target=lambda: out_buf.extend(p.stdout), daemon=True)
    t2 = threading.Thread(target=lambda: err_buf.extend(p.stderr), daemon=True)
    t1.start(), t2.start()
    while p.poll() is None:
        try:
            pr = psutil.Process(p.pid)
            rss = pr.memory_info().rss + sum(c.memory_info().rss for c in pr.children(recursive=True))
            peak = max(peak, rss / 1e6)
        except psutil.Error:
            pass
        time.sleep(0.5)
    t1.join(), t2.join()
    log.write(f"$ {' '.join(cmd[1:])}\n{''.join(out_buf)}{''.join(err_buf)}\n")
    log.flush()
    return p.returncode, "".join(err_buf) + "".join(out_buf[-20:]), peak


def run(did: str, source: str, keep_raw: bool = False, resume: bool = False, upload_dir: Path = UPLOAD_DIR,
        analysis_root: Path = ANALYSIS_ROOT, cache_root: Path = CACHE_ROOT, commands: dict | None = None) -> dict:
    """Koşucu: Faz 1 → Faz 2 → doğrulama → temizlik. commands testlerde sahte komut vermek için."""
    upload_dir = Path(upload_dir)
    dem = upload_dir / f"{did}.dem"
    job = read_job(did, upload_dir) or dict(did=did, source=source, keep_raw=keep_raw, started=time.time())
    job.update(state="running", stage_times={}, message=None, peak_ram_mb={})
    write_job(job, upload_dir)
    cmds = commands or {
        "faz1": [sys.executable, "src/analyze_demo.py", str(dem), "--source", source] + (["--resume"] if resume else []),
        "faz2": [sys.executable, "src/score_demo.py", str(Path(analysis_root) / did)],
    }
    t0 = time.time()
    try:
        with open(upload_dir / f"{did}.log", "a", encoding="utf-8") as log:
            for step in ("faz1", "faz2"):
                ts = time.time()
                code, err, peak = _run_step(cmds[step], log, job, upload_dir)
                job["stage_times"][step] = round(time.time() - ts, 1)
                job["peak_ram_mb"][step] = round(peak)
                if code != 0:
                    job.update(state="failed", failed_step=step, message=friendly_error(err, code),
                               log_tail=err[-2000:])
                    return job
                write_job(job, upload_dir)
        problems = verify_outputs(did, analysis_root)
        if problems:
            res = Path(analysis_root) / did / "model_result.json"
            reasons = []
            if res.exists():
                reasons = json.loads(res.read_text(encoding="utf-8")).get("model_v1_reasons") or []
            msg = ("Model v1 bu demoya uygulanamaz: " + "; ".join(reasons)) if reasons else \
                "Analiz çıktıları doğrulanamadı: " + "; ".join(problems[:3])
            job.update(state="failed", failed_step="dogrulama", message=msg, problems=problems)
            return job
        job["removed"] = cleanup_after_success(did, keep_raw, upload_dir, cache_root)
        job["raw_kept"] = keep_raw
        job.update(state="done")
        return job
    except Exception as e:      # noqa: BLE001 — koşucu çökmesi de kullanıcıya anlaşılır gösterilir
        job.update(state="failed", failed_step="kosucu", message=friendly_error(repr(e)), log_tail=repr(e))
        return job
    finally:
        job["seconds_total"] = round(time.time() - t0, 1)
        job["finished"] = time.time()
        write_job(job, upload_dir)
        release_lock(did, upload_dir)


if __name__ == "__main__":
    import argparse
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("did")
    ap.add_argument("--source", default="other", choices=SOURCES)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        if (read_lock() or {}).get("did") != a.did:      # doğrudan komut satırından: kilidi kendisi alır
            if not acquire_lock(a.did):
                print("Başka bir analiz çalışıyor.", file=sys.stderr)
                sys.exit(3)
        else:
            set_lock_pid(os.getpid())
        r = run(a.did, a.source, keep_raw=a.keep_raw, resume=a.resume)
        print(json.dumps({k: r.get(k) for k in ("state", "message", "seconds_total", "stage_times", "peak_ram_mb")},
                         ensure_ascii=False))
        sys.exit(0 if r["state"] == "done" else 1)


# ---------------------------------------------------------------- 5) ilerleme gösterimi (Faz 3.5) — yalnız okuma
# 7 adımlı gösterim mevcut aşama işaretlerine birebir eşlenir; analiz kodu değişmez. Yüzde / kalan süre ÜRETİLMEZ.
DISPLAY_STEPS = [("yukleme", "Yükleme"), ("donusturme", "Demo okunuyor"), ("nisan", "Özellikler çıkarılıyor"),
                 ("gorunurluk", "Görünürlük hesaplanıyor"), ("baglam", "Kalite kontrol (QC)"),
                 ("skor", "Skorlanıyor"), ("tamam", "Tamamlandı")]


def _marker_mtime(key: str, did: str, analysis_root: Path, cache_root: Path) -> float | None:
    """Aşama işaret dosyasının zamanı (aşamanın bittiği an); yoksa None."""
    w, c = Path(analysis_root) / did, Path(cache_root) / did
    marker = dict((k, m) for k, _, m in STAGES).get(key)
    if marker is None:
        return None
    files = [c / "adapter_meta.json", w / "metadata.json"] if marker == "cache_meta" else [w / marker]
    times = [f.stat().st_mtime for f in files if f.exists()]
    return min(times) if times else None


def progress_view(did: str, job: dict | None, now: float | None = None, analysis_root: Path = ANALYSIS_ROOT,
                  cache_root: Path = CACHE_ROOT) -> dict:
    """Adım durumu (done/running/failed/pending), mevcut adım, toplam ve bu adımda geçen süre (ölçülmüş)."""
    job = job or {}
    now = time.time() if now is None else now
    inner = {k: d for k, _, d in stage_status(did, analysis_root, cache_root)}
    state = job.get("state")
    done = {"yukleme": bool(job), **inner, "tamam": state == "done"}
    if state == "done":
        done = {k: True for k in done}
    cur = next((i for i, (k, _) in enumerate(DISPLAY_STEPS) if not done[k]), None)
    steps = []
    for i, (k, label) in enumerate(DISPLAY_STEPS):
        s = "done" if done[k] else ("failed" if state == "failed" else "running") if i == cur else "pending"
        steps.append(dict(key=k, label=label, state=s))
    started = job.get("started", now)
    prev_done = [t for k, _ in DISPLAY_STEPS[:cur or 0]
                 if (t := _marker_mtime(k, did, analysis_root, cache_root)) is not None]
    stage_start = max([started] + [t for t in prev_done if t >= started])
    return dict(steps=steps, index=cur, n=len(DISPLAY_STEPS),
                current=DISPLAY_STEPS[cur][1] if cur is not None else None, current_key=DISPLAY_STEPS[cur][0]
                if cur is not None else None, elapsed_total=max(0.0, now - started),
                elapsed_stage=max(0.0, now - stage_start) if cur is not None else None)


def past_stage_seconds(key: str = "gorunurluk", analysis_root: Path = ANALYSIS_ROOT) -> list[float]:
    """BU MAKİNEDEKİ önceki analizlerin ölçülmüş aşama süreleri (metadata.json run.seconds_by_stage).
    Yalnız yerel reports/analysis/ (örnek paket başka makinede ölçüldüğü için dahil değil). Tahmin değildir."""
    out = []
    for m in sorted(Path(analysis_root).glob("*/metadata.json")):
        try:
            v = json.loads(m.read_text(encoding="utf-8")).get("run", {}).get("seconds_by_stage", {}).get(key)
        except (OSError, ValueError):
            continue
        if isinstance(v, (int, float)) and v > 0:
            out.append(float(v))
    return out


def past_stage_text(key: str = "gorunurluk", analysis_root: Path = ANALYSIS_ROOT) -> str | None:
    v = past_stage_seconds(key, analysis_root)
    if not v:
        return None
    label = dict(DISPLAY_STEPS)[key].replace(" hesaplanıyor", "").lower()
    return (f"Bu makinedeki önceki analizlerde {label} adımı {min(v) / 60:.1f}–{max(v) / 60:.1f} dk sürdü "
            f"({len(v)} analiz). Bu bir kalan süre tahmini değildir.")
