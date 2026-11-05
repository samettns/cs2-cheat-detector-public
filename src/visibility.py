"""Harita geometrisine ray-cast ile görünürlük (awpy VisibilityChecker).

.tri dosyaları oyunun kendi harita dosyalarından üretilir (awpy'nin indirme sunucusu 403 veriyor):
    Source2Viewer-CLI -i <harita>.vpk -f maps/<harita>/world_physics.vmdl_c -o <klasör>
    Source2Viewer-CLI -i world_physics.vmdl_c -b PHYS  > <harita>.vphys   (başlık satırları silinir)
    awpy tri <harita>.vphys --outpath data/maps/<harita>.tri

Doğrulama (tek maç):
    python src/visibility.py validate no_cheater_present/48
"""
import sys
from functools import lru_cache

import numpy as np
import polars as pl
from awpy.visibility import VisibilityChecker

from load import RAW_DIR, event_df, load_events

MAPS_DIR = "data/maps"

# Göz yüksekliği (ayak konumu Z'ye eklenen ofset). Source varsayılanı; CS2 için doğrulanmadı,
# doğrulama adımı bu değerleri test etmek için.
EYE_STAND = 64.0
EYE_CROUCH = 46.0
# Kurban üzerindeki hedef noktalar (ayaktan yükseklik): kafa, göğüs, bel
TARGET_HEIGHTS = {"kafa": 64.0, "gogus": 48.0, "bel": 32.0}
# Her yükseklikte merkez + 4 yan nokta (±SIDE birim; oyuncu gövdesi ~32 birim genişliğinde).
# Gün V1'de no_cheater_present/48 üzerinde seçildi (±8: %98.2, ±12: %99.4 delmeyen görünür; delenlerde %40).
# Gün V2'de DEĞİŞTİRİLMEDEN başka maç/haritalarda test edilecek.
SIDE = 12.0
OFFSETS = [(0.0, 0.0), (SIDE, 0.0), (-SIDE, 0.0), (0.0, SIDE), (0.0, -SIDE)]


# Geometrisi güvenilmez bölge çiftleri: saldırgan ve kurban bu iki bölgedeyse (hangi yönde olursa olsun)
# görünürlük "bilinmiyor" sayılır. Gün V2: anubis'te başarısızların yarıdan fazlası Bridge<->Canal hattında
# (sulu alan; su yüzeyi/korkuluk görüşü kesiyor olabilir). Kullanıcı kararı: maskele (MODEL_CARD.md).
MASKED_ZONES = {"de_anubis": [{"Bridge", "Canal"}]}


def is_masked(map_name: str, place_a: str | None, place_b: str | None) -> bool:
    return {place_a, place_b} in MASKED_ZONES.get(map_name, [])


@lru_cache(maxsize=1)   # bellek: bir harita BVH'si GB'larca (inferno ~3.9 GB) → işlem başına tek harita tutulur
def checker(map_name: str) -> VisibilityChecker:
    return VisibilityChecker(path=f"{MAPS_DIR}/{map_name}.tri")


def eye_height(duck_amount: float | None) -> float:
    """duck_amount 0 = ayakta, 1 = tam çömelik; arada doğrusal."""
    d = 0.0 if duck_amount is None else float(duck_amount)
    return EYE_STAND - d * (EYE_STAND - EYE_CROUCH)


def validate(match: str, base=RAW_DIR):
    """Duvar delmeyen isabetlerde saldırgan gözünden kurbana görünürlük oranı."""
    ev = load_events(base / match)
    map_name = ev["CSstats_info"][0]["map"]
    vc = checker(map_name)

    hits = event_df(ev, "bullet_damage").filter(pl.col("attacker_steamid").is_not_null())
    ticks = pl.read_parquet(base / f"{match}.parquet",
                            columns=["tick", "steamid", "X", "Y", "Z", "duck_amount", "last_place_name"])
    pos = {(t, s): (x, y, z, d, place) for t, s, x, y, z, d, place in ticks.iter_rows()}

    rows = []
    for h in hits.iter_rows(named=True):
        a = pos.get((h["tick"], h["attacker_steamid"]))
        v = pos.get((h["tick"], h["victim_steamid"]))
        if a is None or v is None:
            continue
        eye = (a[0], a[1], a[2] + eye_height(a[3]))
        vis = {name: any(vc.is_visible(eye, (v[0] + ox, v[1] + oy, v[2] + dz)) for ox, oy in OFFSETS)
               for name, dz in TARGET_HEIGHTS.items()}
        rows.append(dict(tick=h["tick"], pen=h["num_penetrations"], dist=h["distance"], **vis,
                         any=any(vis.values()), masked=is_masked(map_name, a[4], v[4])))
    df = pl.DataFrame(rows)

    print(f"Maç {match} ({map_name}): {df.height} isabet, {df['masked'].sum()} maskeli")
    summary = (df.filter(~pl.col("masked")).with_columns(delmeyen=pl.col("pen") == 0)
                 .group_by("delmeyen")
                 .agg(pl.len().alias("isabet"), *[pl.col(c).mean().round(3) for c in [*TARGET_HEIGHTS, "any"]])
                 .sort("delmeyen", descending=True))
    print(summary.to_pandas().to_string(index=False))
    return df


def validate_all(n_per_map: str = "1", pairs_file: str = "data/pairs.csv",
                 out: str = "data/processed/visibility_validation.csv"):
    """Her haritada çift dosyasındaki ilk n çiftin iki maçında doğrulama (parametreler sabit)."""
    pairs = pl.read_csv(pairs_file).sort("pair")
    rows = []
    for map_name in sorted(pairs["map"].unique()):
        for p in pairs.filter(pl.col("map") == map_name).head(int(n_per_map)).iter_rows(named=True):
            for match in (p["clean"], p["cheat"]):
                d = validate(match).with_columns(delmeyen=pl.col("pen") == 0)
                for delmeyen in (True, False):
                    s = d.filter(pl.col("delmeyen") == delmeyen)
                    m = s.filter(~pl.col("masked"))
                    rows.append(dict(map=map_name, match=match, delmeyen=delmeyen, isabet=s.height,
                                     gorunur=s["any"].sum(), maskeli=s["masked"].sum(),
                                     isabet_maskesiz=m.height, gorunur_maskesiz=m["any"].sum()))
    df = pl.DataFrame(rows)
    df.write_csv(out)
    print("\n== Harita başına (oran = ray-cast'in 'görünür' dediği isabet yüzdesi) ==")
    print(validation_table(df).to_pandas().to_string(index=False))


def validation_table(df: pl.DataFrame) -> pl.DataFrame:
    """validate_all sonuçlarından harita başına özet: maskesiz ve maske sonrası oranlar."""
    s = (df.group_by("map", "delmeyen")
           .agg(pl.col("isabet", "gorunur", "maskeli", "isabet_maskesiz", "gorunur_maskesiz").sum()))
    pen = s.filter(~pl.col("delmeyen")).select("map", delen_n="isabet",
                                               delen_pct=(pl.col("gorunur") / pl.col("isabet") * 100).round(1))
    return (s.filter(pl.col("delmeyen"))
             .select("map", delmeyen_n="isabet",
                     delmeyen_pct=(pl.col("gorunur") / pl.col("isabet") * 100).round(1),
                     maskeli_n="maskeli",
                     maske_sonrasi_pct=(pl.col("gorunur_maskesiz") / pl.col("isabet_maskesiz") * 100).round(1))
             .join(pen, on="map").sort("map"))


WINDOW = 128   # öldürmeden önceki tick sayısı (2 sn)
STEP = 2       # her 2 tick'te bir örnek (saniyede 32)


def view_vector(yaw, pitch):
    """CS açılarından birim bakış vektörü (pitch pozitif = aşağı bakış)."""
    y, p = np.radians(yaw), np.radians(pitch)
    return np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), -np.sin(p)])


def aim_angle(eye, yaw, pitch, target) -> float:
    """Bakış yönü ile göz→hedef yönü arasındaki açı (derece)."""
    d = np.asarray(target) - np.asarray(eye)
    d = d / np.linalg.norm(d)
    return float(np.degrees(np.arccos(np.clip(view_vector(yaw, pitch) @ d, -1.0, 1.0))))


def ray_visible(vc, eye, foot) -> bool:
    """Göz → hedef oyuncunun 15 noktasından (3 yükseklik × 5 nokta) herhangi biri görünür mü?"""
    return any(vc.is_visible(eye, (foot[0] + ox, foot[1] + oy, foot[2] + dz))
               for dz in TARGET_HEIGHTS.values() for ox, oy in OFFSETS)


def match_windows(match: str, kills: pl.DataFrame | None = None, base=RAW_DIR) -> pl.DataFrame:
    """Bir maçtaki her öldürmenin 2 sn penceresinde saldırgan → her canlı rakip:
    ray-cast görünürlük (maskeliyse null), spotted bayrağı, nişan açısı.
    kills: öldürme listesi (match, player, tick); verilmezse global kill_features'tan. base: veri klasörü."""
    ev = load_events(base / match)
    map_name = ev["CSstats_info"][0]["map"]
    vc = checker(map_name)
    if kills is None:
        kills = pl.read_parquet("data/processed/kill_features.parquet").filter(pl.col("match") == match)
    deaths = event_df(ev, "player_death").select("tick", "attacker_steamid", victim="user_steamid")
    # Bellek: maçın tamamı yerine sadece öldürme pencerelerindeki örnek tick'ler okunur. Filtre satır sırasını
    # korur → by_tick'teki oyuncu sırası (ve dolayısıyla çıktı) tam okumayla aynıdır
    # (tests/test_visibility_regression.py).
    needed = sorted({t for k in kills["tick"].to_list() for t in range(k - WINDOW, k + 1, STEP)})
    t = (pl.scan_parquet(base / f"{match}.parquet")
           .select("tick", "steamid", "X", "Y", "Z", "duck_amount", "yaw", "pitch", "is_alive",
                   "team_num", "last_place_name", "approximate_spotted_by")
           .filter(pl.col("tick").is_in(needed))
           .collect())
    by_tick = {}
    for r in t.iter_rows(named=True):
        by_tick.setdefault(r["tick"], {})[r["steamid"]] = r

    rows = []
    for kill in kills.iter_rows(named=True):
        sid = kill["player"].rsplit("_", 2)[-2] + "_" + kill["player"].rsplit("_", 1)[-1]   # "Player_N"
        k = kill["tick"]
        victim = deaths.filter((pl.col("tick") == k) & (pl.col("attacker_steamid") == sid))["victim"]
        victim = victim[0] if len(victim) else None
        for tick in range(k - WINDOW, k + 1, STEP):
            players = by_tick.get(tick, {})
            a = players.get(sid)
            if a is None or not a["is_alive"] or a["team_num"] is None:
                continue
            eye = (a["X"], a["Y"], a["Z"] + eye_height(a["duck_amount"]))
            for eid, e in players.items():
                if e["team_num"] in (None, a["team_num"]) or not e["is_alive"]:
                    continue
                masked = is_masked(map_name, a["last_place_name"], e["last_place_name"])
                foot = (e["X"], e["Y"], e["Z"])
                rows.append(dict(
                    match=match, player=kill["player"], kill_tick=k, tick=tick, enemy=eid,
                    is_victim=eid == victim,
                    visible=None if masked else ray_visible(vc, eye, foot),
                    spotted=sid in (e["approximate_spotted_by"] or []),
                    angle=aim_angle(eye, a["yaw"], a["pitch"], (foot[0], foot[1], foot[2] + 48.0)),
                ))
    return pl.DataFrame(rows)


PARTS_DIR = "data/processed/vis_parts"
HEAVY_MAPS = {"de_inferno", "de_train"}   # ölçülen BVH: inferno ~3.9 GB, train ~2.4 GB (+ maç verisi)


def match_windows_cached(match: str):
    """match_windows + maç başına kalıcı kayıt: iş yarıda kesilirse biten maçlar yeniden hesaplanmaz.
    Değerler match_windows ile aynıdır (sadece saklama şekli farklı)."""
    from pathlib import Path

    d = Path(PARTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    f, empty = d / (match.replace("/", "__") + ".parquet"), d / (match.replace("/", "__") + ".empty")
    if f.exists():
        return match, pl.read_parquet(f)
    if empty.exists():
        return match, pl.DataFrame()
    part = match_windows(match)
    if part.height:
        part.write_parquet(f)
    else:
        empty.touch()
    return match, part


def batch_windows(n_proc: str = "4", pairs_file: str = "data/pairs_new.csv", limit: str = "55"):
    """Parti: çift dosyasındaki, henüz tabloda ve vis_parts'ta olmayan ilk `limit` maçı hesaplar ve maç başına kaydeder.
    Ana tabloya (visibility.parquet) DOKUNMAZ; hepsi bitince `windows ... append` birleştirir."""
    from multiprocessing import Pool
    from pathlib import Path

    pairs = pl.read_csv(pairs_file)
    matches = sorted(pairs["cheat"].to_list() + pairs["clean"].to_list())
    in_table = set(pl.read_parquet("data/processed/visibility.parquet", columns=["match"])["match"].unique().to_list())
    def cached(m):
        b = Path(PARTS_DIR) / m.replace("/", "__")
        return b.with_suffix(".parquet").exists() or b.with_suffix(".empty").exists()
    todo = [m for m in matches if m not in in_table and not cached(m)]
    # Haritaya göre sırala: aynı haritanın maçları art arda işlenir, geometri (tek harita önbelleği) tekrar yüklenmez
    map_of = dict(zip(pairs["cheat"], pairs["map"])) | dict(zip(pairs["clean"], pairs["map"]))
    todo.sort(key=lambda m: (map_of[m], m))
    done_before = len(matches) - len(todo)
    batch = todo[: int(limit)]
    print(f"toplam {len(matches)} maç | önceden biten {done_before} | bu parti {len(batch)} | parti sonrası kalan "
          f"{len(todo) - len(batch)}", flush=True)
    # Ağır haritalarda (BVH işlem başına ~4–5 GB) tek işlem; diğerlerinde n_proc (16 GB RAM'de 2 ağır işlem sığmıyor)
    i = 0
    for map_name in sorted({map_of[m] for m in batch}):
        group = [m for m in batch if map_of[m] == map_name]
        n = 1 if map_name in HEAVY_MAPS else int(n_proc)
        print(f"-- {map_name}: {len(group)} maç, {n} işlem", flush=True)
        with Pool(n) as pool:
            for m, part in pool.imap_unordered(match_windows_cached, group):
                i += 1
                print(f"[{i}/{len(batch)}] {m}: {part.height} satır", flush=True)
    print(f"PARTI BITTI: {done_before + len(batch)}/{len(matches)} maç tamam", flush=True)


def compute_windows(n_proc: str = "4", pairs_file: str = "data/pairs.csv", mode: str = "replace"):
    """Çift dosyasındaki maçların öldürme pencereleri → data/processed/visibility.parquet (paralel).

    mode="append": mevcut tabloda olmayan maçlar hesaplanıp eklenir (mevcut satırlara dokunulmaz).
    """
    from multiprocessing import Pool
    from pathlib import Path

    out = "data/processed/visibility.parquet"
    pairs = pl.read_csv(pairs_file)
    matches = sorted(pairs["cheat"].to_list() + pairs["clean"].to_list())
    existing = None
    if mode == "append" and Path(out).exists():
        existing = pl.read_parquet(out)
        done = set(existing["match"].unique().to_list())
        matches = [m for m in matches if m not in done]
        print(f"ekleme modu: {len(done)} maç zaten var, {len(matches)} maç hesaplanacak", flush=True)
    with Pool(int(n_proc)) as pool:
        parts = []
        for i, (match, part) in enumerate(pool.imap_unordered(match_windows_cached, matches), 1):
            parts.append(part)
            print(f"[{i}/{len(matches)}] {match}: {part.height} satır", flush=True)
    df = pl.concat(([existing] if existing is not None else []) + [p for p in parts if p.height])
    df = df.sort("match", "player", "kill_tick", "tick", "enemy")
    df.write_parquet(out)
    print(f"Toplam {df.height} satır, {df.select('match', 'player', 'kill_tick').n_unique()} öldürme "
          f"-> data/processed/visibility.parquet")


def compare_spotted():
    """spotted bayrağı vs ray-cast görünürlük, silah sınıfına göre (data/processed/visibility.parquet)."""
    from baseline import load

    vis = pl.read_parquet("data/processed/visibility.parquet").drop_nulls("visible")   # maskeliler hariç
    kills = load().select("match", "player", pl.col("tick").alias("kill_tick"), "weapon_class", "group")
    vis = vis.join(kills, on=["match", "player", "kill_tick"])   # "belirsiz" grup load() içinde zaten dışarıda

    def table(d, title):
        t = (d.group_by("weapon_class")
              .agg(pl.len().alias("n"),
                   pl.col("visible").mean().alias("raycast_gorunur"),
                   pl.col("spotted").mean().alias("spotted"),
                   pl.col("spotted").filter(pl.col("visible")).mean().alias("P(spotted|gorunur)"),
                   pl.col("spotted").filter(~pl.col("visible")).mean().alias("P(spotted|gorunmez)"),
                   (pl.col("spotted") == pl.col("visible")).mean().alias("uyum"))
              .sort("weapon_class"))
        print(f"\n== {title} ==")
        print(t.with_columns(pl.exclude("weapon_class", "n").round(3)).to_pandas().to_string(index=False))

    # (1) Kurban, öldürmeden önceki son örnek anda (gerçekten görünür olması beklenen durum)
    last = vis.filter(pl.col("is_victim")).sort("tick").group_by("match", "player", "kill_tick").agg(pl.all().last())
    table(last, f"Kurban, öldürmeden hemen önce ({last.height} öldürme)")
    # (2) Tüm pencere, tüm rakipler
    table(vis, f"Tüm pencere, tüm rakipler ({vis.height} satır)")
    # (3) Grup farkı: bayrağın hatası temiz/hileci arasında farklı mı? (model bunu kısayol olarak kullanabilir)
    print("\n== Kurban son an, gruba göre ==")
    print(last.group_by("group").agg(pl.len(), pl.col("visible").mean().round(3),
                                     pl.col("spotted").filter(pl.col("visible")).mean().round(3).alias("P(spotted|gorunur)"))
              .sort("group").to_pandas().to_string(index=False))


def build_tri(map_name: str):
    """Oyunun harita VPK'sından data/maps/<harita>.tri üretir (Source2Viewer-CLI + awpy tri)."""
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import radar  # yol çözümü tek yerde: ortam değişkeni → otomatik keşif → genel yedek

    vrf = str(radar.vrf_path())
    maps = os.environ.get("CS2_MAPS", str(radar.csgo_dir() / "maps"))
    out = Path(MAPS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Fizik modelini VPK'dan ham olarak çıkar
        subprocess.run([vrf, "-i", f"{maps}/{map_name}.vpk", "-f", f"maps/{map_name}/world_physics.vmdl_c",
                        "-o", tmp], check=True, capture_output=True)
        model = next(Path(tmp).rglob("world_physics.vmdl_c"))
        # 2) PHYS bloğunu KV3 metin olarak dök; baştaki VRF başlık satırlarını at
        dump = subprocess.run([vrf, "-i", str(model), "-b", "PHYS"], check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout
        vphys = out / f"{map_name}.vphys"
        vphys.write_text(dump[dump.index("<!-- kv3"):], encoding="utf-8")
    # 3) awpy'nin gizli `tri` komutu: KV3 → üçgenler
    subprocess.run([str(Path(sys.executable).parent / "awpy.exe"), "tri", str(vphys),
                    "--outpath", str(out / f"{map_name}.tri")], check=True, capture_output=True)
    print(f"{map_name}: {(out / f'{map_name}.tri').stat().st_size / 1e6:.1f} MB .tri")


if __name__ == "__main__":
    {"validate": validate, "validate_all": validate_all, "build": build_tri,
     "windows": compute_windows, "batch": batch_windows, "compare_spotted": compare_spotted}[sys.argv[1]](*sys.argv[2:])
