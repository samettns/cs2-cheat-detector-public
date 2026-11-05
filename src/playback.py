"""Faz 3.5 round oynatma PROTOTİPİ (dal faz35/playback; ana app.py'ye bağlı DEĞİL).

Plotly animasyon kareleri: oynatma tamamen tarayıcıda (Streamlit yeniden çalıştırma yok). Kare = örneklenen tick'te
canlı oyuncuların konumu + kısa iz + o anda etkin duman/molotof. Model skoru / olay skoru / seçili olay KULLANILMAZ.
Oyuncu etiketleri anonim Player_N (mevcut adlandırma). Figür düz dict olarak kurulur (Plotly nesne doğrulaması
kare başına çok yavaş: 920 karede ~8 sn).

    python src/playback.py rounds <analiz_klasörü>      → round uzunlukları (kısa / medyan / uzun)
    python src/playback.py size <analiz_klasörü> <round> [hz] [trail_s]   → kare sayısı, JSON boyutu
"""
import base64
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

import radar as R
import viewer_data as V

TICKRATE = 64
CONTEXT_HZ = 8                                   # map_context örnekleme hızı (8 tick'te bir)
PLAYBACK_HZ_DEFAULT = 8                          # ölçüm: 8 Hz + 1 sn iz bütün round'larda ölçütü sağladı
PLAYBACK_TRAIL_S = 1.0
LONG_ROUND_FRAMES = 1000                         # 8 Hz'de > 1000 kare → otomatik 4 Hz (güvenli oynatma)
SLIDER_LABELS = 12                               # kaydırıcıda en çok ~12 süre etiketi
SPEEDS = [0.5, 1, 2, 4]
T_COL, CT_COL = "#f5a524", "#3b8bff"            # takım renkleri (T turuncu, CT mavi)
SMOKE_FILL, FIRE_FILL = "rgba(200,200,210,0.35)", "rgba(235,104,52,0.35)"
CIRCLE_N = 24                                    # duman/molotof çokgeni köşe sayısı


def playback_hz(frames_8hz: int | None = None) -> int:
    """Varsayılan 8 Hz; CS2_PLAYBACK_HZ=4 (yavaş makine) ya da round 8 Hz'de LONG_ROUND_FRAMES kareden uzunsa 4 Hz.
    Yalnız gösterim/performans (ölçüm: Nuke 1174 karelik round'da 8 Hz ilk çizim 2.06–2.10 sn > 2 sn ölçütü)."""
    v = os.environ.get("CS2_PLAYBACK_HZ", "").strip()
    if v == "4" or (frames_8hz is not None and frames_8hz > LONG_ROUND_FRAMES):
        return 4
    return PLAYBACK_HZ_DEFAULT


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def round_table(work: Path) -> pl.DataFrame:
    """Canlı (live) aşamadaki round'lar: tick aralığı, süre (sn), 8 Hz kare sayısı."""
    c = pl.read_parquet(Path(work) / "map_context.parquet", columns=["tick", "round", "phase"])
    return (c.filter(pl.col("phase") == "live").group_by("round")
             .agg(start=pl.col("tick").min(), end=pl.col("tick").max(), frames_8hz=pl.col("tick").n_unique())
             .with_columns(seconds=((pl.col("end") - pl.col("start")) / TICKRATE).round(1)).sort("round"))


def _utility_intervals(u: pl.DataFrame) -> list[tuple]:
    """Duman/molotof: (başlangıç, bitiş, x, y, tür, z) — viewer_data.utility_intervals ile AYNI eşleme."""
    return [(r["tick"], r["end_tick"], r["x"], r["y"], r["kind"], r["z"])
            for r in V.utility_intervals(u).iter_rows(named=True)]


def build_playback_figure(work: Path, rnd: int, hz: int | None = None, trail_s: float = PLAYBACK_TRAIL_S,
                          with_image: bool = True, height: int = 720, level: str | None = None) -> tuple[dict, dict]:
    """Seçili round için animasyonlu figür (düz dict) ve teknik bilgi (round süresi, kare sayısı, Hz, iz)."""
    work = Path(work)
    meta = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    info = V.radar_info(work, meta["map"])
    # çok katlı harita — seçili katın görseli; diğer kattakiler ayrı soluk izlerde (7–10). Kat ataması histerezisli.
    lvl_on = level is not None and V.level_enabled(info, meta["map"])
    level = level if lvl_on else "default"
    ctx = (pl.read_parquet(work / "map_context.parquet",
                           columns=["tick", "player", "X", "Y", "Z", "is_alive", "team_num", "round", "phase"])
             .filter((pl.col("round") == rnd) & (pl.col("phase") == "live")).sort("player", "tick"))
    ticks_all = ctx["tick"].unique().sort().to_list()
    hz = hz or playback_hz(len(ticks_all))
    t_start = ticks_all[0] if ticks_all else 0
    ticks = ticks_all[::max(1, CONTEXT_HZ // hz)]
    trail_ticks = int(trail_s * TICKRATE)
    px, py = R.world_to_pixel(info, ctx["X"].to_numpy(), ctx["Y"].to_numpy())
    ctx = ctx.with_columns(px=pl.Series(np.round(px, 1)), py=pl.Series(np.round(py, 1))).filter(
        pl.col("tick").is_in(ticks) & pl.col("is_alive").fill_null(False))
    # oyuncu başına sıralı (tick, x, y, kat): iz penceresi ikili aramayla bulunur
    per = {name: (team, g["tick"].to_numpy(), g["px"].to_list(), g["py"].to_list(),
                  V.assign_levels(g["Z"].to_list(), V.level_boundary(info)) if lvl_on else ["default"] * g.height)
           for (name, team), g in ctx.group_by("player", "team_num", maintain_order=True)}
    other_lbl = V.LEVEL_LABEL["lower" if level == "default" else "default"].lower()
    util = _utility_intervals(pl.read_parquet(work / "utility.parquet"))
    ang = np.linspace(0, 2 * math.pi, CIRCLE_N + 1)
    circ = {}

    def poly(x, y, kind):
        if (x, y, kind) not in circ:
            cx, cy = R.world_to_pixel(info, x, y)
            r = (V.SMOKE_RADIUS if kind == "smoke" else V.INFERNO_RADIUS) / info["scale"]
            circ[(x, y, kind)] = ([round(float(v), 1) for v in cx + r * np.cos(ang)] + [None],
                                  [round(float(v), 1) for v in cy + r * np.sin(ang)] + [None])
        return circ[(x, y, kind)]

    def frame_data(i: int, t: int) -> list[dict]:
        tr = {2: ([], []), 3: ([], [])}
        now = {2: ([], [], []), 3: ([], [], [])}
        alive = []                                   # (takım, x, y, ad) — etiketler iki takımda birlikte sadeleşir
        o_tr, o_now = ([], []), {2: ([], [], []), 3: ([], [], [])}     # diğer kat: iz + oyuncular
        for name, (team, tk, xs, ys, lv) in per.items():
            if team not in tr:
                continue
            hi = int(np.searchsorted(tk, t, side="right"))
            if hi == 0 or tk[hi - 1] != t:          # bu tick'te canlı değil
                continue
            lo = int(np.searchsorted(tk, t - trail_ticks, side="right"))
            for j in range(lo, hi):                   # iz noktaları kendi katına göre ayrılır
                dst = tr[team] if lv[j] == level else o_tr
                if j > lo and lv[j - 1] != lv[j]:
                    dst[0].append(None)
                    dst[1].append(None)
                dst[0].append(xs[j])
                dst[1].append(ys[j])
            for d_ in (tr[team], o_tr):
                d_[0].append(None)
                d_[1].append(None)
            if lv[hi - 1] == level:
                alive.append((team, xs[hi - 1], ys[hi - 1], name))
            else:
                o_now[team][0].append(xs[hi - 1])
                o_now[team][1].append(ys[hi - 1])
                o_now[team][2].append(name)
        texts, _ = V.declutter_labels([a[1] for a in alive], [a[2] for a in alive], [a[3] for a in alive])
        for (team, x, y, name), txt in zip(alive, texts):
            now[team][0].append(x)
            now[team][1].append(y)
            now[team][2].append((txt or " ", name))      # " ": redraw=False'da metin öğesi hep var olsun
        out = [dict(type="scatter", x=tr[k][0], y=tr[k][1]) for k in (2, 3)]
        out += [dict(type="scatter", x=now[k][0], y=now[k][1], text=[a for a, _ in now[k][2]],
                     hovertext=[b for _, b in now[k][2]]) for k in (2, 3)]
        ou_x, ou_y = [], []
        for kind in ("smoke", "molotof"):
            xs, ys = [], []
            for a, b, x, y, kd, uz in util:
                if kd == kind and a <= t < b:
                    cx, cy = poly(x, y, kind)
                    if lvl_on and ("lower" if (uz or 0) < V.level_boundary(info) else "default") != level:
                        ou_x += cx
                        ou_y += cy
                    else:
                        xs += cx
                        ys += cy
            out.append(dict(type="scatter", x=xs, y=ys))
        out.append(dict(type="scatter", x=[14], y=[26], text=[f"{mmss(i / hz)} · tick {t}"]))
        out.append(dict(type="scatter", x=o_tr[0], y=o_tr[1]))                                      # 7: diğer kat izleri
        out += [dict(type="scatter", x=o_now[k][0], y=o_now[k][1],
                     hovertext=[f"{n} ({other_lbl})" for n in o_now[k][2]]) for k in (2, 3)]        # 8–9: diğer kat
        out.append(dict(type="scatter", x=ou_x, y=ou_y))                                           # 10: diğer kat duman/molotof
        return out

    frames = [dict(name=str(t), data=frame_data(i, t)) for i, t in enumerate(ticks)]
    style = ([dict(mode="lines", line=dict(color=c, width=2), opacity=0.55, hoverinfo="skip", showlegend=False)
              for c in (T_COL, CT_COL)]
             + [dict(mode="markers+text", textposition="top center", textfont=dict(color="#e6edf3", size=10), name=n,
                     marker=dict(size=10, color=c, line=dict(color="white", width=1)),
                     hovertemplate="%{hovertext}<extra></extra>")
                for c, n in ((T_COL, "T"), (CT_COL, "CT"))]
             + [dict(mode="lines", fill="toself", fillcolor=f, line=dict(width=0), hoverinfo="skip", name=n)
                for f, n in ((SMOKE_FILL, "duman"), (FIRE_FILL, "molotof"))]
             + [dict(mode="text", textposition="bottom right", textfont=dict(color="#e6edf3", size=13),
                     hoverinfo="skip", showlegend=False)]
             + [dict(mode="lines", line=dict(color="#9aa4b1", width=1, dash="dot"), opacity=0.35, hoverinfo="skip",
                     showlegend=False)]
             + [dict(mode="markers", opacity=0.5, name=f"{n} · {other_lbl} (içi boş)", showlegend=lvl_on,
                     marker=dict(size=10, color=c, symbol=sym, line=dict(width=1.5)),
                     hovertemplate="%{hovertext}<extra></extra>")
                for c, n, sym in ((T_COL, "T", "circle-open"), (CT_COL, "CT", "diamond-open"))]
             + [dict(mode="lines", line=dict(color="rgba(220,220,230,0.45)", width=1, dash="dot"), hoverinfo="skip",
                     showlegend=False)])
    data = [s_ | d for s_, d in zip(style, frames[0]["data"])] if frames else []
    images = []
    img_name = next(lv["image"] for lv in info["levels"] if lv["name"] == level) if lvl_on else info["levels"][0]["image"]
    if with_image and img_name:
        b64 = base64.b64encode((R.RADAR_DIR / img_name).read_bytes()).decode()
        images = [dict(source="data:image/png;base64," + b64, x=0, y=0, sizex=info["size"], sizey=info["size"],
                       xref="x", yref="y", sizing="stretch", layer="below", opacity=0.92)]
    dur = 1000 / hz

    def anim(speed):
        return dict(label=f"{speed:g}×", method="animate",
                    args=[None, dict(frame=dict(duration=dur / speed, redraw=False), transition=dict(duration=0),
                                     fromcurrent=True, mode="immediate")])

    step_args = dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))
    span_s = len(ticks) / hz
    label_s = next((v for v in (5, 10, 15, 20, 30, 60) if span_s / v <= SLIDER_LABELS), 120)
    label_every = max(1, label_s * hz)                     # yuvarlak süre aralığı: 0:00, 0:10, 0:20 …
    layout = dict(
        height=height, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="#0b0f14", paper_bgcolor="#0b0f14", images=images,
        xaxis=dict(range=[0, info["size"]], visible=False),
        yaxis=dict(range=[info["size"], 0], visible=False, scaleanchor="x"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, font=dict(color="#c9d1d9")),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                          pad=dict(t=4, r=6), bgcolor="#151b23", font=dict(color="#e6edf3"),
                          buttons=[dict(label="⏸", method="animate", args=[[None], step_args])]
                          + [anim(sp) for sp in SPEEDS])],
        sliders=[dict(active=0, x=0.0, len=1.0, y=0, pad=dict(t=30), currentvalue=dict(visible=False),
                      font=dict(color="#8b949e", size=11), ticklen=3, minorticklen=0,
                      steps=[dict(method="animate", label=mmss(i / hz) if i % label_every == 0 else "",
                                  args=[[str(t)], step_args]) for i, t in enumerate(ticks)])])
    tech = dict(round=rnd, seconds=round((ticks_all[-1] - ticks_all[0]) / TICKRATE, 1) if ticks_all else 0,
                frames=len(frames), hz=hz, trail_s=trail_s, level=level if lvl_on else None,
                safe_mode=hz < PLAYBACK_HZ_DEFAULT and len(ticks_all) > LONG_ROUND_FRAMES)
    return dict(data=data, layout=layout, frames=frames), tech


def json_sizes(fig: dict) -> tuple[float, float]:
    """Figür JSON boyutu (MB): ham ve gzip."""
    raw = json.dumps(fig, separators=(",", ":")).encode("utf-8")
    return round(len(raw) / 1e6, 2), round(len(gzip.compress(raw)) / 1e6, 2)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["rounds"]:
        print(round_table(Path(a[1])))
    elif a[:1] == ["size"]:
        f, t = build_playback_figure(Path(a[1]), int(a[2]), int(a[3]) if len(a) > 3 else 8,
                                     float(a[4]) if len(a) > 4 else 2.0)
        print(t, json_sizes(f))
    else:
        print(__doc__)
