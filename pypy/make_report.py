"""Generate a self-contained HTML benchmark report (report.html).

Embeds the measured numbers and draws the p(sentinel) sweep as an inline SVG
line chart -- no external assets, so the file works anywhere it's copied.
"""

# --- measured data (M elem/s for sums, M calls/s for polysum) -----------------

ENGINES = ["CPython 3.13", "PyPy 3.11", "V8 13.6 / Node 24", "C (gcc -O2)"]

HEADLINE = [
    # label, unit, [cpython, pypy, v8, c]
    ("plain sum (10M ints)", "M elem/s", [55, 1665, 1274, 3846]),
    ("sentinel sum, 10% skipped", "M elem/s", [44, 660, 357, 2353]),
    ("polysum, monomorphic", "M calls/s", [33, 330, 597, 698]),
    ("polysum, polymorphic (3-way)", "M calls/s", [24, 117, 156, 181]),
]

# sentinel sweep across p(sentinel), each point in its own process (M elem/s)
SWEEP_P = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
SWEEP = {
    "CPython 3.13": [44.0, 44.3, 44.9, 45.0, 45.8, 47.1, 51.5, 58.9, 68.3, 80.9, 89.3],
    "PyPy 3.11": [1006.7, 664.5, 420.8, 314.4, 249.3, 217.2, 227.7, 284.2, 400.4, 664.6, 992.8],
    "V8 13.6 / Node 24": [407.9, 361.1, 294.9, 254.5, 228.3, 217.7, 229.5, 258.2, 339.6, 516.7, 736.9],
    "C (gcc -O2)": [2275.8, 2309.4, 2315.6, 2309.0, 2309.7, 2303.7, 2293.3, 2314.1, 2160.2, 2305.1, 2302.7],
}

COLORS = {
    "CPython 3.13": "#1f77b4",
    "PyPy 3.11": "#d62728",
    "V8 13.6 / Node 24": "#2ca02c",
    "C (gcc -O2)": "#9467bd",
}

# --- SVG chart ----------------------------------------------------------------

import math

W, H = 900, 470
M = {"l": 58, "r": 188, "t": 30, "b": 80}
PLOT_H = H - M["t"] - M["b"]

# Left band: the non-sweep "baseline" benchmarks, drawn as disconnected
# per-engine markers (squares) so they share the log throughput axis with the
# sweep without being joined to its lines.
# label line 1, label line 2, [cpython, pypy, v8, c]
COLUMNS = [
    ("uncond.", "sum", [55, 1665, 1274, 3846]),   # plain sum (10M ints)
    ("vtable", "mono", [33, 330, 597, 698]),       # polysum, monomorphic
    ("vtable", "poly", [24, 117, 156, 181]),       # polysum, polymorphic (3-way)
]
COL_X = [M["l"] + 34 + 46 * i for i in range(len(COLUMNS))]  # column centres
DIV_X = COL_X[-1] + 38            # divider between the band and the sweep
SWEEP_X0 = DIV_X + 14
SWEEP_X1 = W - M["r"]
SWEEP_W = SWEEP_X1 - SWEEP_X0

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 20.0, 5000.0
Y_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000]
_LOGMIN, _LOGMAX = math.log10(Y_MIN), math.log10(Y_MAX)


def x_px(p):
    return SWEEP_X0 + (p - X_MIN) / (X_MAX - X_MIN) * SWEEP_W


def y_px(v):
    return M["t"] + (1 - (math.log10(v) - _LOGMIN) / (_LOGMAX - _LOGMIN)) * PLOT_H


def svg_chart():
    y_bot = M["t"] + PLOT_H
    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             'role="img" aria-label="loop-benchmark throughput (log scale)">']

    # y gridlines + labels (span the baseline band and the sweep)
    for v in Y_TICKS:
        y = y_px(v)
        parts.append(f'<line x1="{M["l"]}" y1="{y:.1f}" x2="{SWEEP_X1}" '
                     f'y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{M["l"]-10}" y="{y+4:.1f}" class="ytick">{v:,}</text>')

    # section headers
    bandc = (COL_X[0] + COL_X[-1]) / 2
    parts.append(f'<text x="{bandc:.1f}" y="{M["t"]-14}" class="axislabel">baselines</text>')
    parts.append(f'<text x="{(SWEEP_X0+SWEEP_X1)/2:.1f}" y="{M["t"]-14}" '
                 'class="axislabel">sentinel sweep</text>')

    # divider between the two regions
    parts.append(f'<line x1="{DIV_X}" y1="{M["t"]}" x2="{DIV_X}" '
                 f'y2="{y_bot}" stroke="#8884" stroke-width="1"/>')

    # baseline columns: one square per engine, disconnected (no lines)
    for (l1, l2, vals), cx in zip(COLUMNS, COL_X):
        for i, name in enumerate(ENGINES):
            x = cx + (i - 1.5) * 7
            y = y_px(vals[i])
            parts.append(f'<rect x="{x-3:.1f}" y="{y-3:.1f}" width="6" height="6" '
                         f'fill="{COLORS[name]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y_bot+16}" class="xtick">{l1}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y_bot+30}" class="xtick">{l2}</text>')

    # sweep x ticks + labels
    for p in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        x = x_px(p)
        parts.append(f'<line x1="{x:.1f}" y1="{y_bot}" x2="{x:.1f}" '
                     f'y2="{y_bot+5}" class="axis"/>')
        parts.append(f'<text x="{x:.1f}" y="{y_bot+18}" class="xtick">{p:g}</text>')

    # axes
    parts.append(f'<line x1="{M["l"]}" y1="{y_bot}" x2="{SWEEP_X1}" '
                 f'y2="{y_bot}" class="axis"/>')
    parts.append(f'<line x1="{M["l"]}" y1="{M["t"]}" x2="{M["l"]}" '
                 f'y2="{y_bot}" class="axis"/>')

    # axis titles
    parts.append(f'<text x="{(SWEEP_X0+SWEEP_X1)/2:.1f}" y="{y_bot+44}" '
                 'class="axislabel">p(sentinel)</text>')
    parts.append(f'<text transform="translate(16,{M["t"]+PLOT_H/2:.1f}) rotate(-90)" '
                 'class="axislabel">throughput (M ops/s, log)</text>')

    # marker at p=0.5 (the branch-prediction minimum)
    xmid = x_px(0.5)
    parts.append(f'<line x1="{xmid:.1f}" y1="{M["t"]}" x2="{xmid:.1f}" '
                 f'y2="{y_bot}" class="mid"/>')

    # sweep data lines + points
    for name in ENGINES:
        color = COLORS[name]
        pts = " ".join(f"{x_px(p):.1f},{y_px(v):.1f}"
                       for p, v in zip(SWEEP_P, SWEEP[name]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     'stroke-width="2.5"/>')
        for p, v in zip(SWEEP_P, SWEEP[name]):
            parts.append(f'<circle cx="{x_px(p):.1f}" cy="{y_px(v):.1f}" r="3" '
                         f'fill="{color}"/>')

    # legend: engines, then a marker-shape key
    lx = SWEEP_X1 + 20
    legend_y = M["t"] + 6
    for name in ENGINES:
        color = COLORS[name]
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+22}" '
                     f'y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{lx+11}" cy="{legend_y}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{lx+28}" y="{legend_y+4}" class="legend">{name}</text>')
        legend_y += 22
    legend_y += 10
    parts.append(f'<circle cx="{lx+11}" cy="{legend_y}" r="3" fill="#888"/>')
    parts.append(f'<text x="{lx+28}" y="{legend_y+4}" class="legend">sweep (vs p)</text>')
    legend_y += 20
    parts.append(f'<rect x="{lx+8}" y="{legend_y-3}" width="6" height="6" fill="#888"/>')
    parts.append(f'<text x="{lx+28}" y="{legend_y+4}" class="legend">baseline</text>')

    parts.append("</svg>")
    return "\n".join(parts)


# --- HTML ---------------------------------------------------------------------

def headline_table():
    rows = []
    for label, unit, vals in HEADLINE:
        best = max(vals)
        cells = "".join(
            f'<td class="num{" best" if val == best else ""}">{val:,}</td>'
            for val in vals
        )
        rows.append(f"<tr><td>{label}</td>{cells}<td class='unit'>{unit}</td></tr>")
    return "\n".join(rows)


def sweep_table():
    rows = []
    for i, p in enumerate(SWEEP_P):
        vals = [SWEEP[name][i] for name in ENGINES]
        cells = "".join(f'<td class="num">{v:.0f}</td>' for v in vals)
        cls = ' class="midrow"' if p == 0.50 else ""
        rows.append(f"<tr{cls}><td class='num'>{p:.2f}</td>{cells}</tr>")
    return "\n".join(rows)


HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPython vs PyPy vs V8 — loop benchmarks</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.55 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         max-width: 880px; margin: 2.2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.25rem; margin-top: 2.2rem; border-bottom: 1px solid #8884;
        padding-bottom: .2rem; }}
  .sub {{ color: #888; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .95rem; }}
  th, td {{ padding: .4rem .6rem; text-align: left; border-bottom: 1px solid #8883; }}
  th {{ font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.unit {{ color: #888; font-size: .85rem; }}
  td.best {{ font-weight: 700; }}
  tr.midrow {{ background: #ffd70022; }}
  .chart {{ margin: 1rem 0; overflow-x: auto; }}
  svg text {{ fill: currentColor; }}
  .grid {{ stroke: #8882; stroke-width: 1; }}
  .axis {{ stroke: currentColor; stroke-width: 1.2; }}
  .mid {{ stroke: #f5a; stroke-width: 1; stroke-dasharray: 4 3; opacity: .7; }}
  .ytick, .xtick {{ font-size: 12px; fill: #888; }}
  .ytick {{ text-anchor: end; }}
  .xtick {{ text-anchor: middle; }}
  .axislabel {{ font-size: 13px; text-anchor: middle; fill: #888; }}
  .legend {{ font-size: 13px; }}
  .note {{ color: #888; font-size: .9rem; }}
  code {{ background: #8881; padding: .1em .35em; border-radius: 4px; }}
</style>
</head>
<body>
<h1>CPython vs PyPy vs V8 vs C: pure-language loop benchmarks</h1>
<p class="sub">Equivalent integer summing loops, written idiomatically per
language. One machine, x86_64; best of 5 reps after a warm-up. Rough numbers —
large effects only.</p>

<h2>Headline results</h2>
<table>
<thead><tr><th>Benchmark</th><th class="num">CPython&nbsp;3.13</th>
<th class="num">PyPy&nbsp;3.11</th><th class="num">V8&nbsp;13.6&nbsp;/&nbsp;Node&nbsp;24</th>
<th class="num">C&nbsp;(gcc&nbsp;-O2)</th><th>unit</th></tr></thead>
<tbody>
{headline_table()}
</tbody>
</table>
<p class="note">Throughput; higher is better. <strong>Bold</strong> = fastest
engine for that row. Everything is <code>int</code> now, so the rows are
directly comparable. C (<code>-O2</code>, no SIMD) leads everywhere; the int
loops are fast because int-add is 1-cycle latency and gcc if-converts the
sentinel skip to a branchless <code>cmov</code>. Among the JITs, PyPy wins the
tight numeric loops and V8 wins object method dispatch. (Building C at
<code>-O3</code> lets it auto-vectorize: plain ~5490, sentinel ~4250 M elem/s.)</p>

<h2>sentinel sum throughput vs p(sentinel)</h2>
<p>Summing 10M ints while skipping a <code>-1</code> sentinel
(<code>if v == -1: continue</code>), sweeping the fraction that are sentinels.
Each point is measured in its own process so each engine's JIT compiles fresh
for that fraction. Two effects compete: a <strong>branch-misprediction</strong>
penalty that peaks at p&nbsp;=&nbsp;0.5 (the skip is maximally unpredictable),
and a <strong>less-work</strong> effect as more skips mean fewer adds — and each
engine weights them differently. The y-axis is <strong>log-scaled</strong>; the
three columns at left place the non-sweep benchmarks — the unconditional
&ldquo;plain&rdquo; sum and the two vtable (<code>polysum</code>) dispatch
benchmarks — as disconnected per-engine squares for scale (polysum is
M&nbsp;calls/s, not elem/s).</p>
<div class="chart">
{svg_chart()}
</div>
<ul>
<li><strong>C is dead flat</strong> (~2300) — the <code>cmov</code> runs every
iteration regardless of the data, so there's no branch to mispredict and no
work saved by skipping. The optimizer erased the whole effect.</li>
<li><strong>PyPy traces a symmetric U</strong> (~1007 → ~217 → ~993): it emits a
real guard, so misprediction dominates and bottoms out at p&nbsp;=&nbsp;0.5.</li>
<li><strong>V8 is a tilted U</strong> — a dip at ~0.5 from misprediction, but the
high-skip end runs faster than the low-skip end because it does fewer adds.</li>
<li><strong>CPython rises monotonically</strong> (~44 → ~89): interpreter
overhead dwarfs the branch, so the only visible effect is less work per skip.</li>
</ul>

<table>
<thead><tr><th class="num">p(sentinel)</th>
<th class="num">CPython</th><th class="num">PyPy</th><th class="num">V8</th>
<th class="num">C</th></tr></thead>
<tbody>
{sweep_table()}
</tbody>
</table>
<p class="note">M elements/s. Row at p=0.5 highlighted.</p>
</body>
</html>
"""


def main():
    with open("report.html", "w") as f:
        f.write(HTML)
    print("wrote report.html")


if __name__ == "__main__":
    main()
