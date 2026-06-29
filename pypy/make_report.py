"""Generate a self-contained HTML benchmark report (report.html).

Embeds the measured numbers and draws the p(NaN) sweep as an inline SVG line
chart -- no external assets, so the file works anywhere it's copied.
"""

# --- measured data (M elem/s for sums, M calls/s for polysum) -----------------

ENGINES = ["CPython 3.13", "PyPy 3.11", "V8 13.6 / Node 24", "C (gcc -O2)"]

HEADLINE = [
    # label, unit, [cpython, pypy, v8, c]
    ("plain sum (10M floats)", "M elem/s", [85, 1870, 1706, 2237]),
    ("nansum, 10% random NaN", "M elem/s", [52, 536, 275, 748]),
    ("polysum, monomorphic", "M calls/s", [33, 334, 618, 680]),
    ("polysum, polymorphic (3-way)", "M calls/s", [24, 117, 156, 182]),
]

# nansum sweep, each point measured in its own process (M elem/s)
SWEEP_P = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
SWEEP = {
    "CPython 3.13": [56.3, 57.6, 55.6, 53.2, 44.0, 51.2, 53.6, 56.8, 56.4, 62.7, 65.5],
    "PyPy 3.11": [768.1, 545.0, 345.0, 255.4, 203.0, 181.3, 190.8, 254.6, 344.8, 556.0, 820.8],
    "V8 13.6 / Node 24": [311.0, 274.6, 229.9, 202.2, 178.1, 165.5, 175.6, 201.0, 239.7, 316.0, 384.5],
    "C (gcc -O2)": [1199.0, 753.9, 476.5, 372.0, 321.7, 294.5, 318.4, 366.0, 456.9, 705.2, 1122.4],
}

COLORS = {
    "CPython 3.13": "#1f77b4",
    "PyPy 3.11": "#d62728",
    "V8 13.6 / Node 24": "#2ca02c",
    "C (gcc -O2)": "#9467bd",
}

# --- SVG line chart -----------------------------------------------------------

W, H = 760, 460
M = {"l": 70, "r": 170, "t": 30, "b": 60}
PLOT_W = W - M["l"] - M["r"]
PLOT_H = H - M["t"] - M["b"]
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1200.0
Y_STEP = 200


def x_px(p):
    return M["l"] + (p - X_MIN) / (X_MAX - X_MIN) * PLOT_W


def y_px(v):
    return M["t"] + (1 - (v - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT_H


def svg_chart():
    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             'role="img" aria-label="nansum throughput vs p(NaN)">']

    # y gridlines + labels
    for v in range(0, int(Y_MAX) + 1, Y_STEP):
        y = y_px(v)
        parts.append(f'<line x1="{M["l"]}" y1="{y:.1f}" x2="{M["l"]+PLOT_W}" '
                     f'y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{M["l"]-10}" y="{y+4:.1f}" class="ytick">{v}</text>')

    # x ticks + labels
    for p in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        x = x_px(p)
        parts.append(f'<line x1="{x:.1f}" y1="{M["t"]+PLOT_H}" x2="{x:.1f}" '
                     f'y2="{M["t"]+PLOT_H+5}" class="axis"/>')
        parts.append(f'<text x="{x:.1f}" y="{M["t"]+PLOT_H+22}" '
                     f'class="xtick">{p:g}</text>')

    # axes
    parts.append(f'<line x1="{M["l"]}" y1="{M["t"]+PLOT_H}" x2="{M["l"]+PLOT_W}" '
                 f'y2="{M["t"]+PLOT_H}" class="axis"/>')
    parts.append(f'<line x1="{M["l"]}" y1="{M["t"]}" x2="{M["l"]}" '
                 f'y2="{M["t"]+PLOT_H}" class="axis"/>')

    # axis titles
    parts.append(f'<text x="{M["l"]+PLOT_W/2:.1f}" y="{H-15}" '
                 'class="axislabel">p(NaN)</text>')
    parts.append(f'<text transform="translate(18,{M["t"]+PLOT_H/2:.1f}) rotate(-90)" '
                 'class="axislabel">throughput (M elem/s)</text>')

    # marker at p=0.5 (the branch-prediction minimum)
    xmid = x_px(0.5)
    parts.append(f'<line x1="{xmid:.1f}" y1="{M["t"]}" x2="{xmid:.1f}" '
                 f'y2="{M["t"]+PLOT_H}" class="mid"/>')

    # data lines + points + legend
    legend_y = M["t"] + 6
    for name in ENGINES:
        color = COLORS[name]
        pts = " ".join(f"{x_px(p):.1f},{y_px(v):.1f}"
                       for p, v in zip(SWEEP_P, SWEEP[name]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     'stroke-width="2.5"/>')
        for p, v in zip(SWEEP_P, SWEEP[name]):
            parts.append(f'<circle cx="{x_px(p):.1f}" cy="{y_px(v):.1f}" r="3" '
                         f'fill="{color}"/>')
        lx = M["l"] + PLOT_W + 20
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+22}" '
                     f'y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{lx+11}" cy="{legend_y}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{lx+28}" y="{legend_y+4}" class="legend">{name}</text>')
        legend_y += 22

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
<h1>CPython vs PyPy vs V8: pure-language loop benchmarks</h1>
<p class="sub">Equivalent summing loops, written idiomatically per language.
One machine, x86_64; best of 5 reps after a warm-up. Rough numbers — large
effects only.</p>

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
engine for that row. Optimized C leads everywhere, but by surprisingly small
margins — the JITs land within ~1.2–1.4× of scalar C on most rows. Among the
JITs, PyPy wins the tight numeric loops and V8 wins object method dispatch.
(At <code>-O2</code> the C plain sum stays a scalar reduction; with
<code>-ffast-math -march=native</code> it vectorizes to ~3340 M elem/s.)</p>

<h2>nansum throughput vs p(NaN)</h2>
<p>Summing 10M floats while skipping NaNs (<code>if v != v: continue</code>),
sweeping the fraction that are NaN. Each point is measured in its own process
so each engine's JIT compiles fresh for that fraction.</p>
<div class="chart">
{svg_chart()}
</div>
<p>All three trace a <strong>U</strong>: throughput bottoms out near
<strong>p&nbsp;=&nbsp;0.5</strong> (dashed line), where the skip branch is
maximally unpredictable (~50% mispredict), and recovers toward both extremes
where the CPU predicts it for free. The tighter the loop, the bigger the swing:
C and PyPy move ~4× from end to trough, V8 ~2.3×, while CPython is nearly flat
because interpreter overhead dwarfs the branch. At p&nbsp;=&nbsp;0.5 the branch
mispredict bottlenecks everyone — even C drops to ~290 M elem/s — and PyPy and
V8 nearly converge.</p>

<table>
<thead><tr><th class="num">p(NaN)</th>
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
