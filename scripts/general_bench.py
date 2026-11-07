#!/usr/bin/env python3
"""
General-ability benchmarks (default: the four from the ELVIS correlation study) for base + v3 models:
check every run is complete and judge-scored, then write one self-contained HTML report.

Per dataset x model (results/*/<key>_<dataset>*):
  answers   rows in <key>_<dataset>.xlsx vs the dataset tsv in $LMUData
  scored    a <key>_<dataset>_acc.csv exists (not an _old<date> backup)
  unscored  answers the rules could not parse and no GPT judge was used ("Failed in Prefetch, no GPT-based answer matching ...";
            MMBench's circular "Failed in Prefetching Rolling" is a scored wrong answer, not this)
  random    answers scored by VLMEvalKit's random fallback (judge failed)
A run is OK when all answers are there, it is scored, and unscored == 0.

Report: Overall accuracy (models x datasets), difference vs base (chart + table), and per-category
accuracy for each dataset. MMMU uses the 'validation' row (the reported split), others the first row.

Usage (on Fir, from fine-tuning-ELVIS):
  python scripts/general_bench.py                    # verify only
  python scripts/general_bench.py --html results/general_bench.html
  python scripts/general_bench.py --set t0p-cp --datasets MMStar AI2D_TEST
"""

import argparse
import base64
import io
from html import escape
from pathlib import Path

import pandas as pd

from eval_progress import MODELS, SETS, dataset_size, keys_for

DATASETS = ["MMMU_DEV_VAL", "MMStar", "MMBench_DEV_EN_V11", "AI2D_TEST"]
COLORS = ["#2a6fb0", "#c8553d", "#3a9a5b", "#8a5cb8"]  # v3 models, fixed order


def newest(files):
    files = [f for f in files if "_old" not in f.stem]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def check(key, dataset, results, total):
    """Completeness + judge stats for one run."""
    r = dict(answers=0, total=total, scored=False, unscored=None, random=None, acc=None)
    ans = newest(list(Path(results).glob(f"*/{key}_{dataset}.xlsx")))
    if ans is not None:
        r["answers"] = len(pd.read_excel(ans, usecols=["index"]))
    res = newest(list(Path(results).glob(f"*/{key}_{dataset}_*_result.xlsx")))
    if res is not None:
        d = pd.read_excel(res)
        log = d["log"].astype(str) if "log" in d else pd.Series([], dtype=str)
        r["unscored"] = int(log.str.contains("no GPT-based answer matching").sum())
        r["random"] = int(log.str.contains("randomly generate").sum())
    acc = newest(list(Path(results).glob(f"*/{key}_{dataset}_acc.csv")))
    if acc is not None:
        r["scored"] = True
        a = pd.read_csv(acc)
        val = "split" in a and (a["split"] == "validation").any()
        row = a[a["split"] == "validation"].iloc[0] if val else a.iloc[0]
        r["acc"] = {c: 100 * float(v) for c, v in row.items() if c != "split" and pd.notna(v) and _num(v)}
    r["ok"] = r["scored"] and r["unscored"] == 0 and (total is None or r["answers"] >= total)
    return r


def chart_png(overall, base):
    """Difference vs base per dataset, one bar per v3 model, as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    v3 = [m for m in overall.index if m != base]
    ds = list(overall.columns)
    fig, ax = plt.subplots(figsize=(max(6, 1.9 * len(ds) + 2), 3.8))
    w = 0.8 / max(1, len(v3))
    x = np.arange(len(ds))
    for i, m in enumerate(v3):
        delta = (overall.loc[m] - overall.loc[base]).values.astype(float)
        bars = ax.bar(x + (i - (len(v3) - 1) / 2) * w, delta, w * 0.92, label=m, color=COLORS[i % len(COLORS)])
        for b, v in zip(bars, delta):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.04 if v >= 0 else -0.04), f"{v:+.1f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color="#333")
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ds, fontsize=9)
    ax.set_ylabel("accuracy − base (points)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=len(v3), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def shade(delta):
    """Background for a difference vs base: blue = better, red = worse, stronger with size (capped at 3 points)."""
    if delta is None or pd.isna(delta) or abs(delta) < 0.05:
        return ""
    a = min(abs(delta) / 3, 1) * 0.45
    rgb = "42,111,176" if delta > 0 else "200,85,61"
    return f" style='background:rgba({rgb},{a:.2f})'"


def table(df, base, fmt="{:.1f}"):
    """Models x columns; non-base cells shaded by their difference vs base, which is shown in small text."""
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = ""
    for m, row in df.iterrows():
        cells = ""
        for c, v in row.items():
            if pd.isna(v):
                cells += "<td>–</td>"
                continue
            d = None if m == base or pd.isna(df.at[base, c]) else v - df.at[base, c]
            extra = f"<small>{d:+.1f}</small>" if d is not None else ""
            cells += f"<td{shade(d)}>{fmt.format(v)}{extra}</td>"
        body += f"<tr><th scope='row'>{escape(str(m))}</th>{cells}</tr>"
    return f"<div class='scroll'><table><tr><th></th>{head}</tr>{body}</table></div>"


CSS = """
:root{--bg:#fafaf8;--card:#fff;--ink:#1a1a1a;--muted:#666;--line:#e3e3e0;--ok:#1a7f4b;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--card:#1f1f1f;--ink:#e8e8e6;--muted:#a0a0a0;--line:#333;
  --ok:#5cc58a;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 48px}h1{font-size:21px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 6px}.muted{color:var(--muted)}
.scroll{overflow-x:auto}table{border-collapse:collapse;background:var(--card);font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}
th[scope=row],tr:first-child th{text-align:left}td small{display:block;font-size:11px;color:var(--muted)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--set", default="t0p-cp", choices=list(SETS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--results", default="results")
    parser.add_argument("--html", default=None, help="write the report here (e.g. results/general_bench.html)")
    args = parser.parse_args()

    keys = dict(zip(args.models, keys_for([args.set], args.models)))
    checks = {}
    print(f"{'dataset':20s} {'model':16s} {'answers':>13s} {'scored':6s} {'unscored':>8s} {'random':>6s}  status")
    for ds in args.datasets:
        total = dataset_size(ds)
        for m, key in keys.items():
            r = checks[ds, m] = check(key, ds, args.results, total)
            print(f"{ds:20s} {m:16s} {r['answers']:>6d}/{total or '?':<6} {'yes' if r['scored'] else 'no':6s} "
                  f"{'-' if r['unscored'] is None else r['unscored']:>8} {'-' if r['random'] is None else r['random']:>6}"
                  f"  {'OK' if r['ok'] else '!! check'}")
    bad = [k for k, r in checks.items() if not r["ok"]]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} runs OK"
          + (f"; not OK: {', '.join(f'{d}/{m}' for d, m in bad)}" if bad else ""))
    if not args.html:
        return

    base = "base" if "base" in keys else args.models[0]
    overall = pd.DataFrame({ds: {m: (checks[ds, m]["acc"] or {}).get("Overall") for m in keys}
                            for ds in args.datasets}).astype(float)
    if overall.isna().all().all():
        raise SystemExit("no scored runs - nothing to report")

    ver_rows = "".join(
        f"<tr><th scope='row'>{escape(ds)}</th><th scope='row'>{escape(m)}</th>"
        f"<td>{r['answers']}/{r['total'] or '?'}</td><td>{'yes' if r['scored'] else 'no'}</td>"
        f"<td>{'–' if r['unscored'] is None else r['unscored']}</td><td>{'–' if r['random'] is None else r['random']}</td>"
        f"<td class='{'ok' if r['ok'] else 'bad'}'>{'✓ OK' if r['ok'] else '✗ check'}</td></tr>"
        for (ds, m), r in checks.items())
    n_ok = len(checks) - len(bad)

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>General benchmarks</title>"
             f"<style>{CSS}</style></head><body><main>",
             f"<h1>General benchmarks: base vs v3 ({escape(args.set)})</h1>",
             f"<p class='muted'>VLMEvalKit, GPT-judge scoring (gpt-4o-mini). Accuracy in %. Small numbers = difference "
             f"vs base; blue = higher, red = lower. MMMU = validation split. Runs verified: "
             f"<b class='{'ok' if not bad else 'bad'}'>{n_ok}/{len(checks)} OK</b> (details at the bottom).</p>",
             "<h2>Overall accuracy</h2>", table(overall, base, "{:.2f}"),
             "<h2>Difference vs base</h2>",
             f"<img alt='accuracy minus base per dataset' src='data:image/png;base64,{chart_png(overall, base)}'>"]
    for ds in args.datasets:
        cat = pd.DataFrame({m: checks[ds, m]["acc"] or {} for m in keys}).T
        if cat.empty or "Overall" not in cat:
            continue
        cols = ["Overall"] + [c for c in cat.columns if c != "Overall"]
        parts += [f"<h2>{escape(ds)} by category</h2>", table(cat[cols].astype(float), base)]
    parts += ["<details><summary>Run verification</summary><div class='scroll'><table>"
              "<tr><th>dataset</th><th>model</th><th>answers</th><th>scored</th><th>unscored (no judge)</th>"
              f"<th>random fallback</th><th>status</th></tr>{ver_rows}</table></div></details>",
              "</main></body></html>"]
    out = Path(args.html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
