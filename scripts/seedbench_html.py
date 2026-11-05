#!/usr/bin/env python3
"""
One HTML page to browse the SEED-Bench-2 plots: a tab per decoding set (radar + the 6 per-subpart
skill charts written by plot_seedbench.py --set ...), and the Overall accuracy table (models x sets) on top.

Run on the laptop after copying the folder. Reads <dir>/<set>/*.png and the table in <dir>/overall.txt, written on Fir with
  python scripts/eval_progress.py --table --set t0 t0p t07 t0p-cp t07-cp > results/plots/seedbench/overall.txt
Writes <dir>/index.html with relative image paths.

Usage:
  python scripts/seedbench_html.py ~/Downloads/seedbench
"""

import argparse
from html import escape
from pathlib import Path

import pandas as pd

from eval_progress import SETS

SET_INFO = {"t0": "T0, no penalty, generic prompt", "t0p": "T0 + presence penalty 1.5, generic prompt",
            "t07": "T0.7 + penalty 1.5, generic prompt", "t0p-cp": "T0 + penalty 1.5, Qwen MCQ prompt",
            "t07-cp": "T0.7 + penalty 1.5, Qwen MCQ prompt"}

CSS = """
:root{--bg:#fafaf8;--card:#fff;--ink:#1a1a1a;--muted:#666;--line:#e3e3e0;--accent:#1f5fa8;--hi:#eaf1fb}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--card:#1f1f1f;--ink:#e8e8e6;--muted:#a0a0a0;
  --line:#333;--accent:#7fb0ec;--hi:#1e2a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}
header{padding:20px 24px 8px}h1{font-size:20px;margin:0 0 4px}.muted{color:var(--muted)}
table{border-collapse:collapse;margin:12px 24px;background:var(--card);font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:5px 12px;text-align:right}th:first-child,td:first-child{text-align:left}
td.best{background:var(--hi);font-weight:600}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 24px;
  display:flex;gap:6px;flex-wrap:wrap;align-items:center;z-index:2}
nav button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:6px 14px;
  cursor:pointer;font:inherit}nav button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
nav .sub{margin-left:auto;gap:12px;flex-wrap:wrap}
nav .sub a{color:var(--accent);text-decoration:none;font-size:13px}
section{display:none;padding:12px 24px 40px}section.on{display:block}
.fig{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0;scroll-margin-top:60px}
.fig h3{margin:0 0 8px;font-size:15px}.fig img{max-width:100%;height:auto;display:block;margin:0 auto;background:#fff}
.radar img{max-width:760px}
"""

JS = """
function show(s){document.querySelectorAll('section,nav button').forEach(e=>e.classList.toggle('on',e.dataset.s===s));
  document.querySelectorAll('nav .sub').forEach(e=>e.style.display=e.dataset.s===s?'flex':'none');
  try{localStorage.setItem('sb2set',s)}catch(e){}}
let s0;try{s0=localStorage.getItem('sb2set')}catch(e){}
show(document.querySelector(`section[data-s="${s0}"]`)?s0:document.querySelector('section').dataset.s);
"""


def title_of(png):
    if png.stem == "radar":
        return "Per-subpart accuracy (radar)"
    num, _, slug = png.stem.partition("_")
    return f"{num}. " + slug.replace("_", " ").replace(" text ", " & text ").capitalize()


def render_table(df):
    """overall.txt = eval_progress.py --table output: model names in the first column, '-' = not scored."""
    df = df.set_index(df.columns[0]) if df.columns[0] not in SETS else df
    head = "".join(f"<th title='{escape(SET_INFO.get(s, ''))}'>{escape(s)}</th>" for s in df.columns)
    rows = ""
    for m, r in df.iterrows():
        cells = ""
        for s, v in r.items():
            col = [float(x) for x in df[s] if x != "-"]
            best = v != "-" and col and float(v) == max(col)
            cells += f"<td class='{'best' if best else ''}'>{escape(str(v))}</td>"
        rows += f"<tr><td>{escape(str(m))}</td>{cells}</tr>"
    return f"<table><tr><th>model</th>{head}</tr>{rows}</table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("dir", nargs="?", default="results/plots/seedbench", help="the copied plots folder")
    parser.add_argument("--set", nargs="+", default=list(SETS), choices=list(SETS))
    args = parser.parse_args()

    plots = Path(args.dir).expanduser()
    sets = [s for s in args.set if (plots / s).is_dir()]
    if not sets:
        raise SystemExit(f"no plot folders in {plots}/ - run plot_seedbench.py --set ... first")

    table = ""
    if (plots / "overall.txt").is_file():
        # --table prints the index unnamed: the header line has one field fewer than the rows
        table = render_table(pd.read_csv(plots / "overall.txt", sep=r"\s+", dtype=str))
    else:
        print(f"[note] no {plots}/overall.txt - page without the accuracy table")

    nav, subs, sections = [], [], []
    for s in sets:
        pngs = sorted((plots / s).glob("*.png"), key=lambda p: (p.stem != "radar", p.stem))
        nav.append(f"<button data-s='{s}' onclick=\"show('{s}')\">{escape(s)}</button>")
        figs = "".join(
            f"<div class='fig{' radar' if p.stem == 'radar' else ''}' id='{s}-{p.stem}'><h3>{escape(title_of(p))}</h3>"
            f"<a href='{s}/{p.name}' target='_blank'><img loading='lazy' src='{s}/{p.name}' alt='{escape(title_of(p))}'></a></div>"
            for p in pngs)
        links = "".join(f"<a href='#{s}-{p.stem}'>{escape('radar' if p.stem == 'radar' else title_of(p))}</a>"
                        for p in pngs)
        sections.append(f"<section data-s='{s}'><p class='muted'>{escape(SET_INFO[s])}</p>{figs}</section>")
        subs.append(f"<span class='sub' data-s='{s}'>{links}</span>")

    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>SEED-Bench-2 plots</title>"
            f"<style>{CSS}</style></head><body><header><h1>SEED-Bench-2: base vs v3</h1>"
            f"<div class='muted'>Overall accuracy (%), GPT-judge scoring. Highlighted = best in column. "
            f"Hover a set name for its decoding; click a chart to open it full size.</div></header>"
            f"{table}<nav>{''.join(nav + subs)}</nav>{''.join(sections)}<script>{JS}</script></body></html>")
    out = plots / "index.html"
    out.write_text(html)
    print(f"written: {out}  ({len(sets)} sets: {' '.join(sets)})")


if __name__ == "__main__":
    main()
