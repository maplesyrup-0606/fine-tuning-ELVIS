#!/usr/bin/env python3
"""
Side-by-side review of saved ELVIS results for a random batch of patterns (no model runs).

For each sampled pattern (same patterns for every model) the page shows:
  - the 3+3 Stage 1 example images (<examples split>) and the ground-truth rubric
  - the 6 test images (<test split>) with their true label; images removed by
    rescore_without_leaks.py are marked "excluded"
  - one row per model: the rule it wrote (standard mode), and its answer to every
    test image in standard and rubric mode (correct / wrong)

Models: base + every <run>-<variant> found in the result folders, each at its last
checkpoint. Writes one self-contained HTML file (images embedded).

Usage:
  python scripts/review_samples.py                                  # 5 patterns per principle
  python scripts/review_samples.py --per-principle 10 --seed 1 --principles similarity symmetry
"""

import argparse
import base64
import html
import io
import json
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image

ALL_PRINCIPLES = ["continuity", "proximity", "similarity", "closure", "symmetry"]
MODEL_ORDER = ["base", "continuity-llm", "continuity-both", "all-llm", "all-both"]
IDS = [f"positive_{i}" for i in range(3)] + [f"negative_{i}" for i in range(3)]


def find_runs(root):
    """{model label: run dir} — base (latest timestamp) and each label's last checkpoint."""
    runs = {}
    for summary in sorted(Path(root).glob("**/summary.json")):
        d = summary.parent
        label = d.relative_to(root).parts[0]
        m = re.search(r"checkpoint-(\d+)", d.name)
        step = int(m.group(1)) if m else 0
        if label not in runs or step >= runs[label][0]:
            runs[label] = (step, d)
    return {k: v[1] for k, v in runs.items()}


def load_results(run_dir, principle):
    files = [f for f in sorted((run_dir / principle).glob("*.json")) if not f.name.endswith(".tmp.json")]
    return json.load(open(files[-1])) if files else {}


def img_tag(path, size=112):
    im = Image.open(path).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return (f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
            f'width="{size}" height="{size}" alt="">')


CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --ok:#1a7f4b; --bad:#c0392b; --chip:#f1f0ec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#161615; --ink:#ececea; --ink2:#a5a39d; --line:#33322f; --ok:#4cc38a; --bad:#ff6b5b; --chip:#23221f; } }
:root[data-theme="dark"] { --bg:#161615; --ink:#ececea; --ink2:#a5a39d; --line:#33322f; --ok:#4cc38a; --bad:#ff6b5b; --chip:#23221f; }
body { background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; margin:0 auto; padding:16px; max-width:1500px; }
h1 { font-size:20px; } h2 { font-size:17px; margin-top:36px; border-bottom:2px solid var(--line); padding-bottom:4px; }
.pattern { border:1px solid var(--line); border-radius:8px; padding:12px; margin:18px 0; overflow-x:auto; }
.name { font-family:ui-monospace, monospace; font-size:13px; color:var(--ink2); margin-bottom:6px; }
.rubric { background:var(--chip); border-radius:6px; padding:8px 10px; white-space:pre-wrap; margin:8px 0; }
.label { font-size:12px; color:var(--ink2); margin:8px 0 4px; font-weight:600; }
.imgs { display:flex; gap:6px; flex-wrap:wrap; }
.tile { text-align:center; font-size:12px; color:var(--ink2); }
.tile img { display:block; border:3px solid var(--line); border-radius:4px; }
.tile.pos img { border-color:var(--ok); } .tile.neg img { border-color:var(--bad); }
table { border-collapse:collapse; margin-top:10px; width:100%; }
th, td { border-top:1px solid var(--line); padding:6px 8px; vertical-align:top; text-align:left; }
th { font-size:12px; color:var(--ink2); font-weight:600; }
td.ans, td.model { text-align:center; font-weight:600; white-space:nowrap; }
td.model { text-align:left; }
.ok { color:var(--ok); } .bad { color:var(--bad); } .na { color:var(--ink2); }
.rule { white-space:pre-wrap; min-width:320px; max-width:640px; font-size:13px; }
.excl { text-decoration:line-through; opacity:.6; }
.legend { color:var(--ink2); }
"""


def answer_cell(sample, excluded):
    if sample is None:
        return '<td class="ans na">–</td>'
    ok = sample["predicted"] == sample["label"]
    title = html.escape(f"raw answer: {sample.get('response', '')!r}", quote=True)
    cls = ("ok" if ok else "bad") + (" excl" if excluded else "")
    return f'<td class="ans {cls}" title="{title}">{"✓" if ok else "✗"}</td>'


def tiles(data, principle, split, pattern, excluded=None):
    out = []
    for side, cls in (("positive", "pos"), ("negative", "neg")):
        for i, p in enumerate(sorted((data / principle / split / pattern / side).glob("*.png"))[:3]):
            sid = f"{side}_{i}"
            caption = f"<span class='{'excl' if excluded and (principle, pattern, sid) in excluded else ''}'>{side[:3]} {i}</span>"
            out.append(f"<div class='tile {cls}'>{img_tag(p)}{caption}</div>")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--standard-root", default="results/eval_v3_on_test_v3")
    parser.add_argument("--rubric-root", default="results/eval_v3_rubric_on_test_v3")
    parser.add_argument("--examples-split", default="train_v3")
    parser.add_argument("--test-split", default="test_v3")
    parser.add_argument("--exclusions", default="data/exclusions_test_v3.json")
    parser.add_argument("--principles", nargs="+", default=ALL_PRINCIPLES)
    parser.add_argument("--per-principle", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/review_samples.html")
    args = parser.parse_args()

    data = os.environ.get("ELVIS_DATA")
    if not data:
        sys.exit("ERROR: env var ELVIS_DATA is not set")
    data = Path(data)

    std_runs, rub_runs = find_runs(args.standard_root), find_runs(args.rubric_root)
    models = sorted(set(std_runs) | set(rub_runs),
                    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    if not models:
        sys.exit("ERROR: no results found under --standard-root / --rubric-root")
    excluded = set()
    if Path(args.exclusions).exists():
        for p, entries in json.load(open(args.exclusions))["excluded"].items():
            excluded |= {(p, e["pattern"], e["id"]) for e in entries}
    which = ", ".join(f"{m} ({(std_runs.get(m) or rub_runs.get(m)).name})" for m in models)
    print("models:", which)

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>ELVIS sample review</title><style>{CSS}</style></head><body>",
        "<h1>ELVIS sample review</h1>",
        f"<p class='legend'>{args.per_principle} random patterns per principle (seed {args.seed}); "
        f"examples from <b>{args.examples_split}</b>, test images from <b>{args.test_split}</b>. "
        "Green frame = positive, red frame = negative. ✓ / ✗ = the model's answer was correct / wrong "
        "(hover for the raw answer). Struck-through = image excluded for overlap with the fine-tuning data. "
        f"Models at their last checkpoint: {html.escape(which)}.</p>"]

    head = "".join(f"<th>{i.replace('positive', 'pos').replace('negative', 'neg')}</th>" for i in IDS)
    for principle in args.principles:
        rubrics = json.load(open(f"data/{principle}/rubrics.json"))["rubrics"]
        std = {m: load_results(d, principle) for m, d in std_runs.items()}
        rub = {m: load_results(d, principle) for m, d in rub_runs.items()}
        tables = [set(r) for r in list(std.values()) + list(rub.values()) if r]
        patterns = sorted(set.intersection(*tables)) if tables else []
        rng = random.Random(f"{args.seed}-{principle}")
        picked = sorted(rng.sample(patterns, min(args.per_principle, len(patterns))))
        parts.append(f"<h2>{principle}</h2>")
        for pattern in picked:
            rows = []
            for m in models:
                s_entry, r_entry = std.get(m, {}).get(pattern), rub.get(m, {}).get(pattern)
                s_samples = {s["id"]: s for s in (s_entry or {}).get("samples", [])}
                r_samples = {s["id"]: s for s in (r_entry or {}).get("samples", [])}
                rule = html.escape((s_entry or {}).get("logic_rules") or "–")
                acc = lambda e: f"{e['accuracy']:.0f}%" if e else "–"
                rows.append(
                    f"<tr><td class='model'>{m}</td><td class='rule'>{rule}</td>"
                    + "".join(answer_cell(s_samples.get(i), (principle, pattern, i) in excluded) for i in IDS)
                    + f"<td class='ans'>{acc(s_entry)}</td>"
                    + "".join(answer_cell(r_samples.get(i), (principle, pattern, i) in excluded) for i in IDS)
                    + f"<td class='ans'>{acc(r_entry)}</td></tr>")
            parts.append(
                f"<div class='pattern'><div class='name'>{html.escape(pattern)}</div>"
                f"<div class='label'>Stage 1 examples ({args.examples_split})</div>"
                f"<div class='imgs'>{tiles(data, principle, args.examples_split, pattern)}</div>"
                f"<div class='rubric'><b>Ground-truth rubric:</b> {html.escape(rubrics.get(pattern, '(none)'))}</div>"
                f"<div class='label'>Test images ({args.test_split})</div>"
                f"<div class='imgs'>{tiles(data, principle, args.test_split, pattern, excluded)}</div>"
                "<table><tr><th rowspan='2'>model</th><th rowspan='2'>rule the model wrote (standard mode)</th>"
                "<th colspan='7'>standard: own rule</th><th colspan='7'>rubric given</th></tr>"
                f"<tr>{head}<th>acc</th>{head}<th>acc</th></tr>{''.join(rows)}</table></div>")
    parts.append("</body></html>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
