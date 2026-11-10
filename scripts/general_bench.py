#!/usr/bin/env python3
"""
General-ability benchmarks (default: the four from the ELVIS correlation study) for base + v3 models:
check every run is complete and judge-scored, then write one self-contained interactive HTML report.

Per dataset x model (results/*/<key>_<dataset>*):
  answers   rows in <key>_<dataset>.xlsx vs the dataset tsv in $LMUData
  scored    a <key>_<dataset>_acc.csv exists (not an _old<date> backup)
  unscored  answers the rules could not parse and no GPT judge was used ("Failed in Prefetch, no GPT-based answer matching ...";
            MMBench's circular "Failed in Prefetching Rolling" is a scored wrong answer, not this)
  random    answers scored by VLMEvalKit's random fallback (judge failed)
A run is OK when all answers are there, it is scored, and unscored == 0.

Report (scripts/general_bench.html is the page; the data is embedded as JSON): overview cards and a
difference-vs-base chart with each benchmark's noise band, then per benchmark the category tables for both
category levels (n, ~2 SE noise band, differences beyond it marked), a forest plot of one model vs base, and
example questions per category (image, options, correct answer, every model's answer).
MMMU uses the validation split (the reported one); MMBench counts circular question groups.

Usage (on Fir, from fine-tuning-ELVIS):
  python scripts/general_bench.py                                     # verify only
  python scripts/general_bench.py --html results/general_bench.html
  python scripts/general_bench.py --html results/gb.html --samples 12 --sample-cats structuralized_imagetext_understanding
  python scripts/general_bench.py --html results/general_bench.html --seed-sets t07-cp-s{0..9}   # + seed view
"""

import argparse
import ast
import base64
import io
import json
import os
from pathlib import Path

import pandas as pd

from eval_progress import MODELS, SETS, dataset_size, keys_for

DATASETS = ["MMMU_DEV_VAL", "MMStar", "MMBench_DEV_EN_V11", "AI2D_TEST"]
LABELS = {"MMMU_DEV_VAL": "MMMU (val)", "MMStar": "MMStar", "MMBench_DEV_EN_V11": "MMBench v1.1 (dev, EN)",
          "AI2D_TEST": "AI2D (test)"}
LEVELS = ("category", "l2-category")  # VLMEvalKit's two category levels; each question has one value per level
TEMPLATE = Path(__file__).with_name("general_bench.html")


def newest(files):
    files = [f for f in files if "_old" not in f.stem]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def judged(key, dataset, results):
    """Judged rows of one run (prediction, hit, category columns, log, ...); MMMU validation split only."""
    f = newest(list(Path(results).glob(f"*/{key}_{dataset}_*_result.xlsx")))
    if f is None:
        return None, None
    d = pd.read_excel(f)
    log = d["log"].astype(str) if "log" in d else pd.Series([], dtype=str)
    if "split" in d and (d["split"] == "validation").any():
        d = d[d["split"] == "validation"]
    d["index"] = d["index"].astype(str)
    return d.set_index("index"), log


def check(key, dataset, results, total):
    """Completeness + judge stats + accuracies for one run."""
    r = dict(answers=0, total=total, scored=False, unscored=None, random=None, acc=None, n={}, levels={})
    ans = newest(list(Path(results).glob(f"*/{key}_{dataset}.xlsx")))
    if ans is not None:
        r["answers"] = len(pd.read_excel(ans, usecols=["index"]))
    d, log = judged(key, dataset, results)
    if d is not None:
        r["unscored"] = int(log.str.contains("no GPT-based answer matching").sum())
        r["random"] = int(log.str.contains("randomly generate").sum())
        # Questions per category: the same rows report_acc averages (MMMU validation, MMBench circular groups).
        r["n"] = {"Overall": len(d)}
        for col in LEVELS:
            if col in d:
                counts = d[col].astype(str).value_counts()
                r["levels"][col] = list(counts.index)
                r["n"].update(counts.to_dict())
    acc = newest(list(Path(results).glob(f"*/{key}_{dataset}_acc.csv")))
    if acc is not None:
        r["scored"] = True
        a = pd.read_csv(acc)
        val = "split" in a and (a["split"] == "validation").any()
        row = a[a["split"] == "validation"].iloc[0] if val else a.iloc[0]
        r["acc"] = {c: 100 * float(v) for c, v in row.items() if c != "split" and pd.notna(v) and _num(v)}
    r["ok"] = r["scored"] and r["unscored"] == 0 and (total is None or r["answers"] >= total)
    return r


def thumbnails(cell, image_map, size=520, limit=3):
    """VLMEvalKit image field -> up to `limit` small JPEG data URIs. The field is base64, a list of base64
    strings (multi-image questions), or the index of another row holding the image (to save space)."""
    from PIL import Image
    s = str(cell)
    if 0 < len(s) <= 64:
        s = image_map.get(s, "")
    try:
        imgs = ast.literal_eval(s) if s.startswith("[") else [s]
    except (ValueError, SyntaxError):
        imgs = [s]
    out = []
    for b in imgs[:limit]:
        try:
            im = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
        except Exception:
            continue
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=72)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def samples(ds, runs, levels, k, only=None, seed=0):
    """Up to k example questions per finest-level category, with image thumbnails and every model's answer."""
    tsv = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / f"{ds}.tsv"
    base = next(iter(runs.values()))
    if k <= 0 or base is None or not tsv.is_file():
        return []
    meta = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    image_map = dict(zip(meta["index"], meta["image"])) if "image" in meta else {}
    meta = meta.set_index("index")
    level = max(levels, key=lambda lv: len(levels[lv])) if levels else None
    groups = base.groupby(base[level].astype(str)) if level else [("all questions", base)]
    out = []
    for name, g in groups:
        if only and not (name in only or any(str(g[lv].iloc[0]) in only for lv in levels if lv in g)):
            continue
        for idx in g.sample(min(k, len(g)), random_state=seed).index:
            row = meta.loc[idx] if idx in meta.index else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            out.append(dict(
                index=idx,
                cats={lv: str(g.at[idx, lv]) for lv in levels if lv in g},
                images=thumbnails(row["image"], image_map) if row is not None and "image" in row else [],
                question=str(g.at[idx, "question"]) if "question" in g else "",
                hint=str(row["hint"]) if row is not None and "hint" in row else "",
                options=[[c, str(row[c])] for c in "ABCDEFGHI" if row is not None and c in row and str(row[c]).strip()],
                answer=str(g.at[idx, "answer"]) if "answer" in g else "",
                preds={m: [str(r.at[idx, "prediction"])[:400], int(r.at[idx, "hit"])]
                       for m, r in runs.items() if r is not None and idx in r.index}))
    return out


def report_data(args, keys, checks, base):
    datasets = []
    for ds in args.datasets:
        b = checks[ds, base]
        acc = {m: checks[ds, m]["acc"] or {} for m in keys}
        levels = [dict(name=lv, groups=[dict(cat=c, n=b["n"].get(c), acc={m: acc[m].get(c) for m in keys})
                                         for c in cats if any(c in acc[m] for m in keys)])
                  for lv, cats in b["levels"].items()]
        ex = []
        if args.samples:
            runs = {m: judged(key, ds, args.results)[0] for m, key in keys.items()}
            ex = samples(ds, runs, b["levels"], args.samples, args.sample_cats)
        print(f"  {ds}: {sum(len(l['groups']) for l in levels)} category groups, {len(ex)} examples")
        datasets.append(dict(name=ds, label=LABELS.get(ds, ds), n=b["n"].get("Overall"),
                             overall={m: acc[m].get("Overall") for m in keys},
                             levels=[l for l in levels if l["groups"]], samples=ex))
    verify = [dict(ds=ds, model=m, answers=r["answers"], total=r["total"], scored=r["scored"],
                   unscored=r["unscored"], random=r["random"], ok=r["ok"]) for (ds, m), r in checks.items()]
    return dict(set=args.set, base=base, models=list(keys), datasets=datasets, verify=verify)


def seed_data(args, base):
    """Mean / SD / count over the --seed-sets runs, per dataset x model x (Overall + every category column),
    in the same shape as report_data's datasets (plus sd and k maps). Only verified (OK) runs are used."""
    import statistics
    sizes = {ds: dataset_size(ds) for ds in args.datasets}
    per, first, bad, n_runs = {}, {}, [], 0
    for st in args.seed_sets:
        for m, key in zip(args.models, keys_for([st], args.models)):
            for ds in args.datasets:
                n_runs += 1
                r = check(key, ds, args.results, sizes[ds])
                if not (r["ok"] and r["acc"]):
                    bad.append(f"{ds}/{m}/{st}")
                    continue
                per.setdefault((ds, m), []).append(r["acc"])
                if m == base:
                    first.setdefault(ds, r)

    def agg(ds, m, col):
        vals = [a[col] for a in per.get((ds, m), []) if col in a]
        if not vals:
            return None, None, 0
        return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals)

    out = []
    for ds in args.datasets:
        b = first.get(ds)
        if b is None:
            continue

        def block(col):
            stats = {m: agg(ds, m, col) for m in args.models}
            return ({m: v[0] for m, v in stats.items()}, {m: v[1] for m, v in stats.items()},
                    {m: v[2] for m, v in stats.items()})

        levels = []
        for lv, cats in b["levels"].items():
            groups = []
            for c in cats:
                acc, sd, k = block(c)
                if any(v is not None for v in acc.values()):
                    groups.append(dict(cat=c, n=b["n"].get(c), acc=acc, sd=sd, k=k))
            if groups:
                levels.append(dict(name=lv, groups=groups))
        acc, sd, k = block("Overall")
        out.append(dict(name=ds, label=LABELS.get(ds, ds), n=b["n"].get("Overall"), overall=acc, sd=sd, k=k,
                        levels=levels))
    print(f"  seed runs: {n_runs - len(bad)}/{n_runs} OK" + (f"; left out: {', '.join(bad)}" if bad else ""))
    return dict(label=f"{len(args.seed_sets)} seeds ({args.seed_sets[0].rsplit('-s', 1)[0]})", datasets=out,
                ok=n_runs - len(bad), total=n_runs, bad=bad)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--set", default="t0p-cp", choices=list(SETS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--results", default="results")
    parser.add_argument("--html", default=None, help="write the report here (e.g. results/general_bench.html)")
    parser.add_argument("--samples", type=int, default=4,
                        help="example questions per (finest) category in the report; 0 = none (smaller file)")
    parser.add_argument("--sample-cats", nargs="+", default=None,
                        help="only keep examples for these categories (e.g. structuralized_imagetext_understanding)")
    parser.add_argument("--seed-sets", nargs="+", default=None, choices=list(SETS),
                        help="also add a 'mean ± sd over seeds' view from these sets (e.g. t07-cp-s{0..9})")
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
    if not any(checks[ds, base]["acc"] for ds in args.datasets):
        raise SystemExit("base has no scored runs - nothing to report")
    data = report_data(args, keys, checks, base)
    if args.seed_sets:
        data["seeds"] = seed_data(args, base)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    out = Path(args.html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"written: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
