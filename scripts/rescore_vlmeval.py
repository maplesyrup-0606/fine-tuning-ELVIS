#!/usr/bin/env python3
"""
Re-score saved VLMEvalKit MCQ answers with a working GPT judge (no model, no GPU).

Why: scoring at the end of a Slurm job runs on a compute node without internet, so the OpenAI
judge silently fails and VLMEvalKit falls back to exact matching: every answer the rules cannot
parse becomes 'Z' (log: "Failed in Prefetch ...") and counts as wrong. Run this on a LOGIN node
(compute nodes have no internet, so a GPU/CPU job would fail the same way).

For each run folder results/*/<key>_<dataset>.xlsx (the saved answers):
  1. rename the old scoring files in place, <key>_<dataset>_<x>.<ext> -> <key>_<dataset>_<x>_old<YYYYMMDD>.<ext>
     (kept for comparison; VLMEvalKit would otherwise reuse its cached *_result.pkl)
  2. call VLMEvalKit's own ImageMCQDataset.evaluate() with the judge, using a light dataset object
     with only index/question/answer/options (the full tsv with images gets killed on a login node)
  3. the new scoring takes the standard names, so plot/analysis scripts pick it up automatically;
     report old vs new accuracy and how many answers were still unscored ("Failed in Prefetch")

Usage (login node, OPENAI_API_KEY set):
  python scripts/rescore_vlmeval.py --dry-run              # list runs and their current scoring
  nohup python scripts/rescore_vlmeval.py > rescore.log 2>&1 &
  python scripts/rescore_vlmeval.py --keys Qwen3-VL-4B-Instruct-t0p Qwen3-VL-4B-Instruct
"""

import argparse
import glob
import os
import string
import time
from pathlib import Path

import pandas as pd


def light_dataset(dataset, tsv):
    """ImageMCQDataset with only the text columns its evaluate() needs (index, question, answer, A-Z)."""
    from vlmeval.dataset.image_mcq import ImageMCQDataset
    keep = lambda c: c in ("index", "question", "answer") or c in string.ascii_uppercase
    data = pd.concat(pd.read_csv(tsv, sep="\t", usecols=keep, chunksize=2000, dtype={"index": str}))
    # Same index handling as VLMEvalKit's ImageBaseDataset (image_base.py:157, 179-180): str, then int
    # if every index is an integer; chunked reads can otherwise leave mixed str/int indices.
    data["index"] = [str(x).strip() for x in data["index"]]
    if all(x.lstrip("-").isdigit() for x in data["index"]):
        data["index"] = [int(x) for x in data["index"]]
    ds = ImageMCQDataset.__new__(ImageMCQDataset)
    ds.dataset_name, ds.data = dataset, data
    return ds


def current_result(folder, key, dataset):
    """The standard-named result file (not an _old<date> copy)."""
    files = [f for f in glob.glob(f"{folder}/{key}_{dataset}_*_result.xlsx") if "_old" not in Path(f).stem]
    return sorted(files)[0] if files else None


def summary(result_xlsx):
    d = pd.read_excel(result_xlsx, usecols=["hit", "log"])
    log = d["log"].astype(str)
    return 100 * d["hit"].mean(), int(log.str.contains("Failed in Prefetch").sum()), \
        int(log.str.contains("randomly generate").sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--keys", nargs="+", default=None, help="model keys (default: every run found)")
    parser.add_argument("--dataset", default="SEEDBench2")
    parser.add_argument("--judge", default="gpt-4o-mini")
    parser.add_argument("--nproc", type=int, default=4, help="parallel judge calls (too many -> OpenAI rate limits)")
    parser.add_argument("--verbose", action="store_true", help="print the judge's raw API responses (shows API errors)")
    parser.add_argument("--max-random", type=int, default=0,
                        help="stop after a run with more random fallbacks than this (judge/API failures)")
    parser.add_argument("--results", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answers = sorted(Path(args.results).glob(f"*/*_{args.dataset}.xlsx"))
    runs = [(p.name[:-len(f"_{args.dataset}.xlsx")], p) for p in answers]
    if args.keys:
        runs = [(k, p) for k, p in runs if k in args.keys]
    if not runs:
        raise SystemExit(f"no saved answers matching {args.results}/*/<key>_{args.dataset}.xlsx")

    print(f"{len(runs)} runs:", flush=True)
    for key, p in runs:
        cur = current_result(p.parent, key, args.dataset)
        info = "acc {:.2f} | unscored {} | random {}".format(*summary(cur)) if cur else "no scoring yet"
        print(f"  {p.parent.name:62s} {info}")
    if args.dry_run:
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in this shell")
    from vlmeval.dataset.utils import build_judge
    judge = build_judge(model=args.judge)
    reply = str(judge.generate("Reply with the single letter A."))  # VLMEvalKit's working() missed a 401 key
    if not judge.working() or "Failed to obtain answer" in reply or "A" not in reply.upper():
        raise SystemExit(f"judge {args.judge} is not working from here (test reply: {reply[:120]!r}); "
                         "check OPENAI_API_KEY / internet - nothing changed")

    tsv = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / f"{args.dataset}.tsv"
    ds = light_dataset(args.dataset, tsv)
    tag = "_old" + time.strftime("%Y%m%d")
    print(f"\njudge {args.judge} OK; old scoring files kept with suffix {tag}\n", flush=True)
    for key, p in runs:
        cur = current_result(p.parent, key, args.dataset)
        old = summary(cur) if cur else None
        for f in map(Path, glob.glob(f"{p.parent}/{key}_{args.dataset}_*")):
            if "_old" in f.stem:
                continue  # an earlier backup
            target = f.with_name(f.stem + tag + f.suffix)
            if target.exists():
                # The original is already kept as target: f is left over from an interrupted re-score
                # (possibly with failed judge calls) -> discard it and re-score this run from scratch.
                print(f"  discarding unfinished re-score file {f.name} (original kept as {target.name})")
                f.unlink()
            else:
                f.rename(target)
        t0 = time.time()
        ds.evaluate(str(p), model=args.judge, nproc=args.nproc, verbose=args.verbose)
        new = summary(current_result(p.parent, key, args.dataset))
        before = f"{old[0]:.2f} (unscored {old[1]}, random {old[2]})" if old else "-"
        print(f"  {p.parent.name:62s} acc {before} -> {new[0]:.2f} (unscored {new[1]}, random {new[2]})"
              f"  [{time.time() - t0:.0f}s]", flush=True)
        if new[2] > args.max_random:
            raise SystemExit(f"\n!! {new[2]} answers in {p.parent.name} were scored with VLMEvalKit's RANDOM fallback "
                             f"(judge/API failures, e.g. rate limits). Stopping. Its old scoring is kept with suffix {tag}; "
                             f"fix the API issue (or lower --nproc), then rerun with --keys {key}")


if __name__ == "__main__":
    main()
