"""Verify the SEEDBench2 index fix on Fir BEFORE launching the re-runs. Read-only.

Loads SEEDBench2_Fix exactly as a run would (VLMEvalKit build_dataset) and checks:
  1. 24,371 rows, 24,371 distinct indices, indices are the text ids ('5_0', not 50)
  2. questions / options / answers identical to the raw tsv, row for row
  3. every row's image file names are its own id ('1_10.jpg' / '1_10_0.jpg'), checked in memory;
     only a few files are looked up on disk (to confirm runs can find them)
  4. the formerly colliding pairs (e.g. '1_10' vs '11_0') now differ in question and image
Prints PASS/FAIL per check; exit code 1 on any FAIL.

    python scripts/seedbench2_verify_fix.py [--name SEEDBench2_Fix]
"""
import argparse
import os.path as osp
import re
import sys
from collections import Counter

import pandas as pd
from vlmeval.dataset import build_dataset
from vlmeval.smp import LMUDataRoot, toliststr

COLS = ["question", "A", "B", "C", "D", "answer"]


def report(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="SEEDBench2_Fix")
    a = ap.parse_args()
    root = LMUDataRoot()
    for f in (f"{a.name}.tsv", f"{a.name}_local.tsv"):
        p = osp.join(root, f)
        print(f"{p}: exists={osp.exists(p)} -> {osp.realpath(p)}")

    ds = build_dataset(a.name)
    d = ds.data
    print("dataset class:", type(ds).__name__, "| img_root:", ds.img_root)
    ok = True

    idx = list(d["index"])
    ok &= report(len(d) == 24371, f"rows = {len(d)} (expect 24371)")
    ok &= report(len(set(idx)) == len(idx), f"distinct indices = {len(set(idx))}")
    ok &= report(all(isinstance(x, str) and "_" in x for x in idx),
                 f"indices kept as text ids, e.g. {idx[:3]}")

    raw = pd.read_csv(osp.join(root, "SEEDBench2.tsv"), sep="\t", dtype=str,
                      usecols=["index"] + COLS).set_index("index")
    mine = d.set_index("index")[COLS].fillna("nan").astype(str)
    theirs = raw.loc[mine.index, COLS].fillna("nan").astype(str)
    bad = int((mine != theirs).sum().sum())
    ok &= report(bad == 0, f"question/options/answer match raw tsv for every row (mismatching cells: {bad})")

    paths = [toliststr(p) for p in d["image_path"]] if "image_path" in d else None
    if paths is None:
        print("note: no image_path column (images inline); skipping file checks")
    else:
        full = lambda x: x if osp.isabs(x) else osp.join(ds.img_root, x)  # noqa: E731
        print("image path example:", paths[0][:2], "->", full(paths[0][0]))
        wrong = [x for x, ps in zip(idx, paths)
                 if not all(re.fullmatch(rf"{re.escape(x)}(_\d+)?\.\w+", osp.basename(p)) for p in ps)]
        ok &= report(not wrong, f"rows whose image file names are not their own id: {len(wrong)} "
                     f"{wrong[:5]}")
        few = [p for ps in paths[::max(1, len(paths) // 20)] for p in ps[:1]]
        missing = [p for p in few if not osp.exists(full(p))]
        ok &= report(not missing, f"spot check on disk: {len(few) - len(missing)}/{len(few)} files found "
                     f"{missing[:3]}")
        shared = Counter(tuple(ps) for ps in paths)
        print(f"info: rows whose image set is shared with another row: "
              f"{sum(v for v in shared.values() if v > 1)} "
              "(fine if they are different questions about the same picture/video)")

    print("\nformerly colliding pairs (old int index -> rows):")
    by_int = {}
    for x in idx:
        by_int.setdefault(int(x), []).append(x)
    pairs = [v for v in by_int.values() if len(v) > 1]
    ok &= report(len(pairs) == 3099, f"pairs that WOULD have collided: {len(pairs)} (expect 3099)")
    dd = d.set_index("index")
    diff_q = sum(dd.at[p[0], "question"] != dd.at[p[1], "question"] or
                 dd.at[p[0], "answer"] != dd.at[p[1], "answer"] for p in pairs)
    print(f"info: of those, pairs differing in question or answer: {diff_q}")
    pmap = dict(zip(idx, map(tuple, paths))) if paths is not None else {}
    if paths is not None:
        same_img = sum(pmap[p[0]] == pmap[p[1]] for p in pairs)
        ok &= report(same_img == 0, f"pairs pointing at the same image file(s): {same_img}")
    for p in pairs[:3]:
        for x in p:
            r = dd.loc[x]
            img = pmap[x][0] if pmap else "-"
            print(f"  {x:>6}  ans={r['answer']}  q={str(r['question'])[:60]!r}  img={img}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - do not launch")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
