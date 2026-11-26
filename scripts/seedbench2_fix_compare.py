#!/usr/bin/env python3
"""
Old (collided) vs fixed SEED-Bench-2 results, per model and per question type. Read-only, prints tables
and writes CSVs to --out.

Old runs: dataset SEEDBench2, ids '1_10' and '11_0' were both read as 110, so 3,099 pairs shared one answer
and one key. Fixed runs: dataset SEEDBench2_Fix, ids kept as text. Rows are matched by
(int(id), question, options, answer); the type is the id's number before '_'.

Sections:
  1. per model: old acc, new acc, new acc on the 18,173 never-collided rows vs old there, and how many
     of those rows changed prediction (should be ~0: same model, same greedy setup)
  2. per type: base old vs new, n collided; FT-base (new) per model
  3. base vs each FT model (new): accuracy difference, gains / losses, paired 2SE noise band

Usage (on Fir, from fine-tuning-ELVIS, after the runs finished):
  python scripts/seedbench2_fix_compare.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval_progress import MODELS, keys_for

TYPES = ["Scene Understanding", "Instance Identity", "Instance Attributes", "Instance Location",
         "Instances Counting", "Spatial Relation", "Instance Interaction", "Visual Reasoning",
         "Text Understanding", "Celebrity Recognition", "Landmark Recognition", "Chart Understanding",
         "Visual Referring Expression", "Science Knowledge", "Emotion Recognition", "Visual Mathematics",
         "Difference Spotting", "Meme Comprehension", "Global Video Understanding", "Action Recognition",
         "Action Prediction", "Procedure Understanding", "In-Context Captioning",
         "Interleaved Image-Text Analysis", "Text-to-Image Generation", "Next Image Prediction",
         "Text-Image Creation"]
KEY_COLS = ["question", "A", "B", "C", "D", "answer"]


def result_file(key, ds, results):
    # the old name SEEDBench2 is a prefix of SEEDBench2_Fix: exclude the Fix files when looking for old ones
    files = [f for f in Path(results).glob(f"*/{key}_{ds}_*_result.xlsx")
             if "_old" not in f.stem and (ds.endswith("_Fix") or "_Fix" not in f.stem)]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def load(key, ds, results):
    f = result_file(key, ds, results)
    if f is None:
        return None
    d = pd.read_excel(f)
    d["match"] = d["index"].astype(str).str.replace("_", "").astype(int).astype(str) + "|" + \
        d[KEY_COLS].fillna("nan").astype(str).agg("|".join, axis=1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="t0p-cp")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/seedbench2_fix_compare")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    new, old = {}, {}
    for m, key in zip(a.models, keys_for([a.set], a.models)):
        n, o = load(key, "SEEDBench2_Fix", a.results), load(key, "SEEDBench2", a.results)
        if n is None or len(n) != 24371:
            print(f"[skip] {m}: fixed result missing or incomplete ({None if n is None else len(n)} rows)")
            continue
        new[m], old[m] = n, o
    if "base" not in new:
        raise SystemExit("no fixed base result yet")

    b = new["base"]
    b["type_id"] = b["index"].astype(str).str.split("_").str[0].astype(int)
    b["type"] = b["type_id"].map(lambda t: TYPES[t - 1])
    nint = b["index"].astype(str).str.replace("_", "").astype(int)
    b["collided"] = nint.map(nint.value_counts()) > 1
    print(f"rows {len(b)}, collided {int(b['collided'].sum())}, never collided {int((~b['collided']).sum())}\n")

    # 1. per model
    rows = []
    for m, n in new.items():
        n = n.set_index("match").loc[b["match"]].reset_index()
        new[m] = n
        r = {"model": m, "new_acc": n["hit"].mean() * 100}
        o = old.get(m)
        if o is not None:
            o = o.drop_duplicates("match").set_index("match").reindex(b["match"])
            ok = ~b["collided"].values
            r["old_acc"] = o["hit"].mean() * 100
            r["old_acc_never_coll"] = o["hit"].values[ok].mean() * 100
            r["new_acc_never_coll"] = n["hit"].values[ok].mean() * 100
            r["pred_changed_never_coll"] = int((o["prediction"].astype(str).values[ok] !=
                                                n["prediction"].astype(str).values[ok]).sum())
            r["hit_changed_never_coll"] = int((o["hit"].values[ok] != n["hit"].values[ok]).sum())
        r["new_acc_collided"] = n["hit"].values[b["collided"].values].mean() * 100
        rows.append(r)
    t1 = pd.DataFrame(rows).set_index("model").round(2)
    print("1. per model (accuracy %)\n", t1.to_string(), "\n")
    t1.to_csv(Path(a.out) / "per_model.csv")

    # 2. per type
    by = pd.DataFrame({"type_id": b["type_id"], "type": b["type"], "collided": b["collided"]})
    for m, n in new.items():
        by[m] = n["hit"].values
    if old.get("base") is not None:
        by["base_old"] = old["base"].drop_duplicates("match").set_index("match").reindex(b["match"])["hit"].values
    g = by.groupby(["type_id", "type"])
    t2 = pd.DataFrame({"n": g.size(), "collided": g["collided"].sum()})
    if "base_old" in by:
        t2["base_old"] = g["base_old"].mean() * 100
    t2["base_new"] = g["base"].mean() * 100
    for m in new:
        if m != "base":
            t2[f"{m}-base"] = (g[m].mean() - g["base"].mean()) * 100
    t2 = t2.round(1)
    pd.set_option("display.width", 250)
    print("2. per type: base accuracy old/new and FT - base (new), points\n", t2.to_string(), "\n")
    t2.to_csv(Path(a.out) / "per_type.csv")

    # 3. base vs FT (new)
    rows = []
    hb = new["base"]["hit"].values
    for m, n in new.items():
        if m == "base":
            continue
        h = n["hit"].values
        gain, loss = int(((h == 1) & (hb == 0)).sum()), int(((h == 0) & (hb == 1)).sum())
        rows.append({"model": m, "acc": h.mean() * 100, "diff": (h.mean() - hb.mean()) * 100,
                     "gains": gain, "losses": loss,
                     "noise_2SE": 2 * np.sqrt(gain + loss) / len(h) * 100})
    t3 = pd.DataFrame(rows).set_index("model").round(2)
    print(f"3. fixed SEED-Bench-2: FT vs base (base {hb.mean() * 100:.2f}%); noise_2SE = paired 2 SE\n",
          t3.to_string())
    t3.to_csv(Path(a.out) / "ft_vs_base.csv")


if __name__ == "__main__":
    main()
