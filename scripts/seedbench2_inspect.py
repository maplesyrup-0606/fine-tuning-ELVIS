"""Read-only check of VLMEvalKit's SEEDBench2.tsv before building SEEDBench2_Fix.

Reports duplicated indices, image references (short `image` fields that point to
another row's index), references that point at a duplicated index (ambiguous),
multi-image storage, and what the existing _local.tsv did with the duplicates.
Writes nothing.

    python scripts/seedbench2_inspect.py [--tsv ~/LMUData/SEEDBench2.tsv]
"""
import argparse
import os
import os.path as osp
from collections import Counter

import pandas as pd


def lmu_root():
    r = os.environ.get("LMUData")
    return r if r and osp.exists(r) else osp.expanduser("~/LMUData")


def kind(v):
    if not isinstance(v, str):
        return "non-str"
    if len(v) <= 64:
        return "ref"
    if v.startswith("["):
        return "list"
    return "b64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=osp.join(lmu_root(), "SEEDBench2.tsv"))
    a = ap.parse_args()

    print("file:", a.tsv, f"{osp.getsize(a.tsv) / 1e9:.2f} GB")
    d = pd.read_csv(a.tsv, sep="\t", dtype={"index": str})
    print("rows:", len(d))
    print("columns:", list(d.columns))
    for c in ("image_path", "question_type_id", "data_type", "data_source"):
        if c in d:
            print(f"  {c} examples:", d[c].astype(str).head(3).tolist())

    cnt = Counter(d["index"])
    dup = {k for k, v in cnt.items() if v > 1}
    print(f"\nduplicated index values: {len(dup)} (rows {sum(cnt[k] for k in dup)}); "
          f"max copies {max(cnt.values())}")

    kinds = d["image"].map(kind)
    print("\nimage field kinds:", kinds.value_counts().to_dict())
    lst = d.loc[kinds == "list", "image"]
    if len(lst):
        print("  list example start:", lst.iloc[0][:80].replace("\n", " "))

    refs = d.loc[kinds == "ref", ["index", "image"]]
    if len(refs):
        tgt = refs["image"].astype(str)
        missing = (~tgt.isin(cnt.keys())).sum()
        amb = tgt.isin(dup).sum()
        from_dup = refs["index"].isin(dup).sum()
        print(f"  reference rows: {len(refs)}; target missing {missing}; "
              f"target is a duplicated index (ambiguous) {amb}; "
              f"referencing row itself duplicated {from_dup}")
        first_kind = d.drop_duplicates("index").set_index("index")["image"].map(kind)
        uniq_t = tgt[~tgt.isin(dup)]
        print("  kinds of unambiguous targets:",
              uniq_t.map(lambda t: first_kind.get(t, "missing")).value_counts().to_dict())
        print("  examples:", refs.head(5).values.tolist())

    print("\nduplicated pairs, first 5 (index, question_type_id, question[:60], image kind):")
    sub = d[d["index"].isin(dup)].sort_values("index", kind="stable")
    qt = "question_type_id" if "question_type_id" in d else None
    for k, g in list(sub.groupby("index", sort=False))[:5]:
        for _, r in g.iterrows():
            print(" ", k, r[qt] if qt else "", str(r["question"])[:60].replace("\n", " "), kind(r["image"]))

    local = a.tsv.replace(".tsv", "_local.tsv")
    if osp.exists(local):
        L = pd.read_csv(local, sep="\t", dtype={"index": str})
        same = L[L["index"].isin(dup)].groupby("index")["image_path"].nunique()
        print(f"\n_local.tsv: {len(L)} rows; duplicated indices whose two rows got the same "
              f"image_path: {(same == 1).sum()} / {len(same)}")
        print("  examples:", L.loc[L["index"].isin(dup), ["index", "image_path"]].head(4).values.tolist())
    else:
        print("\nno _local.tsv next to the file")


if __name__ == "__main__":
    main()
