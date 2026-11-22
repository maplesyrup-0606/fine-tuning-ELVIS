#!/usr/bin/env python3
"""
Find VLMEvalKit cache files (.pkl) that cannot be loaded, e.g. judge caches left empty or half-written
when the OpenAI quota ran out. VLMEvalKit reloads them on every attempt and fails with
"Ran out of input" (EOFError), so a run can never finish until they are moved away.

Lists unreadable files under $VLMEVALKIT_DIR/outputs/<pattern>/ (default: every *_result.pkl);
with --move, moves them to a backup folder (nothing is deleted). VLMEvalKit then rebuilds them
from the saved answers on the next run.

Usage (on Fir):
  python scripts/find_broken_pkl.py --pattern "*vit*-e3-*"
  python scripts/find_broken_pkl.py --pattern "*vit*-e3-*" --move
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--pattern", default="*", help="model folder pattern under outputs/ (e.g. '*vit*-e3-*')")
    parser.add_argument("--files", default="*_result.pkl", help="file pattern to check (default: judge caches)")
    parser.add_argument("--move", action="store_true", help="move unreadable files to --backup")
    parser.add_argument("--backup", default=os.path.expandvars("$SCRATCH/broken_vlmeval_cache"))
    args = parser.parse_args()

    root = Path(os.environ.get("VLMEVALKIT_DIR", os.path.expandvars("$SCRATCH/VLMEvalKit"))) / "outputs"
    files = sorted(f for d in root.glob(args.pattern) for f in d.rglob(args.files) if not f.is_symlink())
    broken = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                pickle.load(fh)
        except Exception as e:
            broken.append(f)
            print(f"{type(e).__name__:12s} {f.stat().st_size:>9d} B  {f.relative_to(root)}")
    print(f"\n{len(broken)} unreadable of {len(files)} checked")

    if args.move and broken:
        backup = Path(args.backup)
        for f in broken:
            target = backup / f.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), target)
        print(f"moved {len(broken)} files to {backup} (same folder structure)")


if __name__ == "__main__":
    main()
