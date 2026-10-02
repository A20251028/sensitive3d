#!/usr/bin/env python3
"""Generate the synthetic street dataset (OSGB tiles with 4 LODs + OBJ + GLB)."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensitive3d.synthetic import generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="data/synthetic", help="output directory")
    ap.add_argument("--formats", default="osgb,obj,glb", help="comma separated: osgb,obj,glb")
    args = ap.parse_args()
    t0 = time.time()
    res = generate(args.out, formats=tuple(args.formats.split(",")))
    print(f"done in {time.time() - t0:.1f}s -> {args.out}")
    if "osgb" in res:
        print(f"  osgb: {res['osgb']['files']} files, {res['osgb']['triangles']} triangles")
    print(f"  {len(res['ground_truth'])} ground-truth signs")


if __name__ == "__main__":
    main()
