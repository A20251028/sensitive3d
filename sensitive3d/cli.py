"""Command line interface.

    sensitive3d process <input> <output> [options]   # detect + remove signs
    sensitive3d detect  <input> [--json out.json]     # detection only
    sensitive3d serve   [--host 0.0.0.0 --port 8000]  # web UI
    sensitive3d synth   <out_dir>                     # synthetic test data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _config(args):
    from .pipeline import PipelineConfig

    cfg = PipelineConfig()
    if getattr(args, "categories", None):
        cfg.detection.categories = tuple(args.categories.split(","))
    if getattr(args, "yolo", None):
        cfg.detection.yolo_model = args.yolo
    if getattr(args, "min_score", None) is not None:
        cfg.detection.min_score = args.min_score
    if getattr(args, "inpaint", None):
        cfg.texture.method = args.inpaint
    if getattr(args, "lama", None):
        cfg.texture.lama_model = args.lama
        cfg.texture.method = "lama"
    if getattr(args, "texture_only", False):
        cfg.remove_geometry = False
    if getattr(args, "no_previews", False):
        cfg.previews = False
    if getattr(args, "regions", None):
        cfg.regions = json.loads(Path(args.regions).read_text(encoding="utf-8"))
        if isinstance(cfg.regions, dict):
            cfg.regions = cfg.regions.get("regions", [])
    return cfg


def cmd_process(args) -> int:
    from .pipeline import run

    def progress(f, msg):
        print(f"[{f * 100:5.1f}%] {msg}", flush=True)

    t0 = time.time()
    report = run(args.input, args.output, _config(args), progress=progress, log=None, work_dir=args.report_dir)
    print(json.dumps({k: v for k, v in report.items() if k not in ("regions",)}, ensure_ascii=False, indent=1))
    for r in report["regions"]:
        print(f"  #{r['id']} {r['category_label']} {r['label']} score={r['score']} mount={r['mount']} at {r['center']}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def cmd_detect(args) -> int:
    from .io.dataset import Dataset
    from .pipeline import Pipeline

    ds = Dataset(args.input)
    ds.scan()
    regions, _ = Pipeline(_config(args), log=print).detect(ds)
    data = {"regions": [r.to_dict() for r in regions]}
    if args.json:
        Path(args.json).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in data["regions"]:
        print(f"  #{r['id']} {r['category_label']} {r['label']} score={r['score']} at {r['center']} size {r['width']}x{r['height']}m")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(args.workspace), host=args.host, port=args.port)
    return 0


def cmd_synth(args) -> int:
    from .synthetic import generate

    generate(args.out, formats=tuple(args.formats.split(",")))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sensitive3d", description="Detect and remove sensitive traffic signs from 3D reality meshes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--categories", help="comma separated categories to remove (default: all)")
        p.add_argument("--yolo", help="optional YOLO ONNX traffic-sign model")
        p.add_argument("--min-score", type=float, default=None)

    p = sub.add_parser("process", help="detect and remove sensitive signs")
    p.add_argument("input", help="OSGB dataset folder (with Data/), or a folder / file with .obj / .glb")
    p.add_argument("output", help="output folder (same layout as the input)")
    common(p)
    p.add_argument("--inpaint", choices=["auto", "telea", "ns", "lama"], default=None)
    p.add_argument("--lama", help="LaMa ONNX model for inpainting")
    p.add_argument("--texture-only", action="store_true", help="keep geometry, only repaint textures")
    p.add_argument("--no-previews", action="store_true")
    p.add_argument("--regions", help="JSON with regions to remove instead of automatic detection")
    p.add_argument("--report-dir", help="where report.json and previews go (default: <output>_s3d)")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("detect", help="detect only")
    p.add_argument("input")
    p.add_argument("--json", help="write regions to this file")
    common(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workspace", default="workspace")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("synth", help="generate the synthetic test dataset")
    p.add_argument("out")
    p.add_argument("--formats", default="osgb,obj,glb")
    p.set_defaults(func=cmd_synth)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
