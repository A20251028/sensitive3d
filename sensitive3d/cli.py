"""Command line interface.

Auditable, step by step (recommended for real data)::

    sensitive3d scan    <input> --work W                 # import check (structure, textures, hashes)
    sensitive3d detect  <input> --work W [--diagnostics] # read only: W/detection.json + W/evidence/
    sensitive3d review-template W/detection.json -o W/review.json   # edit: accept / operation / protected
    sensitive3d repair  <input> <output> --work W --review W/review.json

Fully automatic (only confident candidates; the rest is reported as pending)::

    sensitive3d process <input> <output>

Other::

    sensitive3d serve   [--host 127.0.0.1 --port 8000 --allow-local-root DIR]
    sensitive3d synth   <out_dir>
    sensitive3d doctor  [--json]                      # check Python packages + OSGB bridge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _add_limits(p) -> None:
    p.add_argument("--max-seconds", type=float, help="stop a stage after this many seconds (evidence so far is kept)")
    p.add_argument("--max-memory-mb", type=float, help="stop when the process uses more memory than this")
    p.add_argument("--max-decoded-mb", type=float, help="refuse to decode more texture memory than this at once")
    p.add_argument("--bridge-timeout", type=float, help="timeout of one osgb_bridge call (seconds)")


def _config(args):
    from .pipeline import PipelineConfig
    from .runctx import RunLimits

    cfg = PipelineConfig()
    if getattr(args, "categories", None):
        cfg.detection.categories = tuple(args.categories.split(","))
    if getattr(args, "yolo", None):
        cfg.detection.yolo_model = args.yolo
    if getattr(args, "min_score", None) is not None:
        cfg.detection.min_score = args.min_score
    if getattr(args, "auto_score", None) is not None:
        cfg.auto_accept_score = args.auto_score
    if getattr(args, "inpaint", None):
        cfg.texture.method = args.inpaint
    if getattr(args, "lama", None):
        cfg.texture.lama_model = args.lama
        cfg.texture.method = "lama"
    if getattr(args, "texture_only", False):
        cfg.remove_geometry = False
    if getattr(args, "no_previews", False):
        cfg.previews = False
    if getattr(args, "diagnostics", False):
        cfg.diagnostics = True
    if getattr(args, "allow_scan_errors", False):
        cfg.allow_scan_errors = True
    if getattr(args, "regions", None):
        data = json.loads(Path(args.regions).read_text(encoding="utf-8"))
        cfg.regions = data.get("regions", []) if isinstance(data, dict) else data
    cfg.limits = RunLimits.from_dict(
        {
            "max_seconds": getattr(args, "max_seconds", None),
            "max_memory_mb": getattr(args, "max_memory_mb", None),
            "max_decoded_mb": getattr(args, "max_decoded_mb", None),
            "bridge_timeout": getattr(args, "bridge_timeout", None),
        }
    )
    return cfg


def _progress(f, msg):
    print(f"[{f * 100:5.1f}%] {msg}", flush=True)


def _pipeline(args):
    from .pipeline import Pipeline

    return Pipeline(_config(args), progress=_progress)


def cmd_scan(args) -> int:
    pipe = _pipeline(args)
    _, rep = pipe.scan(args.input, Path(args.work))
    print(json.dumps({k: rep.get(k) for k in ("format", "files", "files_ok", "tiles", "leaf_files", "levels", "ok_for_repair", "blocking_errors", "input_manifest")}, ensure_ascii=False, indent=1))
    print(f"report: {Path(args.work) / 'scan.json'}")
    return 0 if rep.get("ok_for_repair") else 2


def cmd_detect(args) -> int:
    pipe = _pipeline(args)
    work = Path(args.work)
    ds, scan = pipe.scan(args.input, work)
    det = pipe.detect(ds, work)
    from .review import default_decisions

    tmpl = work / "review.template.json"
    tmpl.write_text(json.dumps({"decisions": default_decisions(det), "protected": []}, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in det["candidates"]:
        st = "可自动" if c["status"] == "auto" else "待审核"
        print(f"  #{c['id']} [{st}] {c['category_label']} {c['label']} score={c['score']} mount={c['mount']['type']}/{c['mount']['confidence']} at {c['region']['center']}")
        for r in c["review_reasons"]:
            print(f"        - {r}")
    print(f"detection: {work / 'detection.json'}  evidence: {work / 'evidence'}  review template: {tmpl}")
    if det.get("stopped"):
        print(f"STOPPED: {det['stopped']['message']}")
        return 3
    return 0


def cmd_review_template(args) -> int:
    from .review import default_decisions

    det = json.loads(Path(args.detection).read_text(encoding="utf-8"))
    Path(args.output).write_text(json.dumps({"decisions": default_decisions(det), "protected": []}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {args.output}: set accept true/false, operation geometry|texture, optional margin; add protected boxes")
    return 0


def cmd_repair(args) -> int:
    from .io.dataset import Dataset

    pipe = _pipeline(args)
    work = Path(args.work)
    det = json.loads((Path(args.detection) if args.detection else work / "detection.json").read_text(encoding="utf-8"))
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    ds = Dataset(args.input)
    ds.scan()
    rep = pipe.repair(ds, det, review, args.output, work)
    print(json.dumps({k: rep.get(k) for k in ("files_modified", "faces_removed", "faces_added", "texels_painted", "warnings", "input_check", "stopped")}, ensure_ascii=False, indent=1))
    print(f"report: {work / 'repair.json'}")
    return 0 if not rep.get("stopped") and not rep.get("warnings") else 4


def cmd_process(args) -> int:
    from .pipeline import run

    t0 = time.time()
    report = run(args.input, args.output, _config(args), progress=_progress, log=None, work_dir=args.report_dir)
    print(json.dumps({k: v for k, v in report.items() if k not in ("regions", "files", "targets", "skipped", "atlas_residual", "protected_check")}, ensure_ascii=False, indent=1, default=str))
    for r in report["regions"]:
        print(f"  #{r['id']} {r['category_label']} {r['label']} score={r['score']} op={r.get('operation')} mount={r['mount']} at {r['center']}")
    for p in report.get("pending", []):
        print(f"  pending #{p['id']} {p['label']}: {'; '.join(p['reasons'])}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    if args.allow_local_root and args.host not in ("127.0.0.1", "localhost", "::1"):
        print("--allow-local-root only works when the server listens on localhost (--host 127.0.0.1)", file=sys.stderr)
        return 2
    defaults = {"limits": {k: v for k, v in {"max_seconds": args.max_seconds, "max_memory_mb": args.max_memory_mb, "max_decoded_mb": args.max_decoded_mb, "bridge_timeout": args.bridge_timeout}.items() if v is not None}}
    uvicorn.run(create_app(args.workspace, allow_local_roots=args.allow_local_root, defaults=defaults), host=args.host, port=args.port)
    return 0


def cmd_synth(args) -> int:
    from .synthetic import generate

    generate(args.out, formats=tuple(args.formats.split(",")))
    return 0


def cmd_doctor(args) -> int:
    from .doctor import environment_report, format_report

    report = environment_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(format_report(report))
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sensitive3d", description="Detect and remove sensitive traffic signs from 3D reality meshes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def detect_opts(p):
        p.add_argument("--categories", help="comma separated categories to remove (others are listed for review)")
        p.add_argument("--yolo", help="optional YOLO ONNX traffic-sign model (needs real weights)")
        p.add_argument("--min-score", type=float, default=None)
        p.add_argument("--auto-score", type=float, default=None, help="minimum score for automatic processing (default 0.85)")
        p.add_argument("--diagnostics", action="store_true", help="keep close-ups and reasons of rejected colour candidates")

    p = sub.add_parser("scan", help="import check: structure, LOD graph, textures, input hashes (read only)")
    p.add_argument("input")
    p.add_argument("--work", required=True, help="directory for scan.json and input_manifest.json")
    _add_limits(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("detect", help="detection only (read only): detection.json + evidence images + review template")
    p.add_argument("input")
    p.add_argument("--work", required=True)
    detect_opts(p)
    p.add_argument("--regions", help="JSON with regions to use instead of automatic detection")
    _add_limits(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("review-template", help="write an editable review.json from detection.json")
    p.add_argument("detection")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_review_template)

    p = sub.add_parser("repair", help="execute reviewed decisions into a separate output folder")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--work", required=True, help="work directory of the detect step")
    p.add_argument("--review", required=True)
    p.add_argument("--detection", help="default: <work>/detection.json")
    p.add_argument("--inpaint", choices=["auto", "telea", "ns", "lama"], default=None)
    p.add_argument("--lama", help="LaMa ONNX model for inpainting")
    p.add_argument("--no-previews", action="store_true")
    p.add_argument("--allow-scan-errors", action="store_true", help="repair although the scan reported blocking problems (not recommended)")
    _add_limits(p)
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("process", help="fully automatic: detect and remove confident signs, list the rest as pending")
    p.add_argument("input", help="OSGB dataset folder (with Data/), or a folder / file with .obj / .glb")
    p.add_argument("output", help="output folder (same layout as the input)")
    detect_opts(p)
    p.add_argument("--inpaint", choices=["auto", "telea", "ns", "lama"], default=None)
    p.add_argument("--lama", help="LaMa ONNX model for inpainting")
    p.add_argument("--texture-only", action="store_true", help="keep geometry, only repaint textures")
    p.add_argument("--no-previews", action="store_true")
    p.add_argument("--regions", help="JSON with regions to remove instead of automatic detection")
    p.add_argument("--report-dir", help="where reports and previews go (default: <output>_s3d)")
    _add_limits(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--allow-local-root", action="append", default=[], help="allow importing (copying) local folders under this directory; localhost only")
    _add_limits(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("synth", help="generate the synthetic test dataset")
    p.add_argument("out")
    p.add_argument("--formats", default="osgb,obj,glb")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("doctor", help="check the environment (Python packages, OSGB bridge selftest)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
