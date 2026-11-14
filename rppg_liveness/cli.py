import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from .core import Config, config_dict, decide
from .evaluation import evaluate, quick_evaluate, write_json
from .video import analyze_video, camera


def resolve_threshold(args):
    if args.calibration:
        model = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if (model.get("schema_version") != 1 or model.get("pipeline_version") != "0.1.0"
                or model.get("config") != config_dict()):
            raise ValueError("Calibration version/config does not match this pipeline")
        threshold = float(model["threshold"])
    else:
        threshold = 0.55 if args.threshold is None else args.threshold
    if not np.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError("Threshold must be finite and between 0 and 1.000001")
    return threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental rPPG liveness baseline; replay can pass.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("video", "camera"):
        p = sub.add_parser(name)
        if name == "video":
            p.add_argument("path", type=Path)
        else:
            p.add_argument("--index", type=int, default=0)
            p.add_argument("--seconds", type=float, default=0, help="0: until Q/ESC, otherwise capture duration")
            p.add_argument("--no-display", action="store_true")
        group = p.add_mutually_exclusive_group()
        group.add_argument("--threshold", type=float, help="Manual experimental threshold; default 0.55")
        group.add_argument("--calibration", type=Path, help="calibration.json from independent development data")
        p.add_argument("--output", type=Path)
    p = sub.add_parser("evaluate", help="Calibrate on dev then report held-out test metrics")
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    p = sub.add_parser("quick-evaluate", help="Fixed-threshold exploratory accuracy from live/print/replay video folders")
    p.add_argument("data_dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/quick-evaluation"))
    p.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            report = evaluate(args.manifest, args.output)
            print(json.dumps(report["test"], ensure_ascii=False, indent=2, allow_nan=False))
            print(f"Report: {args.output / 'report.json'}")
            return 0
        if args.command == "quick-evaluate":
            if not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
                raise ValueError("--threshold must be finite and between 0 and 1")
            report = quick_evaluate(args.data_dir, args.output, args.threshold)
            print(json.dumps(report["all"], ensure_ascii=False, indent=2, allow_nan=False))
            print(f"Report: {args.output / 'report.json'}")
            return 0
        threshold = resolve_threshold(args)
        if args.command == "video":
            if not args.path.is_file():
                raise ValueError(f"Video file does not exist: {args.path}")
            result = analyze_video(args.path)
        else:
            if not np.isfinite(args.seconds) or args.seconds < 0:
                raise ValueError("--seconds must be finite and >= 0")
            if args.no_display and args.seconds <= 0:
                raise ValueError("--no-display requires a positive --seconds duration")
            result = camera(args.index, threshold, args.seconds, not args.no_display)
        result.update(decision=decide(result, threshold), threshold=threshold,
                      threshold_source="development_calibration" if args.calibration else "unvalidated_manual_or_default",
                      warning="Pulse evidence is not proof of liveness; high-quality replay may pass.",
                      config=config_dict())
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
