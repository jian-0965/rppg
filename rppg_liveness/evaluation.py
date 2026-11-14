"""Subject-disjoint dev/test evaluation; quality failures never silently disappear."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from .core import Config, config_dict
from .video import analyze_video


def sample_diagnostic(row, threshold):
    """Expose score arithmetic for each sample without claiming root cause."""
    penalties = row.get("score_penalties", {})
    return {"path": row["path"], "label": row["label"],
            "decision": ("inconclusive" if row["status"] != "ok" else
                         "live_candidate" if row["score"] >= threshold else "attack_candidate"),
            "score": row["score"], "threshold": threshold,
            "features": row.get("features", {}),
            "largest_score_penalties": [{"feature": key, "value": value}
                                        for key, value in sorted(penalties.items(), key=lambda item: -item[1])[:3]],
            "roi_bpm": row.get("roi_bpm"), "chrom_roi_bpm": row.get("chrom_roi_bpm"),
            "roi_half_bpm": row.get("roi_half_bpm"),
            "quality_reason": row.get("reason")}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "subject_id", "source_id", "split", "attack_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest columns must include {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    seen_paths, seen_hashes = set(), set()
    groups = {"dev": {"subject_id": set(), "source_id": set()}, "test": {"subject_id": set(), "source_id": set()}}
    for row in rows:
        for key in required:
            row[key] = (row[key] or "").strip()
        if row["label"] not in {"live", "attack"} or row["split"] not in groups:
            raise ValueError("label must be live/attack; split must be dev/test")
        if any(not row[k] for k in required):
            raise ValueError("Required fields cannot be empty")
        if (row["label"] == "live") != (row["attack_type"] == "none"):
            raise ValueError("Live attack_type must be none; attack must have a named attack_type")
        video = (path.parent / row["path"]).resolve()
        if not video.is_file():
            raise ValueError(f"Video does not exist: {video}")
        digest = sha256(video)
        if str(video) in seen_paths or digest in seen_hashes:
            raise ValueError("Duplicate video content/path in manifest")
        seen_paths.add(str(video))
        seen_hashes.add(digest)
        row["absolute_path"], row["sha256"] = str(video), digest
        for key in groups[row["split"]]:
            groups[row["split"]][key].add(row[key])
    for key in groups["dev"]:
        if groups["dev"][key] & groups["test"][key]:
            raise ValueError(f"Dev/test leakage: shared {key}")
    for split in groups:
        if {r["label"] for r in rows if r["split"] == split} != {"live", "attack"}:
            raise ValueError(f"{split} needs both live and attack videos")
    return rows


def wilson(successes, total):
    if not total:
        return None
    p, z = successes / total, 1.95996398454
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    margin = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [float(max(0, center-margin)), float(min(1, center+margin))]


def metrics(rows, threshold):
    """Operational fail-closed: inconclusive never accepts, even for attacks."""
    live = [r for r in rows if r["label"] == "live"]
    attack = [r for r in rows if r["label"] == "attack"]
    accepted = lambda r: r["status"] == "ok" and r["score"] is not None and r["score"] >= threshold
    tp, fp = sum(map(accepted, live)), sum(map(accepted, attack))
    fn, tn = len(live) - tp, len(attack) - fp
    apcer = fp / len(attack) if attack else None
    bpcer = fn / len(live) if live else None
    scorable = [r for r in rows if r["status"] == "ok" and r["score"] is not None]
    auc = None
    if {r["label"] for r in scorable} == {"live", "attack"}:
        auc = float(roc_auc_score([r["label"] == "live" for r in scorable], [r["score"] for r in scorable]))
    return {"n": len(rows), "threshold": threshold, "coverage": len(scorable)/len(rows) if rows else None,
            "inconclusive": len(rows)-len(scorable), "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "accuracy_fail_closed": (tp+tn)/len(rows) if rows else None,
            "accuracy_ci95_wilson_video_level": wilson(tp+tn, len(rows)),
            "APCER_fail_closed": apcer, "BPCER_fail_closed": bpcer,
            "APCER_ci95_wilson_video_level": wilson(fp, len(attack)),
            "ACER_fail_closed": (apcer+bpcer)/2 if apcer is not None and bpcer is not None else None,
            "AUC_scorable_only": auc,
            "live_inconclusive_rate": sum(r["status"] != "ok" for r in live)/len(live) if live else None,
            "attack_inconclusive_rate": sum(r["status"] != "ok" for r in attack)/len(attack) if attack else None}


def calibrate(dev):
    if {r["label"] for r in dev if r["status"] == "ok"} != {"live", "attack"}:
        raise ValueError("Calibration requires scorable live AND attack development videos")
    candidates = sorted({0.0, 1.000001, *(float(r["score"]) for r in dev if r["status"] == "ok")})
    # Minimize balanced error on development set; tie favors lower attack acceptance.
    return min(candidates, key=lambda t: (metrics(dev, t)["ACER_fail_closed"],
                                          metrics(dev, t)["APCER_fail_closed"], -t))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(manifest, output):
    rows, results = load_manifest(manifest), []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        print(f"[{index+1}/{len(rows)}] {row['split']} {row['path']}", flush=True)
        # File errors are retained as inconclusive; programming errors propagate.
        result = analyze_video(row["absolute_path"])
        results.append({**{k: v for k, v in row.items() if k != "absolute_path"}, **result})
    write_json(output / "predictions.json", results)
    dev, test = [r for r in results if r["split"] == "dev"], [r for r in results if r["split"] == "test"]
    threshold = calibrate(dev)
    calibration = {"schema_version": 1, "pipeline_version": "0.1.0", "threshold": threshold,
                   "config": config_dict(), "selection": "minimum dev ACER; ties lower APCER",
                   "dev_subjects": sorted({r["subject_id"] for r in dev}),
                   "dev_sources": sorted({r["source_id"] for r in dev})}
    write_json(output / "calibration.json", calibration)
    report = {"warning": "Experimental rPPG baseline; replay can preserve pulse. No production security claim.",
              "calibration": calibration, "dev": metrics(dev, threshold), "test": metrics(test, threshold),
              "test_scorable_only": metrics([r for r in test if r["status"] == "ok"], threshold),
              "test_by_attack_type": {kind: metrics([r for r in test if r["attack_type"] == kind or r["label"] == "live"], threshold)
                                      for kind in sorted({r["attack_type"] for r in test if r["label"] == "attack"})},
              "ci_note": "Wilson intervals assume independent videos; repeated subjects/sessions violate this. Not subject-bootstrap intervals."}
    write_json(output / "report.json", report)
    fields = ["path", "split", "subject_id", "label", "attack_type", "status", "reason", "score", "bpm"]
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    plot_report(test, threshold, output / "evaluation.png")
    return report


def quick_evaluate(data_dir, output, threshold=0.55, max_motion=Config().max_motion):
    """A fixed-threshold exploratory trial, with no calibration or population claim.

    data_dir/live, data_dir/print and data_dir/replay contain independently
    recorded camera presentations. Each file counts exactly once.
    """
    root = Path(data_dir).resolve()
    output = Path(output).resolve()
    if not np.isfinite(max_motion) or not 0 < max_motion <= 0.2:
        raise ValueError("max_motion must be finite and in (0, 0.2]")
    config = Config(max_motion=float(max_motion))
    types = {"live": ("live", "none"), "print": ("attack", "print"),
             "replay": ("attack", "replay")}
    extensions = {".avi", ".mp4", ".mov", ".mkv"}
    rows = []
    inventory = {}
    for folder, (label, attack_type) in types.items():
        location = root / folder
        if not location.is_dir():
            raise ValueError(f"Missing folder: {location}")
        entries = sorted(location.iterdir())
        recognized = []
        for path in entries:
            if path.is_file() and path.suffix.lower() in extensions:
                rows.append({"path": str(path), "label": label, "attack_type": attack_type})
                recognized.append(path.name)
        inventory[folder] = {"recognized": recognized,
                             "found": [p.name + ("/" if p.is_dir() else "") for p in entries]}
    counts = {folder: sum(row["attack_type"] == kind for row in rows)
              for folder, (_, kind) in types.items()}
    if any(counts[k] < 1 for k in types):
        details = "; ".join(f"{folder}: {counts[folder]} recognized, found "
                            f"{', '.join(inventory[folder]['found']) or '(empty)'}"
                            for folder in types)
        raise ValueError("Provide at least one video in each of live/, print/ and replay/. "
                         f"Checked {root}. {details}. Supported: .mp4 .avi .mov .mkv "
                         "(files directly inside each folder).")
    hashes = [sha256(row["path"]) for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate video contents across trial folders")
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['attack_type']} {Path(row['path']).name}", flush=True)
        row.update(analyze_video(row["path"], config=config))
        row["sha256"] = hashes[index-1]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "predictions.json", rows)
    scorable = [r for r in rows if r["status"] == "ok" and r["score"] is not None]
    classes = {r["label"] for r in scorable}
    detection = ({"status": "measurable_on_scorable_subset", **metrics(scorable, threshold)}
                 if classes == {"live", "attack"} else
                 {"status": "not_measurable", "accuracy": None, "APCER": None,
                  "BPCER": None, "AUC": None,
                  "reason": "Need at least one scorable live AND attack video"})
    failures = [{"path": r["path"], "label": r["label"], "reason": r["reason"],
                 "quality": r.get("quality", {})} for r in rows if r["status"] != "ok"]
    report = {"kind": "exploratory_fixed_threshold", "threshold": threshold,
              "config": {"max_motion": config.max_motion},
              "warning": "Measured only on these submitted videos. Default threshold is unvalidated; repeated subjects are correlated. Replay may preserve pulse.",
              "counts": counts, "detection": detection, "quality_failures": failures,
              "sample_diagnostics": [sample_diagnostic(r, threshold) for r in rows],
              "all": metrics(rows, threshold),
              "by_attack_type": {kind: metrics([r for r in rows if r["attack_type"] in {"none", kind}], threshold)
                                 for kind in ("print", "replay")},
              "note": "largest_score_penalties are arithmetic contributions, not proven causes. all.*_fail_closed describes operational reject policy; its accuracy can look high even with zero scorable samples. Use detection metrics only with both classes scorable, alongside coverage. This is not a population accuracy estimate."}
    write_json(output / "report.json", report)
    fields = ["path", "label", "attack_type", "status", "reason", "score", "bpm", "sha256"]
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_report(rows, threshold, output / "evaluation.png", title="Trial confusion (fail-closed)")
    return report


def plot_report(rows, threshold, path, title="Test confusion (fail-closed)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    m = metrics(rows, threshold)["confusion"]
    matrix = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Reject', 'Accept'],
                yticklabels=['Attack', 'Live'], title=title, xlabel='Decision', ylabel='Truth')
    valid = [r for r in rows if r["status"] == "ok"]
    if {r["label"] for r in valid} == {"live", "attack"}:
        fpr, tpr, _ = roc_curve([r["label"] == "live" for r in valid], [r["score"] for r in valid])
        axes[1].plot(fpr, tpr)
    else:
        axes[1].text(0.1, 0.5, "Insufficient scorable classes")
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set(title='ROC: scorable videos ONLY', xlabel='Attack acceptance', ylabel='Live acceptance')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
