import csv
import numpy as np
import pytest
from rppg_liveness.evaluation import metrics, calibrate, load_manifest, wilson


def result(label, score):
    return dict(label=label, score=score, status="ok" if score is not None else "insufficient_quality")


def test_fail_closed_and_abstention_accounting():
    rows = [result("live", 0.9), result("live", None), result("attack", 0.8), result("attack", None)]
    m = metrics(rows, 0.55)
    assert m["confusion"] == dict(TP=1, FP=1, TN=1, FN=1)
    assert m["coverage"] == 0.5
    assert m["ACER_fail_closed"] == 0.5
    assert m["AUC_scorable_only"] == 1


def test_calibration_independent_of_test():
    dev = [result("live", 0.8), result("live", 0.7), result("attack", 0.6), result("attack", 0.1)]
    threshold = calibrate(dev)
    assert threshold == 0.7
    test = [result("live", 0.4), result("attack", 0.9)]
    assert metrics(test, threshold)["ACER_fail_closed"] == 1


def test_all_rejected_is_not_claimed_perfect():
    rows = [result("live", None), result("attack", None)]
    m = metrics(rows, 0.55)
    assert m["ACER_fail_closed"] == 0.5 and m["coverage"] == 0
    assert m["AUC_scorable_only"] is None
    with pytest.raises(ValueError, match="Calibration requires"):
        calibrate(rows)


def test_wilson_small_sample():
    assert wilson(0, 3)[1] > 0.5  # Zero observed false accepts is not zero population risk.
    assert wilson(0, 0) is None


def manifest(tmp_path, collision=None):
    rows = []
    for i, (split, label) in enumerate([("dev", "live"), ("dev", "attack"), ("test", "live"), ("test", "attack")]):
        p = tmp_path / f"{i}.avi"
        p.write_bytes(b"placeholder" + bytes([i]))
        rows.append(dict(path=p.name, label=label, subject_id=str(i), source_id=str(i), split=split,
                         attack_type="none" if label == "live" else "replay"))
    if collision in {"subject_id", "source_id"}:
        rows[2][collision] = rows[0][collision]
    if collision == "duplicate":
        (tmp_path / "2.avi").write_bytes((tmp_path / "0.avi").read_bytes())
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("collision", ["subject_id", "source_id", "duplicate"])
def test_leakage_rejected(tmp_path, collision):
    with pytest.raises(ValueError, match="leakage|Duplicate"):
        load_manifest(manifest(tmp_path, collision))


def test_relative_manifest_paths(tmp_path):
    assert len(load_manifest(manifest(tmp_path))) == 4


def test_evaluation_writes_report_and_chart(tmp_path, monkeypatch):
    from rppg_liveness import evaluation
    import json
    source = manifest(tmp_path)
    # Synthetic predictions test orchestration/serialization, NOT biometric accuracy.
    scores = {"0.avi": 0.9, "1.avi": 0.2, "2.avi": 0.85, "3.avi": 0.95}
    monkeypatch.setattr(evaluation, "analyze_video", lambda path: dict(
        status="ok", reason=None, bpm=72., score=scores[evaluation.Path(path).name]))
    report = evaluation.evaluate(source, tmp_path / "out")
    assert report["calibration"]["threshold"] == 0.9
    assert report["test"]["ACER_fail_closed"] == 1.0
    assert (tmp_path / "out/evaluation.png").stat().st_size > 100
    saved = json.loads((tmp_path / "out/report.json").read_text())
    assert saved["test"] == report["test"]
    assert (tmp_path / "out/predictions.csv").exists()
