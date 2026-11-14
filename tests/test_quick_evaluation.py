import json
from pathlib import Path
import pytest
from rppg_liveness import evaluation
from rppg_liveness.cli import main


def folders(tmp_path):
    for name in ("live", "print", "replay"):
        location = tmp_path / name
        location.mkdir(parents=True)
        (location / "trial.avi").write_bytes(name.encode())
    return tmp_path


def test_exploratory_trial_counts_live_false_reject_and_replay_false_accept(tmp_path, monkeypatch):
    root = folders(tmp_path / "data")
    scores = {"live": 0.22, "print": None, "replay": 0.8}
    def fake(path):
        score = scores[Path(path).parent.name]
        return dict(status="ok" if score is not None else "insufficient_quality",
                    reason=None if score is not None else "face_missing_or_multiple", score=score, bpm=None)
    monkeypatch.setattr(evaluation, "analyze_video", fake)
    out = tmp_path / "results"
    report = evaluation.quick_evaluate(root, out)
    assert report["counts"] == {"live": 1, "print": 1, "replay": 1}
    assert report["all"]["confusion"] == {"TP": 0, "FP": 1, "TN": 1, "FN": 1}
    assert report["all"]["accuracy_fail_closed"] == pytest.approx(1 / 3)
    assert report["by_attack_type"]["replay"]["APCER_fail_closed"] == 1.0
    assert len(json.loads((out / "predictions.json").read_text())) == 3
    assert (out / "predictions.csv").exists() and (out / "evaluation.png").stat().st_size > 100


def test_missing_attack_class_refuses_accuracy(tmp_path):
    root = folders(tmp_path)
    (root / "replay" / "trial.avi").unlink()
    with pytest.raises(ValueError, match="at least one"):
        evaluation.quick_evaluate(root, tmp_path / "out")


def test_duplicate_file_across_classes_rejected(tmp_path):
    root = folders(tmp_path)
    (root / "replay" / "trial.avi").write_bytes(b"live")
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.quick_evaluate(root, tmp_path / "out")


def test_quick_cli_invalid_threshold(tmp_path):
    assert main(["quick-evaluate", str(tmp_path), "--threshold", "nan"]) == 2
