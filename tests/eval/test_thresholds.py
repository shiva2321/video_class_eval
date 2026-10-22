import subprocess
import sys

from classeval.eval.metrics import exact_and_within_one
from classeval.eval.thresholds import fit_cutoffs, grade_of, loo_predict
from classeval.schema import MediaInfo, Report


def test_fit_cutoffs_separable_data_recovers_boundaries():
    # 4 linearly separable classes: 0, 1, 2, 3
    scores = [20.0, 25.0, 30.0, 50.0, 55.0, 60.0, 70.0, 75.0, 80.0, 90.0, 95.0]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

    c1, c2, c3 = fit_cutoffs(scores, labels, soft_le_average=[])

    assert 30.0 < c1 <= 50.0
    assert 60.0 < c2 <= 70.0
    assert 80.0 < c3 <= 90.0

    preds = [grade_of(s, (c1, c2, c3)) for s in scores]
    exact, within_one = exact_and_within_one(labels, preds)
    assert exact == 1.0
    assert within_one == 1.0


def test_fit_cutoffs_respects_soft_constraint():
    # Scores that might otherwise prefer a cutoff c2 <= 65
    scores = [20.0, 50.0, 70.0, 90.0]
    labels = [0, 1, 2, 3]

    # Soft constraint: class with score 68.0 must be Average or below (<= 1 -> score < c2)
    soft_scores = [68.0]
    c1, c2, c3 = fit_cutoffs(scores, labels, soft_le_average=soft_scores)

    # c2 must be strictly greater than 68.0 so grade_of(68.0) <= 1
    assert c2 > 68.0
    assert grade_of(68.0, (c1, c2, c3)) <= 1


def test_loo_does_not_peek():
    scores = [25.0, 55.0, 75.0, 92.0]
    labels = [0, 1, 2, 3]
    soft_scores = [40.0]

    preds1 = loo_predict(scores, labels, soft_scores)

    # Change the held-out label of index 1 from 1 to 3
    labels_tampered = [0, 3, 2, 3]
    preds2 = loo_predict(scores, labels_tampered, soft_scores)

    # Index 1's prediction should NOT change because its own label was held out
    assert preds1[1] == preds2[1]


def test_eval_loo_script_runs_cleanly_on_dummy_dir(tmp_path):
    # Build fake runs directory
    runs_dir = tmp_path / "eval_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    dummy_media = MediaInfo(
        duration_s=600.0,
        has_audio=True,
        has_video=True,
        container="mp4",
        size_bytes=1000,
    )

    sentinel_text = "SECRET_SENTINEL_STUDENT_PHRASE"

    classes_to_create = [
        ("excellent__B101", "Excellent", 88.0, 3),
        ("good__B102", "Good", 72.0, 2),
        ("average__B103", "Average", 55.0, 1),
        ("weak__B104", "Poor", 35.0, 0),
        ("quality_issue__B105", "Average", 48.0, None),
    ]

    for folder_name, grade, score, _ in classes_to_create:
        c_dir = runs_dir / folder_name
        c_dir.mkdir(parents=True, exist_ok=True)
        rep = Report(
            job_id=folder_name,
            created_at="2026-10-22T00:00:00Z",
            media=dummy_media,
            language="en",
            language_confidence=0.99,
            grade=grade,
            overall_score=score,
            confidence=0.9,
            coverage=1.0,
            summary=sentinel_text,  # Must never appear in stdout
            degraded=False,
        )
        (c_dir / "report.json").write_text(rep.model_dump_json(indent=2), encoding="utf-8")

    truth_yaml = tmp_path / "truth.yaml"
    truth_yaml.write_text("excellent/B101:\n  R1: true\n  R2: true\n", encoding="utf-8")

    out_cutoffs = runs_dir / "cutoffs.json"

    # Run eval_loo.py
    cmd = [
        sys.executable,
        "scripts/eval_loo.py",
        "--runs",
        str(runs_dir),
        "--truth",
        str(truth_yaml),
        "--out",
        str(out_cutoffs),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"eval_loo.py failed: {res.stderr}"

    # Sentinel text must never leak into stdout
    assert sentinel_text not in res.stdout
    # Output file was created
    assert out_cutoffs.exists()
