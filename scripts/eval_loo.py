import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from classeval.eval.metrics import confusion, exact_and_within_one, kendall_tau
from classeval.eval.thresholds import fit_cutoffs, grade_of, loo_predict
from classeval.schema import Report

LABEL_MAP: dict[str, int | str] = {
    "excellent": 3,
    "good": 2,
    "average": 1,
    "weak": 0,
    "poor": 0,
    "quality_issue": "soft",
    "quality issue": "soft",
    "demo": "demo",
}


def _parse_folder_name(name: str) -> tuple[int | str | None, str, str]:
    """Parse folder name like 'excellent__B491' into (label, truth_key, id)."""
    if "__" not in name:
        return None, "", name
    prefix, batch = name.split("__", 1)
    label = LABEL_MAP.get(prefix.lower())
    truth_prefix = prefix.replace("_", " ")
    truth_key = f"{truth_prefix}/{batch}"
    return label, truth_key, name


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation harness, LOO validation, and cutoffs fitting.")
    parser.add_argument("--runs", required=True, type=Path, help="Directory containing eval run outputs")
    parser.add_argument("--truth", required=False, type=Path, default=None, help="YAML file with checklist truth")
    parser.add_argument("--out", required=False, type=Path, default=None, help="Output path for fitted cutoffs JSON")
    parser.add_argument("--fixed", action="store_true", help="Score with the cutoffs in config.yaml instead of fitting")
    parser.add_argument("--summary", type=Path, default=None, help="Write the aggregate numbers to this JSON file")
    args = parser.parse_args()

    runs_dir = args.runs.resolve()
    if not runs_dir.exists():
        sys.exit(1)

    hard_runs: list[dict] = []
    soft_runs: list[dict] = []
    skipped_count = 0

    subdirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    for sub in subdirs:
        report_file = sub / "report.json"
        if not report_file.exists():
            continue

        label_kind, truth_key, class_id = _parse_folder_name(sub.name)
        if label_kind is None or label_kind == "demo":
            continue

        try:
            rep = Report.model_validate_json(report_file.read_text(encoding="utf-8"))
        except (ValueError, KeyError, OSError):
            continue

        if rep.grade == "Insufficient evidence" or rep.overall_score is None:
            skipped_count += 1
            continue

        entry = {
            "id": class_id,
            "truth_key": truth_key,
            "score": float(rep.overall_score),
            "confidence": float(rep.confidence),
            "dimensions": {d.id: d.score for d in rep.dimensions},
        }

        if isinstance(label_kind, int):
            entry["label"] = label_kind
            hard_runs.append(entry)
        elif label_kind == "soft":
            soft_runs.append(entry)

    # 1. Fit cutoffs & leave-one-out predictions
    hard_scores = [r["score"] for r in hard_runs]
    hard_labels = [r["label"] for r in hard_runs]
    soft_scores = [r["score"] for r in soft_runs]

    if args.fixed:
        configured = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))["cutoffs"]
        cutoffs = (float(configured["poor_lt"]), float(configured["average_lt"]), float(configured["good_lt"]))
        loo_preds = [grade_of(s, cutoffs) for s in hard_scores]
    else:
        cutoffs = fit_cutoffs(hard_scores, hard_labels, soft_scores)
        loo_preds = loo_predict(hard_scores, hard_labels, soft_scores)

    # 2. Compute metrics
    exact, within_one = exact_and_within_one(hard_labels, loo_preds)
    tau = kendall_tau(hard_scores, hard_labels)
    conf_mat = confusion(hard_labels, loo_preds, k=4)

    # 3. Print ONLY numbers
    # Per-class run lines: id, true_label, score, predicted_label, confidence
    for r, pred in zip(hard_runs, loo_preds, strict=False):
        print(f"{r['id']}\t{r['label']}\t{r['score']:.1f}\t{pred}\t{r['confidence']:.2f}")

    print(f"skipped: {skipped_count}")
    print(f"exact: {exact:.4f}\twithin_one: {within_one:.4f}")
    print(f"kendall_tau: {tau:.4f}")

    # Confusion matrix (rows true, cols pred)
    print("confusion_matrix:")
    for row in conf_mat:
        print("\t".join(str(val) for val in row))

    # Soft classes status (must be Average or below -> score < c2, grade <= 1)
    soft_ok = 0
    for r in soft_runs:
        pred_g = grade_of(r["score"], cutoffs)
        status = 1 if pred_g <= 1 else 0
        soft_ok += status
        print(f"{r['id']}\tsoft\t{r['score']:.1f}\t{pred_g}\t{status}")

    # Cutoffs
    print(f"cutoffs:\t{cutoffs[0]:.1f}\t{cutoffs[1]:.1f}\t{cutoffs[2]:.1f}")

    # Save cutoffs JSON if requested
    if args.out:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cutoffs_data = {
            "poor_lt": cutoffs[0],
            "average_lt": cutoffs[1],
            "good_lt": cutoffs[2],
        }
        out_path.write_text(json.dumps(cutoffs_data, indent=2), encoding="utf-8")

    agreement: dict[str, dict[str, float | int]] = {}
    # 4. Per-dimension agreement with ground truth if provided
    if args.truth and args.truth.exists():
        try:
            truth_content = yaml.safe_load(args.truth.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            truth_content = {}

        if truth_content:
            all_runs = hard_runs + soft_runs
            dim_ids = [f"R{i}" for i in range(1, 13)]
            print("dimension_agreement:")
            for dim in dim_ids:
                count = 0
                agree = 0
                for r in all_runs:
                    t_key = r.get("truth_key")
                    if t_key in truth_content and dim in truth_content[t_key]:
                        expected_bool = bool(truth_content[t_key][dim])
                        dim_score = r["dimensions"].get(dim)
                        if dim_score is not None:
                            pred_bool = dim_score >= 60.0
                            count += 1
                            if pred_bool == expected_bool:
                                agree += 1
                frac = round(float(agree) / float(count), 4) if count > 0 else 0.0
                print(f"{dim}\t{count}\t{frac:.4f}")
                agreement[dim] = {"classes": count, "agreement": frac}

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "mode": "configured_cutoffs" if args.fixed else "leave_one_out_fit",
            "labelled_classes": len(hard_runs),
            "soft_labelled_classes": len(soft_runs),
            "exact": exact,
            "within_one": within_one,
            "kendall_tau": tau,
            "confusion_rows_true_cols_predicted": conf_mat,
            "soft_classes_at_or_below_average": soft_ok,
            "cutoffs": {"poor_lt": cutoffs[0], "average_lt": cutoffs[1], "good_lt": cutoffs[2]},
            "dimension_agreement": agreement,
        }
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
