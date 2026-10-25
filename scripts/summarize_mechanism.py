"""Export audit tables from the two independently generated diagnostics."""
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1] / "results/mechanism_20260909"
    analysis = json.loads((root / "prediction_analysis.json").read_text(encoding="utf-8"))
    replay = json.loads((root / "retrieval_replay.json").read_text(encoding="utf-8"))
    assert analysis["source_sha256"] == replay["source_sha256"]
    assert analysis["n_runs"] == 80 and len(replay["checks"]) == 20
    write_csv(root / "prediction_groups.csv", analysis["records"])
    write_csv(root / "prediction_summary.csv", analysis["summary"])
    write_csv(root / "replay_groups.csv", replay["records"])
    groups = {}
    for row in replay["records"]:
        key = (row["dataset"], row["model"], row["quality"], row["reference_group"])
        groups.setdefault(key, []).append(row)
    summary = []
    metrics = ["reference_better_fraction", "full_gain_pct", "local_only_gain_pct",
               "retrieval_only_gain_pct", "retrieval_marginal_mae", "local_marginal_mae",
               "local_magnitude", "retrieval_magnitude", "similarity", "beta"]
    for key, rows in sorted(groups.items()):
        item = dict(zip(["dataset", "model", "quality", "reference_group"], key))
        item["runs"] = len(rows)
        item["units"] = sum(r["units"] for r in rows)
        item.update({m: statistics.mean(r[m] for r in rows) for m in metrics})
        summary.append(item)
    write_csv(root / "replay_summary.csv", summary)
    report = {"replay_summary": summary,
              "max_enhanced_replay_discrepancy": max(r["max_abs_replay_error"]["enhanced"] for r in replay["checks"]),
              "note": "Equal-seed descriptive means, not independent-window statistical tests"}
    (root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
