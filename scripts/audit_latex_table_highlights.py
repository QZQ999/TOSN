"""Audit whether LaTeX result tables highlight exactly the lowest three cells."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NUMBER = re.compile(r"([0-9]+\.[0-9]+)")


def table_block(text: str, label: str) -> str:
    label_position = text.index(r"\label{" + label + "}")
    start = text.rfind(r"\begin{table}", 0, label_position)
    end = text.index(r"\end{table}", label_position)
    if start < 0:
        raise ValueError(f"table start not found for {label}")
    return text[start:end]


def parse_rows(block: str) -> list[dict]:
    rows: list[dict] = []
    base_model = ""
    for raw in block.splitlines():
        if "& MAE" not in raw and "& RMSE" not in raw:
            continue
        cells = [cell.strip() for cell in raw.split("&")]
        first = cells[0]
        if first and "U-GMA" not in first:
            base_model = first
            model = base_model
        elif "U-GMA" in first:
            model = f"{base_model}+U-GMA"
        else:
            model = base_model
        values = []
        for cell in cells[2:10]:
            match = NUMBER.search(cell)
            if match is None:
                raise ValueError(f"numeric cell not found in row: {raw}")
            values.append({"value": float(match.group(1)), "highlighted": r"\cellcolor" in cell})
        rows.append({"model": model, "metric": cells[1], "values": values})
    return rows


def audit_table(text: str, label: str) -> dict:
    rows = parse_rows(table_block(text, label))
    report = {"label": label, "row_count": len(rows), "metrics": {}, "mismatches": []}
    for metric in ("MAE", "RMSE"):
        metric_rows = [row for row in rows if row["metric"] == metric]
        counts = []
        for column in range(8):
            ranked = sorted(
                range(len(metric_rows)),
                key=lambda index: (metric_rows[index]["values"][column]["value"], index),
            )
            expected = ranked[:3]
            observed = [
                index
                for index, row in enumerate(metric_rows)
                if row["values"][column]["highlighted"]
            ]
            counts.append(len(observed))
            if set(expected) != set(observed):
                report["mismatches"].append(
                    {
                        "metric": metric,
                        "column_1_based": column + 1,
                        "expected": [
                            {
                                "model": metric_rows[index]["model"],
                                "value": metric_rows[index]["values"][column]["value"],
                            }
                            for index in expected
                        ],
                        "observed": [metric_rows[index]["model"] for index in observed],
                    }
                )
        report["metrics"][metric] = {"highlight_counts": counts}
    report["valid"] = not report["mismatches"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["tab:main-etth-ettm", "tab:main-boiler-iot"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = args.tex.read_text(encoding="utf-8")
    reports = [audit_table(text, label) for label in args.labels]
    payload = {"tables": reports, "valid": all(report["valid"] for report in reports)}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if payload["valid"] else 1)


if __name__ == "__main__":
    main()
