#!/usr/bin/env python3
"""Summarize V11 reference-level AUCs using equal class and reference weights."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"], int(row["shot"]))].append(row)
    summary = []
    for (dataset, task, shot), selected in sorted(groups.items()):
        record = {"dataset": dataset, "task": task, "shot": shot}
        for method in ["baseline", "coverage"]:
            column = method + "_auc_pct"
            reference_means = [np.mean([float(r[column]) for r in selected if r["reference_group"] == group])
                               for group in sorted({r["reference_group"] for r in selected})]
            record[column] = float(np.mean(reference_means))
            record[method + "_reference_sd"] = float(np.std(reference_means, ddof=1)) if len(reference_means) > 1 else 0.0
        record["delta_pp"] = record["coverage_auc_pct"] - record["baseline_auc_pct"]
        summary.append(record)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(list(csv.DictReader(args.input.open())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
