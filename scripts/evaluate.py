#!/usr/bin/env python3
"""Evaluate only the original AnomalyDINO score and the final V11 score."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.common import WORKSPACE, cache_path, feature_key, image_paths, masking_enabled, reference_groups, write_json
from src.coverage_detector import score_features, seeded_mask, unit_features
from src.extractor import official_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-type", type=int)
    args = parser.parse_args()
    if (args.output / "per_reference.csv").exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    data, cache = WORKSPACE / config["data_root"], WORKSPACE / "work/features"
    args.output.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output / "predictions"
    pred_dir.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    masker = object.__new__(official_module("backbones").DINOv2Wrapper)
    masker.layers = None

    def load(path, angle=None):
        key = feature_key(config, path, data, angle)
        file = cache_path(cache, key)
        if not file.exists():
            raise FileNotFoundError(f"Run extract_features.py first: {key}")
        with np.load(file) as z:
            return z["features"], z["mask"], tuple(z["grid"])

    rows = []
    for category in config["categories"]:
        paths = image_paths(data / category / "test", "*/*")
        if args.limit_per_type:
            paths = [p for kind in sorted({p.parent.name for p in paths})
                     for p in [x for x in paths if x.parent.name == kind][:args.limit_per_type]]
        tests = [(p, *load(p)) for p in paths]
        train = image_paths(data / category / "train/good")
        for shot in config["shots"]:
            for group in reference_groups(config, shot):
                references = train[group * shot:(group + 1) * shot]
                if len(references) != shot:
                    raise ValueError("The requested reference budget is unavailable")
                arrays, keep = [], []
                for path in references:
                    for angle in range(0, 360, 45):
                        features, _, grid = load(path, angle)
                        key = feature_key(config, path, data, angle) + ":reference_mask"
                        keep.append(seeded_mask(masker, features, grid, masking_enabled(config, category), key))
                        arrays.append(features)
                bank = torch.as_tensor(unit_features(np.concatenate(arrays)), device=args.device)
                reference_keep = torch.as_tensor(np.concatenate(keep), device=args.device)
                baseline, coverage = [], []
                for _, features, mask, _ in tests:
                    result = score_features(features, mask, bank, reference_keep, config["beta"], config["top_fraction"])
                    baseline.append(result["forward_score"])
                    coverage.append(result["score"])
                labels = np.array([int(p.parent.name != "good") for p in paths])
                kinds = np.array([p.parent.name for p in paths])
                ref_names = [str(p.relative_to(data)) for p in references]
                np.savez_compressed(
                    pred_dir / f"{category}_shot{shot}_group{group}.npz",
                    samples=np.array([str(p.relative_to(data)) for p in paths]), labels=labels,
                    defect_types=kinds, baseline=baseline, coverage=coverage,
                    reference_files=np.array(ref_names),
                )
                tasks = ["all"]
                if "logical_anomalies" in kinds:
                    tasks += ["logical_anomalies", "structural_anomalies"]
                for task in tasks:
                    selected = np.ones(len(labels), bool) if task == "all" else (kinds == "good") | (kinds == task)
                    base = 100 * roc_auc_score(labels[selected], np.array(baseline)[selected])
                    improved = 100 * roc_auc_score(labels[selected], np.array(coverage)[selected])
                    rows.append({
                        "dataset": config["result_dataset"], "task": task, "category": category,
                        "shot": shot, "reference_group": group, "reference_files": json.dumps(ref_names),
                        "baseline_auc_pct": base, "coverage_auc_pct": improved, "delta_pp": improved - base,
                    })
                print(f"DONE {category}, {shot}-shot, group {group}", flush=True)
    with (args.output / "per_reference.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "config.json", config)
    write_json(args.output / "run_info.json", {"version": 11, "subset_test": bool(args.limit_per_type), "device": args.device})


if __name__ == "__main__":
    main()
