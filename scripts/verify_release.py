#!/usr/bin/env python3
"""Check the final scoring formula, retained metrics and optional local-cache examples."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.common import WORKSPACE, cache_path, feature_key, image_paths, masking_enabled, write_json
from src.coverage_detector import distance_values, score_features, seeded_mask, unit_features
from src.extractor import official_module

DATASETS = {"mvtec_ad": "MVTec AD", "visa": "VisA", "loco": "LOCO original refs", "loco_reference_audit": "LOCO new refs"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    generator = torch.Generator().manual_seed(11)
    q = torch.nn.functional.normalize(torch.randn(7, 6, generator=generator), dim=1)
    bank = torch.nn.functional.normalize(torch.randn(9, 6, generator=generator), dim=1)
    f, r = distance_values(q, bank, reverse=True, chunk_size=3)
    explicit = ((q[:, None] - bank[None]) ** 2).sum(-1)
    assert torch.allclose(f, explicit.amin(1), atol=1e-6)
    assert torch.allclose(r, explicit.amin(0), atol=1e-6)
    basis = torch.eye(3)
    result = score_features(np.array([[1., 0., 0.]], np.float32), np.array([True]), basis[:2], np.array([True, False]))
    assert result["score"] == 0.0  # Reference background alone must not raise the score.
    empty = score_features(np.eye(3, dtype=np.float32), np.zeros(3, bool), basis, np.ones(3, bool))
    assert empty["forward_score"] == 0 and empty["score"] == 0.5
    records = list(csv.DictReader((PROJECT / "results/per_reference.csv").open()))
    expected = {(r["dataset"], r["task"], r["category"], int(r["shot"]), int(r["reference_group"])): r for r in records}
    checked = 0
    for slug, label in DATASETS.items():
        for path in sorted((PROJECT / "results/predictions" / slug).glob("*.npz")):
            match = re.fullmatch(r"(.*)_shot(\d+)_group(\d+)\.npz", path.name)
            category, shot, group = match.group(1), int(match.group(2)), int(match.group(3))
            with np.load(path) as z:
                tasks = ["all"] + (["logical_anomalies", "structural_anomalies"] if slug.startswith("loco") else [])
                for task in tasks:
                    keep = np.ones(len(z["labels"]), bool) if task == "all" else (z["defect_types"] == "good") | (z["defect_types"] == task)
                    row = expected[(label, task, category, shot, group)]
                    for method in ["baseline", "coverage"]:
                        actual = 100 * roc_auc_score(z["labels"][keep], z[method][keep])
                        assert abs(actual - float(row[method + "_auc_pct"])) < 1e-9
                    checked += 1
    assert checked == len(records)
    main_rows = list(csv.DictReader((PROJECT / "results/main_results.csv").open()))
    for row in main_rows:
        selected = [r for r in records if r["dataset"] == row["dataset"] and r["task"] == row["task"] and r["shot"] == row["shot"]]
        for method in ["baseline", "coverage"]:
            actual = np.mean([float(r[method + "_auc_pct"]) for r in selected])
            assert abs(actual - float(row[method + "_auc_pct"])) < 1e-9
    for row in csv.DictReader((PROJECT / "results/per_category.csv").open()):
        selected = [r for r in records if r["dataset"] == row["dataset"] and r["task"] == row["task"] and r["category"] == row["category"] and r["shot"] == row["shot"]]
        for method in ["baseline", "coverage"]:
            actual = np.mean([float(r[method + "_auc_pct"]) for r in selected])
            assert abs(actual - float(row[method + "_auc_pct"])) < 1e-9
    cached = []
    if args.cached_check:
        masker = object.__new__(official_module("backbones").DINOv2Wrapper)
        masker.layers = None
        for name, slug, category in [("mvtec", "mvtec_ad", "cable"), ("visa", "visa", "capsules")]:
            cfg = json.loads((PROJECT / "configs" / (name + ".json")).read_text())
            data = WORKSPACE / cfg["data_root"]
            reference = image_paths(data / category / "train/good")[0]
            arrays, masks = [], []
            for angle in range(0, 360, 45):
                key = feature_key(cfg, reference, data, angle)
                with np.load(cache_path(WORKSPACE / "work/features", key)) as z:
                    features, grid = z["features"], tuple(z["grid"])
                arrays.append(features)
                masks.append(seeded_mask(masker, features, grid, masking_enabled(cfg, category), key + ":reference_mask"))
            bank = torch.as_tensor(unit_features(np.concatenate(arrays)))
            reference_keep = np.concatenate(masks)
            with np.load(PROJECT / "results/predictions" / slug / f"{category}_shot1_group0.npz") as predictions:
                indices = [np.flatnonzero(predictions["labels"] == label)[0] for label in [0, 1]]
                for i in indices:
                    sample = str(predictions["samples"][i])
                    with np.load(cache_path(WORKSPACE / "work/features", feature_key(cfg, data / sample, data))) as z:
                        actual = score_features(z["features"], z["mask"], bank, reference_keep)
                    error = abs(actual["score"] - float(predictions["coverage"][i]))
                    assert error < 3e-5, (sample, error)
                    cached.append({"dataset": name, "sample": sample, "absolute_error": error})
    result = {"status": "pass", "version": 11, "formula_and_edge_cases": True, "auc_rows_recomputed": checked, "summary_tables_checked": True,
              "cached_code_checks": cached, "note": "Retained measurements are unchanged. This cleanup verifies consistency; it is not a new full benchmark or latency measurement."}
    write_json(PROJECT / "results/cleanup_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
