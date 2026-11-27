#!/usr/bin/env python3
"""Extract V11 features for selected normal references and the test images."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.common import WORKSPACE, cache_path, feature_key, image_paths, masking_enabled, reference_groups, write_json
from src.coverage_detector import seeded_mask
from src.extractor import load_extractor, official_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = WORKSPACE / config["data_root"]
    cache = WORKSPACE / "work/features"
    cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wrapper = load_extractor(WORKSPACE / "work/dinov2", WORKSPACE / "work/dinov2_vits14_pretrain.pth", config["resolution"], args.device)
    utils = official_module("utils")
    requests = []
    for category in config["categories"]:
        train = image_paths(data / category / "train/good")
        selected = set()
        for shot in config["shots"]:
            for group in reference_groups(config, shot):
                references = train[group * shot:(group + 1) * shot]
                if len(references) != shot:
                    raise ValueError(f"Not enough normal images: {category}")
                selected.update(references)
        requests.extend((path, angle) for path in sorted(selected) for angle in range(0, 360, 45))
        requests.extend((path, None) for path in image_paths(data / category / "test", "*/*"))
    for i, (path, angle) in enumerate(requests):
        key = feature_key(config, path, data, angle)
        dest = cache_path(cache, key)
        if not dest.exists():
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            if angle is not None:
                image = utils.rotate_image(image, angle)
            tensor, grid = wrapper.prepare_image(image)
            features = wrapper.extract_features(tensor)
            enabled = angle is None and masking_enabled(config, path.relative_to(data).parts[0])
            mask = seeded_mask(wrapper, features, grid, enabled, key)
            temporary = dest.with_suffix(".tmp.npz")
            np.savez(temporary, features=features, grid=np.array(grid), mask=mask)
            temporary.replace(dest)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(requests)}", flush=True)
    write_json(args.output / "feature_manifest.json", {
        "version": 11, "cache_namespace": config["cache_namespace"], "requests": len(requests),
        "records": [{"path": str(p.relative_to(data)), "angle": a} for p, a in requests],
    })
    write_json(args.output / "config.json", config)


if __name__ == "__main__":
    main()
