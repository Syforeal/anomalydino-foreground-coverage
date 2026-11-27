"""Version 11 only: original forward score plus foreground reverse coverage."""
import hashlib
import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

from src.common import masking_enabled, write_json
from src.extractor import load_extractor, official_module


def unit_features(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(features / norms)


def top_mean(values, fraction=0.01):
    if not len(values):
        return values.new_tensor(0.0)
    return values.topk(max(1, int(len(values) * fraction))).values.mean()


def distance_values(query, bank, reverse=False, chunk_size=1024):
    """Exact squared L2; optional reverse minima reuse the same distance blocks."""
    if not len(bank):
        raise ValueError("The normal reference bank is empty")
    if not len(query):
        return query.new_empty((0,)), bank.new_full((len(bank),), 4.0) if reverse else None
    norms = (bank * bank).sum(1)
    forward = []
    backward = bank.new_full((len(bank),), float("inf")) if reverse else None
    for q in query.split(chunk_size):
        distances = (q * q).sum(1, keepdim=True) + norms[None] - 2 * q @ bank.T
        distances.clamp_min_(0)
        forward.append(distances.amin(1))
        if reverse:
            backward = torch.minimum(backward, distances.amin(0))
    return torch.cat(forward), backward


def seeded_mask(wrapper, features, grid, enabled, key):
    previous = np.random.get_state()
    np.random.seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    try:
        return wrapper.compute_background_mask(features, grid, masking_type=enabled)
    finally:
        np.random.set_state(previous)


def score_features(features, mask, bank, reference_keep, beta=0.25, fraction=0.01, baseline_only=False):
    """Return image scores and the unchanged forward patch map.

    The forward bank is complete. Only the reverse direction filters reference
    background. Query-background entries remain zero in forward aggregation.
    """
    query = torch.as_tensor(unit_features(features[mask]), device=bank.device)
    forward, backward = distance_values(query, bank, reverse=not baseline_only)
    patch = forward.new_zeros(len(features))
    patch[torch.as_tensor(mask, device=bank.device)] = forward / 2
    f_score = top_mean(patch, fraction)
    r_score = f_score.new_tensor(0.0)
    if not baseline_only:
        keep = torch.as_tensor(reference_keep, dtype=torch.bool, device=bank.device)
        r_score = top_mean(backward[keep] / 2, fraction)
    return {
        "score": float(f_score + beta * r_score),
        "forward_score": float(f_score),
        "reverse_score": float(r_score),
        "patch_map": patch.cpu().numpy(),
    }


class CoverageDetector:
    def __init__(self, bank_dir, dino_repo, weights, device="cuda:0", head="coverage"):
        if head not in {"baseline", "coverage"}:
            raise ValueError("head must be baseline or coverage")
        bank_dir = Path(bank_dir)
        self.metadata = json.loads((bank_dir / "metadata.json").read_text())
        actual = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if actual != self.metadata["weight_sha256"]:
            raise ValueError("Backbone weight hash mismatch")
        self.wrapper = load_extractor(dino_repo, weights, self.metadata["resolution"], device)
        index = faiss.read_index(str(bank_dir / "index.faiss"))
        self.bank = torch.as_tensor(index.reconstruct_n(0, index.ntotal), device=device)
        self.reference_keep = torch.as_tensor(np.load(bank_dir / "reference_mask.npy"), device=device)
        self.head = head

    def predict(self, path, data_root):
        path, data_root = Path(path), Path(data_root)
        relative = str(path.resolve().relative_to(data_root.resolve()))
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        tensor, grid = self.wrapper.prepare_image(image)
        features = self.wrapper.extract_features(tensor)
        key = self.metadata["cache_namespace"] + "/" + relative
        mask = seeded_mask(self.wrapper, features, grid, self.metadata["query_masking"], key)
        result = score_features(
            features, mask, self.bank, self.reference_keep,
            self.metadata["beta"], self.metadata["reverse_top_fraction"], self.head == "baseline",
        )
        result["patch_map"] = result["patch_map"].reshape(grid)
        result["head"] = self.head
        return result


def build_coverage_bank(references, output, data_root, dino_repo, weights, config, category, device="cuda:0"):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(output)
    wrapper = load_extractor(dino_repo, weights, config["resolution"], device)
    utils = official_module("utils")
    enabled = masking_enabled(config, category)
    arrays, masks, views = [], [], []
    for path in references:
        path = Path(path)
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        relative = str(path.resolve().relative_to(Path(data_root).resolve()))
        for angle in range(0, 360, 45):
            rotated = utils.rotate_image(image, angle)
            tensor, grid = wrapper.prepare_image(rotated)
            features = wrapper.extract_features(tensor)
            key = config["cache_namespace"] + "/" + relative + f":angle={angle}:reference_mask"
            masks.append(seeded_mask(wrapper, features, grid, enabled, key))
            arrays.append(unit_features(features))
            views.append({"path": relative, "angle": angle, "grid": grid, "patches": len(features)})
    index = faiss.IndexFlatL2(384)
    index.add(np.concatenate(arrays))
    faiss.write_index(index, str(output / "index.faiss"))
    np.save(output / "reference_mask.npy", np.concatenate(masks))
    metadata = {
        "version": 11, "model": "dinov2_vits14", "resolution": config["resolution"],
        "beta": config["beta"], "reverse_top_fraction": config["top_fraction"],
        "query_masking": bool(enabled), "views": views,
        "cache_namespace": config["cache_namespace"],
        "weight_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
    }
    write_json(output / "metadata.json", metadata)
    return metadata
