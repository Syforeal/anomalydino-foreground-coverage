"""Dataset paths and deterministic feature-cache helpers."""
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKSPACE = PROJECT.parents[1] if PROJECT.parent.name == "outputs" else PROJECT
WORKSPACE = Path(os.environ.get("ANOMALYDINO_WORKSPACE", _DEFAULT_WORKSPACE)).resolve()
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_paths(folder, pattern="*"):
    return sorted(p for p in Path(folder).glob(pattern) if p.suffix.lower() in EXTENSIONS)


def reference_groups(config, shot):
    return config.get("shot_seed_map", {}).get(str(shot), config["seeds"])


def masking_enabled(config, category):
    return config.get("query_masking_by_category", {}).get(category, config.get("query_masking", False))


def feature_key(config, path, data_root, angle=None):
    key = config["cache_namespace"] + "/" + str(Path(path).relative_to(data_root))
    return key if angle is None else key + f":angle={angle}"


def cache_path(cache, key):
    return Path(cache) / (hashlib.sha256(key.encode()).hexdigest() + ".npz")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
