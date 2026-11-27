"""Offline official DINOv2 preprocessing and frozen feature extraction."""
import importlib.util
from pathlib import Path
import torch

PROJECT = Path(__file__).resolve().parents[1]


def official_module(name):
    path = PROJECT/"vendor/AnomalyDINO/src"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"original_{name}",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_extractor(repo, weights, resolution=448, device="cuda:0"):
    model = torch.hub.load(str(Path(repo).resolve()), "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    backbones = official_module("backbones")
    class OfflineDINO(backbones.DINOv2Wrapper):
        def load_model(self):
            self.patch_size = model.patch_size
            self.transform = self._dino_transform()
            return model.eval().to(self.device)
    return OfflineDINO("dinov2_vits14", device, smaller_edge_size=resolution)
