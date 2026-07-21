"""
embedder.py
-----------
Generates appearance embeddings for person crops.

Backend priority (configurable in configs/config.yaml -> reid.backend):
    1. "torchreid_osnet"      - OSNet via the `torchreid` package (best
                                 accuracy/speed trade-off for person ReID;
                                 purpose-built for this exact task).
    2. "torchvision_resnet50" - Always-available fallback (ImageNet-pretrained
                                 ResNet50 penultimate layer as a general
                                 appearance descriptor). Lower accuracy than a
                                 real ReID model but has zero extra
                                 dependencies, so the POC still runs if
                                 torchreid isn't installed.

Swapping in FastReID / TransReID later only requires adding a new branch
in `_load_backend()` — everything downstream consumes plain numpy vectors.
"""

from __future__ import annotations

from typing import List

import numpy as np

from backend.utils.colab_setup import device
from backend.utils.config_loader import get_config


class ReIDEmbedder:
    def __init__(self, config: dict | None = None):
        cfg = config or get_config()
        self.cfg = cfg["reid"]
        self.runtime_cfg = cfg["runtime"]
        self._device = device(self.runtime_cfg.get("device", "auto"))
        self._backend_name = None
        self._model = None
        self._preprocess = None

    # ------------------------------------------------------------------ #
    # Backend loading
    # ------------------------------------------------------------------ #
    def _load_backend(self):
        if self._model is not None:
            return

        requested = self.cfg["backend"]
        if requested == "torchreid_osnet":
            try:
                self._load_torchreid_osnet()
                self._backend_name = "torchreid_osnet"
                return
            except Exception as e:  # noqa: BLE001
                print(
                    f"[embedder] torchreid backend unavailable ({e}); "
                    "falling back to torchvision ResNet50 appearance embedding. "
                    "Install `torchreid` for proper person-ReID accuracy."
                )
        self._load_torchvision_resnet50()
        self._backend_name = "torchvision_resnet50"

    def _load_torchreid_osnet(self):
        import torch
        import torchreid

        self._model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,  # unused at inference; we take the feature layer
            pretrained=True,
        )
        self._model.eval().to(self._device)

        import torchvision.transforms as T

        self._preprocess = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._torch = torch

    def _load_torchvision_resnet50(self):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        base.fc = torch.nn.Identity()  # strip classifier -> 2048-d feature vector
        self._model = base.eval().to(self._device)

        self._preprocess = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalized embedding vector for a single person crop."""
        return self.embed_batch([crop_bgr])[0]

    def embed_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Return an (N, D) L2-normalized embedding matrix. Empty/degenerate
        crops (zero height or width, which can happen from noisy boxes near
        frame edges) are embedded as zero vectors and should be filtered by
        the caller if precision matters.
        """
        self._load_backend()
        import cv2

        valid_idx = []
        tensors = []
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._preprocess(rgb))
            valid_idx.append(i)

        dim = self.cfg["embedding_dim"] if self._backend_name == "torchreid_osnet" else 2048
        out = np.zeros((len(crops_bgr), dim), dtype=np.float32)
        if not tensors:
            return out

        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad():
            feats = self._model(batch)
        feats = feats.cpu().numpy().astype(np.float32)

        # L2 normalize so cosine similarity == dot product downstream.
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms

        for out_row, i in enumerate(valid_idx):
            out[i] = feats[out_row]
        return out
