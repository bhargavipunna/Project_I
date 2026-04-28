"""
core/reid/embedder.py
STEP 4a — Appearance Embedder using OSNet (TorchReID)

Converts a person crop image into a 512-dimensional appearance
embedding vector. This vector is a numerical "fingerprint" of
how the person looks — same person = similar vectors.

How it works:
  - OSNet (Omni-Scale Network) is trained specifically for
    person re-identification tasks
  - Input : BGR crop of a person (any size)
  - Output: 512-dimensional float32 numpy array (L2 normalised)

Why L2 normalise?
  - After normalisation, similarity = cosine similarity = dot product
  - FAISS can then use fast inner-product search
  - All vectors have length 1.0, so distance is purely about direction

Design decisions:
  - Lazy model load — model only downloaded on first use
  - Batch support — can embed multiple crops at once (faster on GPU)
  - Falls back to torchreid OSNet if available, else uses a
    lightweight CNN fallback so the system still runs without torchreid
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from loguru import logger

from config.settings import OSNET_MODEL, EMBEDDING_DIM, YOLO_DEVICE

# Standard input size for OSNet
OSNET_INPUT_SIZE = (256, 128)   # (height, width) — standard Re-ID input


class PersonEmbedder:
    """
    Converts person crop images → 512-dim appearance embeddings.

    Usage:
        embedder = PersonEmbedder()
        vector   = embedder.embed(crop)          # single crop
        vectors  = embedder.embed_batch(crops)   # list of crops (faster)
    """

    def __init__(
        self,
        model_name : str = OSNET_MODEL,
        device     : str = YOLO_DEVICE,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.model_name    = model_name
        self.device        = device
        self.embedding_dim = embedding_dim
        self._model        = None
        self._transform    = None
        self._use_torchreid = False

        self._load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Embed a single person crop.

        Args:
            crop: BGR image of a person (from TrackedPerson.crop)

        Returns:
            numpy array of shape (512,) — L2 normalised float32
            None if crop is invalid
        """
        if crop is None or crop.size == 0:
            return None

        if crop.shape[0] < 10 or crop.shape[1] < 10:
            logger.debug("Crop too small to embed — skipping.")
            return None

        batch = self.embed_batch([crop])
        return batch[0] if batch else None

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Embed multiple crops at once (more efficient than one-by-one).

        Args:
            crops: list of BGR person crops

        Returns:
            list of (512,) numpy arrays — same order as input
        """
        if not crops:
            return []

        tensors = []
        valid_indices = []

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            t = self._preprocess(crop)
            if t is not None:
                tensors.append(t)
                valid_indices.append(i)

        if not tensors:
            return []

        batch_tensor = torch.cat(tensors, dim=0).to(self.device)

        with torch.no_grad():
            features = self._model(batch_tensor)
            # L2 normalise
            features = F.normalize(features, p=2, dim=1)

        embeddings = features.cpu().numpy()

        # Return in same order as input — None for invalid crops
        result = [None] * len(crops)
        for out_idx, in_idx in enumerate(valid_indices):
            result[in_idx] = embeddings[out_idx]

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load OSNet via torchreid if available, else lightweight fallback."""
        try:
            import torchreid
            import os, torch

            # Build OSNet with Market-1501 classes
            self._model = torchreid.models.build_model(
                name        = self.model_name,
                num_classes = 751,
                loss        = 'softmax',
                pretrained  = False,   # we load weights manually below
            )

            # Try to load Market-1501 Re-ID weights (downloaded or cached)
            weight_path = "data/osnet_x0_25_market.pth"
            if os.path.exists(weight_path):
                state = torch.load(weight_path, map_location="cpu")
                # handle different checkpoint formats
                state_dict = state.get("state_dict", state.get("model", state))
                # remove classifier head — we only need feature extractor
                state_dict = {k: v for k, v in state_dict.items()
                              if "classifier" not in k}
                missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
                logger.success(
                    f"OSNet loaded with Market-1501 Re-ID weights: {weight_path}"
                )
            else:
                # Fall back to ImageNet weights — lower Re-ID accuracy
                # but still much better than MobileNetV2
                torchreid.utils.load_pretrained_weights(
                    self._model, weight_path
                ) if False else None
                # Load ImageNet pretrained
                model_imagenet = torchreid.models.build_model(
                    name=self.model_name, num_classes=751,
                    loss='softmax', pretrained=True,
                )
                # Copy backbone weights only (no classifier)
                backbone_state = {k: v for k, v
                                  in model_imagenet.state_dict().items()
                                  if "classifier" not in k}
                self._model.load_state_dict(backbone_state, strict=False)
                logger.warning(
                    f"Market-1501 weights not found at {weight_path}. "
                    f"Using ImageNet backbone — similarity threshold should be ~0.50. "
                    f"Download Re-ID weights for better accuracy: "
                    f"python -c \"import gdown; gdown.download("
                    f"'https://drive.google.com/uc?id=1rb8UN5ZzPKEndkgbi9qqElafLlbCZziV', "
                    f"'{weight_path}', quiet=False)\""
                )

            self._model.eval()
            self._model.to(self.device)
            self._use_torchreid = True
            logger.success(
                f"OSNet active: {self.model_name} — REAL Re-ID running"
            )

        except ImportError:
            logger.warning(
                "torchreid not installed — using MobileNetV2 fallback.\n"
                "Install: pip install torchreid"
            )
            self._load_mobilenet_fallback()

        except Exception as e:
            logger.error(f"OSNet load error: {e}")
            logger.warning("Falling back to MobileNetV2.")
            self._load_mobilenet_fallback()

    def _load_mobilenet_fallback(self):
        """
        MobileNetV2 fallback when torchreid is unavailable.
        Uses torchvision pretrained model — always available.
        Output is projected to EMBEDDING_DIM via a linear layer.
        """
        import torchvision.models as models
        import torch.nn as nn

        base = models.mobilenet_v2(weights="IMAGENET1K_V1")
        # Remove classifier, keep feature extractor
        self._mobilenet_features = nn.Sequential(*list(base.children())[:-1])
        self._mobilenet_features.eval()
        self._mobilenet_features.to(self.device)

        # Project 1280-dim MobileNet output → EMBEDDING_DIM
        self._projector = nn.Linear(1280, self.embedding_dim, bias=False)
        self._projector.eval()
        self._projector.to(self.device)

        # Wrap both into a single callable
        class _FallbackModel(torch.nn.Module):
            def __init__(self, features, projector):
                super().__init__()
                self.features  = features
                self.projector = projector
            def forward(self, x):
                f = self.features(x)
                f = f.mean([2, 3])   # global average pool
                return self.projector(f)

        self._model = _FallbackModel(self._mobilenet_features, self._projector)
        self._model.eval()
        self._model.to(self.device)
        logger.info("MobileNetV2 fallback embedder ready.")

    def _preprocess(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """
        Convert BGR crop → normalised tensor ready for the model.
        Standard ImageNet normalisation (works for both OSNet and MobileNet).
        """
        try:
            # Resize to standard Re-ID input
            img = cv2.resize(crop, (OSNET_INPUT_SIZE[1], OSNET_INPUT_SIZE[0]))
            # BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # HWC → CHW, normalise to [0,1]
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            # ImageNet normalisation
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img  = (img - mean[:, None, None]) / std[:, None, None]
            tensor = torch.from_numpy(img).unsqueeze(0)   # → (1, 3, H, W)
            return tensor
        except Exception as e:
            logger.debug(f"Preprocess failed: {e}")
            return None


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from core.video.stream_reader import StreamReader
    from core.tracking.person_tracker import PersonTracker

    source = sys.argv[1] if len(sys.argv) > 1 else "data/cam1.mp4"
    print(f"\nStep 4a — Embedder test on: {source}")
    print("Prints embedding vector stats for each tracked person.\n")

    embedder = PersonEmbedder()
    tracker  = PersonTracker()
    reader   = StreamReader(source=source, frame_skip=2)

    try:
        for frame_num, frame in reader.frames():
            tracked = tracker.track(frame, camera_id="cam1", frame_num=frame_num)

            if frame_num % 30 == 0 and tracked:
                print(f"\nFrame #{frame_num}")
                for p in tracked:
                    vec = embedder.embed(p.crop)
                    if vec is not None:
                        print(f"  ID:{p.track_id:3d} | "
                              f"embedding shape={vec.shape} | "
                              f"norm={np.linalg.norm(vec):.4f} | "
                              f"mean={vec.mean():.4f}")

            if frame_num > 300:
                break

    finally:
        reader.stop()
        print("\nDone. Embedding norm should be ~1.0 (L2 normalised).")