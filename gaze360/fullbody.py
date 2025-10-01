# gaze360/fullbody.py
"""
Gaze360 (full-body crop): predicts gaze using the entire person bbox region.

Usage:
    from gaze360.fullbody import Gaze360FullBody

    fullbody = Gaze360FullBody()
    ...
    person = fullbody.detect_gaze(frame, person)   # uses frame[y1:y2, x1:x2]
    frame  = fullbody.draw(frame, person.bbox, person.gaze_vec3)

Config (reads [gaze360] like the face version):
  GAZE_MODEL=./gaze360/models/gaze360.pkl
  INPUT_SIZE=224
  USE_IMAGENET_NORM=True
  DEVICE=auto
  ARROW_LEN=80
  HEAD_POINT=center      # center|top (anchor for arrow)
  FLIP_X=True
  FLIP_Y=False
  USE_ABS_Z=True
  STRICT_LOAD=False
"""

from __future__ import annotations
import configparser
import logging
import os
from typing import Tuple, Sequence

import cv2
import numpy as np
import torch

logger = logging.getLogger("Gaze360FullBody")


def _imagenet_norm(t: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalization to a [B,3,H,W] tensor in [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device)[None, :, None, None]
    return (t - mean) / std

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-8
    return v / n

def _vec_to_yaw_pitch(vec3: np.ndarray) -> Tuple[float, float]:
    """
    3D unit vector -> (yaw, pitch) degrees; camera coords: z-forward, x-left, y-up.
    yaw   : left/right (positive = left)
    pitch : up/down    (positive = up)
    """
    x, y, z = vec3.astype(np.float32)
    yaw   = np.degrees(np.arctan2(x, z + 1e-8))
    pitch = np.degrees(np.arctan2(-y, np.sqrt(x * x + z * z) + 1e-8))
    return float(yaw), float(pitch)

def _looks_like_cnn_state(state_dict: dict) -> bool:
    """Heuristic to detect a real CNN state_dict vs random/dummy keys."""
    if not isinstance(state_dict, dict) or len(state_dict) < 5:
        return False
    joined = " ".join(state_dict.keys()).lower()
    return any(k in joined for k in ["conv", "layer", "bn", "res", "backbone", "features", "stem"])


class Gaze360FullBody:
    """
    Gaze from full-person crop (person.bbox), not face.

    API:
      - detect_gaze(frame, person)  -> mutates person with .gaze_vec3 / .gaze_yaw_deg / .gaze_pitch_deg
      - draw(frame, bbox, gaze_vec3)-> draws the 2D arrow

    Reads the same [gaze360] section as the face/regular class so you can reuse config.
    """

    def __init__(self):
        cfg = configparser.ConfigParser()
        cfg.read("config.cfg")
        has = cfg.has_section("gaze360")
        def get(key, default):     return cfg["gaze360"].get(key, str(default)) if has else str(default)
        def getint(key, default):  return cfg["gaze360"].getint(key, default)   if has else int(default)
        def getbool(key, default): return cfg["gaze360"].getboolean(key, default) if has else bool(default)

        self.ckpt_path         = get("GAZE_MODEL", "./gaze360/models/gaze360.pkl")
        self.input_size        = getint("INPUT_SIZE", 224)
        self.use_imagenet_norm = getbool("USE_IMAGENET_NORM", True)
        self.arrow_len         = getint("ARROW_LEN", 80)
        self.head_point        = get("HEAD_POINT", "center").lower()
        self.flip_x            = getbool("FLIP_X", True)
        self.flip_y            = getbool("FLIP_Y", False)
        self.use_abs_z         = getbool("USE_ABS_Z", True)
        self.strict_load       = getbool("STRICT_LOAD", False)

        # device
        dev_cfg = get("DEVICE", "auto").lower()
        if dev_cfg == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif dev_cfg == "auto" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"Gaze360 checkpoint not found at {self.ckpt_path}. "
                f"Set [gaze360].GAZE_MODEL in config.cfg or place the file there."
            )
        logger.info("Loading Gaze360 checkpoint (full-body): %s on %s", self.ckpt_path, self.device)

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        if self.strict_load:
            self.model = self._build_or_extract_model_strict(ckpt)
        else:
            self.model = self._build_or_extract_model_lenient(ckpt)
        self.model.to(self.device).eval()
        logger.info("Gaze360FullBody initialized.")

    def _build_or_extract_model_lenient(self, ckpt):
        """Lenient: accept pickled nn.Module, dict['model'] nn.Module, or try load state_dict into a stub."""
        if isinstance(ckpt, torch.nn.Module):
            return ckpt
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), torch.nn.Module):
            return ckpt["model"]

        state_dict = None
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt

        class _Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(3, 3)
            def forward(self, x):
                return self.fc(self.pool(x).flatten(1))

        if state_dict is not None:
            model = _Stub()
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.warning(
                    "Loaded checkpoint into a stub network (lenient). "
                    "Set STRICT_LOAD=True and use the proper model to avoid this."
                )
                return model
            except Exception as e:
                logger.warning("Failed to map state_dict to stub (continuing with stub weights): %s", e)
                return model
        logger.warning(
            "Unrecognized checkpoint format (lenient). Using stub model; results may be poor."
        )
        return _Stub()

    def _build_or_extract_model_strict(self, ckpt):
        """Strict: accept pickled nn.Module/dict['model'] or a CNN-like state_dict mapped to a defined model."""
        if isinstance(ckpt, torch.nn.Module):
            return ckpt
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), torch.nn.Module):
            return ckpt["model"]

        state_dict = None
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt

        if state_dict is not None and _looks_like_cnn_state(state_dict):
            # TODO: Replace DummyNet with your actual Gaze360 architecture
            class DummyNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.AdaptiveAvgPool2d(1),
                    )
                    self.head = torch.nn.Linear(32, 2)  # [yaw, pitch] (radians)
                def forward(self, x):
                    x = self.features(x).flatten(1)
                    return self.head(x)
            model = DummyNet()
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded state_dict (strict) with missing=%d unexpected=%d",
                            len(missing), len(unexpected))
                return model
            except Exception as e:
                raise RuntimeError(
                    "Failed to load state_dict into expected architecture. "
                    "Swap DummyNet with your actual Gaze360 model."
                ) from e

        raise RuntimeError(
            "Unrecognized Gaze360 checkpoint format in STRICT_LOAD mode. "
            "Provide a pickled nn.Module or a compatible state_dict and model class."
        )

    # ---------- inference ----------
    @torch.no_grad()
    def _predict_from_region(self, region_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        if region_bgr is None or region_bgr.size == 0:
            raise ValueError("Empty region for gaze estimation.")

        rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).float() / 255.0              # [H,W,3]
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)        # [1,3,H,W]
        if self.use_imagenet_norm:
            t = _imagenet_norm(t)

        out = torch.atleast_2d(self.model(t))                      # [1,2] or [1,3]

        # Case A: [yaw, pitch] radians -> 3D vec
        if out.shape[-1] == 2:
            yaw, pitch = out[0].tolist()
            gaze_vec = np.array([
                np.sin(yaw) * np.cos(pitch),   # x  (left +)
                -np.sin(pitch),                # y  (up +)
                np.cos(yaw) * np.cos(pitch),   # z  (forward +)
            ], dtype=np.float32)
        else:
            # Case B: direct 3-vector
            gaze_vec = out[0].detach().float().cpu().numpy().astype(np.float32)

        gaze_vec = _unit(gaze_vec)
        yaw_deg, pitch_deg = _vec_to_yaw_pitch(gaze_vec)
        return gaze_vec, yaw_deg, pitch_deg

    @torch.no_grad()
    def detect_gaze(self, frame_bgr: np.ndarray, person):
        """
        Mutates and returns `person` with gaze attributes, using the full person bbox crop.
        """
        # Crop the person box from the frame
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, getattr(person, "bbox", (0, 0, 0, 0)))
        x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid person bbox for full-body gaze crop.")
        person_crop = frame_bgr[y1:y2, x1:x2]

        gaze_vec, yaw_deg, pitch_deg = self._predict_from_region(person_crop)
        setattr(person, "gaze_vec3", gaze_vec)
        setattr(person, "gaze_yaw_deg", yaw_deg)
        setattr(person, "gaze_pitch_deg", pitch_deg)
        return person

    # ---------- drawing ----------
    def draw(self, frame_bgr: np.ndarray, bbox: Sequence[float], gaze_vec3: np.ndarray, thickness: int = 2):
        """
        Draw a 2D arrow projecting gaze onto the image plane.

        Coordinate rationale:
        - Image coords: +x right, +y down
        - Model vector (camera coords): +x left, +y up, +z forward
        - To align: flip X (left<->right) and usually keep Y as-is (image y is down)
        """
        x1, y1, x2, y2 = map(int, bbox)
        anchor = (int((x1 + x2) / 2), y1 if self.head_point == "top" else int((y1 + y2) / 2))

        gx, gy, gz = gaze_vec3.astype(np.float32)

        # Apply configurable axis flips (defaults fix left/right mirroring)
        if self.flip_x:
            gx = -gx
        if self.flip_y:
            gy = -gy

        den = (abs(gz) if self.use_abs_z else (gz if gz != 0 else 1e-3)) + 1e-3
        end = (
            int(anchor[0] + self.arrow_len * (gx / den)),
            int(anchor[1] + self.arrow_len * (gy / den))
        )

        cv2.arrowedLine(frame_bgr, anchor, end, (0, 255, 255), thickness, tipLength=0.25)
        return frame_bgr
