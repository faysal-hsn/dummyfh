# gaze360/main.py
"""
Gaze360/L2CS wrapper: loads ./gaze360/models/L2CS_packed.pth (or compatible)
and predicts 3D gaze.
- Adds person.gaze_vec3 / person.gaze_yaw_deg / person.gaze_pitch_deg / person.gaze_frame_idx
- draw(): renders a 2D arrow for gaze direction (with axis-flip fixes)

Drop-in: from gaze360.main import Gaze360
"""

from __future__ import annotations
import configparser
import logging
import os
from typing import Tuple, Sequence, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, BasicBlock

from gaze360.l2cs import L2CS

logger = logging.getLogger("Gaze360")


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

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _infer_bins_from_state(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    for head in ("fc_yaw_gaze.weight", "fc_pitch_gaze.weight"):
        if head in sd:
            return int(sd[head].shape[0])
    return None

def _build_l2cs(arch: str, bins: int) -> L2CS:
    if arch == "ResNet18":
        return L2CS(BasicBlock,   [2, 2,  2, 2], bins)
    if arch == "ResNet34":
        return L2CS(BasicBlock,   [3, 4,  6, 3], bins)
    if arch == "ResNet101":
        return L2CS(Bottleneck,   [3, 4, 23, 3], bins)
    if arch == "ResNet152":
        return L2CS(Bottleneck,   [3, 8, 36, 3], bins)
    # default ResNet50
    return L2CS(Bottleneck,       [3, 4,  6, 3], bins)


class Gaze360:
    def __init__(self):
        # --- config (safe even if section is missing)
        cfg = configparser.ConfigParser()
        cfg.read("config.cfg")
        has = cfg.has_section("gaze360")
        def get(key, default):     return cfg["gaze360"].get(key, str(default)) if has else str(default)
        def getint(key, default):  return cfg["gaze360"].getint(key, default)   if has else int(default)
        def getbool(key, default): return cfg["gaze360"].getboolean(key, default) if has else bool(default)

        # Defaults to your packed L2CS checkpoint
        self.ckpt_path         = get("GAZE_MODEL", "./gaze360/models/L2CS_packed.pth")
        self.input_size        = getint("INPUT_SIZE", 448)  # L2CS uses 448 by default
        self.use_imagenet_norm = getbool("USE_IMAGENET_NORM", True)
        self.arrow_len         = getint("ARROW_LEN", 80)
        self.head_point        = get("HEAD_POINT", "center").lower()
        self.flip_x            = getbool("FLIP_X", True)
        self.flip_y            = getbool("FLIP_Y", False)
        self.use_abs_z         = getbool("USE_ABS_Z", True)
        self.strict_load       = getbool("STRICT_LOAD", True)     # we want real model, not stub
        self.override_arch     = get("L2CS_ARCH", "ResNet50")      # can override packed metadata
        self.engagement_window = getint("ENGAGEMENT_WINDOW_DEG", 20)

        # device selection
        dev_cfg = get("DEVICE", "auto").lower()
        if dev_cfg == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif dev_cfg == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif dev_cfg == "auto" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif dev_cfg == "auto" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"Gaze360 checkpoint not found at {self.ckpt_path}. "
                f"Set [gaze360].GAZE_MODEL in config.cfg or place the file there."
            )
        logger.info("Loading L2CS checkpoint: %s on %s", self.ckpt_path, self.device)

        # --- load model
        ckpt = torch.load(self.ckpt_path, map_location="cpu")  # load on CPU first (safer), then move
        if self.strict_load:
            self.model = self._build_l2cs_from_pth(ckpt)
        else:
            self.model = self._build_or_extract_model_lenient(ckpt)

        self.model.to(self.device).eval()
        logger.info("Gaze360/L2CS initialized: arch=%s bins=%d bin_size=%g°",
                    self.l2cs_arch, self.bins, self.bin_size_deg)

    # ---------- L2CS (strict) ----------
    def _build_l2cs_from_pth(self, ckpt: Any) -> nn.Module:
        """
        Load packed L2CS dict OR raw state_dict and adapt head shape:
        - Variant A (classic): 2048 -> 256 (fc_finetune) -> bins
        - Variant B (direct):  2048 -> bins  (no bottleneck)
        """
        # 1) unpack
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            arch  = str(ckpt.get("arch", self.override_arch))
            bins_meta = ckpt.get("bins", None)
        elif isinstance(ckpt, dict):
            state = ckpt
            arch  = self.override_arch
            bins_meta = None
        elif isinstance(ckpt, nn.Module):
            mdl = ckpt
            self.bins = getattr(mdl, "bins", 90)
            self.bin_size_deg = 4.0 if self.bins == 90 else (3.0 if self.bins == 28 else 360.0 / self.bins)
            self.l2cs_arch = self.override_arch
            return mdl
        else:
            raise RuntimeError("Unsupported checkpoint type for strict L2CS load.")

        # 2) clean keys / infer bins
        state = _strip_module_prefix(state)
        bins = int(bins_meta) if bins_meta is not None else _infer_bins_from_state(state)
        if bins is None:
            raise RuntimeError("Could not infer 'bins' from checkpoint heads.")

        # 3) build a baseline (classic) L2CS: 2048 -> 256 -> bins
        model = _build_l2cs(arch, bins)

        # 4) detect head style from checkpoint
        head_in = None
        if "fc_yaw_gaze.weight" in state:
            head_in = state["fc_yaw_gaze.weight"].shape[1]  # expected 256 or 2048

        # 5) if checkpoint heads are 2048-wide, switch to 'direct' variant
        if head_in == 2048:
            # remove any incompatible fc_finetune weights from state
            for k in list(state.keys()):
                if k.startswith("fc_finetune."):
                    state.pop(k, None)

            # rebuild heads to in_features=2048 and no bottleneck
            model.fc_finetune = nn.Identity()
            model.fc_yaw_gaze = nn.Linear(2048, bins)
            model.fc_pitch_gaze = nn.Linear(2048, bins)
            logger.info("Using DIRECT heads (2048->bins), fc_finetune removed.")
        elif head_in == 256 or head_in is None:
            # keep classic layout
            logger.info("Using CLASSIC heads (2048->256->bins).")
        else:
            logger.warning("Unexpected head in_features=%s; attempting flexible load.", head_in)

        # 6) load state_dict non-strict, report diffs
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Filter out batch-norm tracking keys from the noise
            missing_f = [m for m in missing if not m.endswith("num_batches_tracked")]
            if missing_f or unexpected:
                logger.warning("Non-strict load. missing=%d unexpected=%d", len(missing_f), len(unexpected))
                if missing_f:
                    logger.debug("Missing (first 12): %s", missing_f[:12])
                if unexpected:
                    logger.debug("Unexpected (first 12): %s", unexpected[:12])

        # 7) store decode params
        self.l2cs_arch = arch
        self.bins = bins
        self.bin_size_deg = 4.0 if bins == 90 else (3.0 if bins == 28 else 360.0 / bins)
        return model

    # ---------- lenient fallback (kept for completeness) ----------
    def _build_or_extract_model_lenient(self, ckpt):
        # Case 1: full module
        if isinstance(ckpt, torch.nn.Module):
            return ckpt

        # Case 2: packaged model object
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), torch.nn.Module):
            return ckpt["model"]

        # Case 3: state_dict -> try L2CS by inferring bins
        if isinstance(ckpt, dict):
            state = _strip_module_prefix(ckpt.get("state_dict", ckpt))
            bins = _infer_bins_from_state(state) or 90
            arch = self.override_arch
            model = _build_l2cs(arch, bins)
            model.load_state_dict(state, strict=False)
            self.l2cs_arch, self.bins = arch, bins
            self.bin_size_deg = 4.0 if bins == 90 else (3.0 if bins == 28 else 360.0 / bins)
            logger.warning("Lenient load into L2CS (arch=%s bins=%d). Verify results.", arch, bins)
            return model

        # Fallback stub (should not happen with your packed .pth)
        class _Stub(torch.nn.Module):
            def __init__(self): super().__init__(); self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(3,3)
            def forward(self, x): return self.fc(self.pool(x).squeeze(-1).squeeze(-1))
        self.l2cs_arch, self.bins, self.bin_size_deg = "stub", 3, 120.0
        logger.warning("Using stub network; set STRICT_LOAD=True with packed L2CS .pth.")
        return _Stub()

    # ---------- inference ----------
    @torch.no_grad()
    def predict_from_crop(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty face crop for gaze estimation.")

        # Preprocess
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).float() / 255.0              # [H,W,3]
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)        # [1,3,H,W]
        if self.use_imagenet_norm:
            t = _imagenet_norm(t)

        # Forward - L2CS typically returns (pitch_logits, yaw_logits)
        out = self.model(t)

        # Case A: L2CS tuple of logits
        if isinstance(out, (tuple, list)) and len(out) == 2:
            pitch_logits, yaw_logits = out
            # softmax over bins
            pitch_prob = torch.softmax(pitch_logits, dim=1)
            yaw_prob   = torch.softmax(yaw_logits,   dim=1)
            # expected bin index
            idx = torch.arange(self.bins, dtype=torch.float32, device=out[0].device).view(1, -1)
            pitch_deg = (pitch_prob * idx).sum(dim=1) * self.bin_size_deg - (self.bins * self.bin_size_deg) / 2.0
            yaw_deg   = (yaw_prob   * idx).sum(dim=1) * self.bin_size_deg - (self.bins * self.bin_size_deg) / 2.0
            # Convert to radians for vector synthesis
            yaw  = float(torch.deg2rad(yaw_deg).item())
            pitch= float(torch.deg2rad(pitch_deg).item())

            gaze_vec = np.array([
                np.sin(yaw) * np.cos(pitch),   # x  (left +)
                -np.sin(pitch),                # y  (up +)
                np.cos(yaw) * np.cos(pitch),   # z  (forward +)
            ], dtype=np.float32)

            yaw_deg_f   = float(yaw_deg.item())
            pitch_deg_f = float(pitch_deg.item())

        # Case B: direct [1,2] -> (yaw, pitch) radians
        elif torch.is_tensor(out) and out.ndim == 2 and out.shape[-1] == 2:
            yaw, pitch = out[0].tolist()
            gaze_vec = np.array([
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ], dtype=np.float32)
            yaw_deg_f, pitch_deg_f = _vec_to_yaw_pitch(gaze_vec)

        # Case C: direct 3D vector
        elif torch.is_tensor(out) and out.ndim == 2 and out.shape[-1] == 3:
            gaze_vec = out[0].detach().float().cpu().numpy().astype(np.float32)
            gaze_vec = _unit(gaze_vec)
            yaw_deg_f, pitch_deg_f = _vec_to_yaw_pitch(gaze_vec)

        else:
            raise RuntimeError("Unsupported model output format for gaze estimation.")

        gaze_vec = _unit(gaze_vec)

        # Light sanity logging (throttled)
        if np.random.rand() < 0.02:
            logger.debug("gaze vec=%s  yaw=%.1f  pitch=%.1f",
                         np.round(gaze_vec, 3), yaw_deg_f, pitch_deg_f)

        return gaze_vec, yaw_deg_f, pitch_deg_f

    @torch.no_grad()
    def detect_gaze(self, person):
        """Mutates and returns `person` with gaze attributes and the frame index it belongs to."""
        logger.info("Starting Gaze Detection")

        gaze_vec, yaw_deg, pitch_deg = self.predict_from_crop(person.crop)
        setattr(person, "gaze_vec3", gaze_vec)
        setattr(person, "gaze_yaw_deg", yaw_deg)
        setattr(person, "gaze_pitch_deg", pitch_deg)
        setattr(person, "gaze_frame_idx", getattr(getattr(person, "frame_data", None), "idx", None))

        # Count as engagement if gaze is within the engagement window
        if (yaw_deg > (-1 * self.engagement_window) and yaw_deg < self.engagement_window
            and pitch_deg > (-1 * self.engagement_window) and pitch_deg < self.engagement_window):
            person.engagement = True
            logger.info("Person (%d)'s engagement recorded", person.tid)

        return person

    # ---------- drawing ----------
    def draw(self, frame_bgr: np.ndarray, person, thickness: int = 2):
        """
        Draw a 2D arrow projecting gaze onto the image plane.

        Coordinate rationale:
        - Image coords: +x right, +y down
        - Model vector (camera coords): +x left, +y up, +z forward
        - To align: flip X (left<->right) and usually keep Y as-is (image y is down)
        """
        x1, y1, x2, y2 = map(int, person.face.bbox)
        anchor = (int((x1 + x2) / 2), y1 if self.head_point == "top" else int((y1 + y2) / 2))

        gx, gy, gz = person.gaze_vec3.astype(np.float32)

        # Apply configurable axis flips (defaults fix left/right mirroring)
        if self.flip_x:
            gx = -gx
        if self.flip_y:
            gy = -gy

        # scale denominator — using |z| stabilizes length for near-parallel gazes
        den = (abs(gz) if self.use_abs_z else (gz if gz != 0 else 1e-3)) + 1e-3

        end = (
            int(anchor[0] + self.arrow_len * (gx / den)),
            int(anchor[1] + self.arrow_len * (gy / den))
        )

        # change the colour depending on engagement
        if person.engagement is True:
            gaze_colour = (255, 0, 255)
        else:
            gaze_colour = (0, 255, 255)

        cv2.arrowedLine(frame_bgr, anchor, end, gaze_colour, thickness, tipLength=0.25)
        return frame_bgr
