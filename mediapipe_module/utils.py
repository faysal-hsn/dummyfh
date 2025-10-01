"""Utility functions for the mediapipe module"""

import cv2
import numpy as np

def clamp(v, lo, hi):
    """Keep a value v within the range of lo and hi"""
    return max(lo, min(hi, v))

def crop_square(img, x1, y1, x2, y2, pad_ratio=1.2, min_pad=16):
    """Pad a rect to a square and crop safely."""
    h, w = img.shape[:2]
    bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)
    side = int(max(bw, bh) * pad_ratio)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(side // 2, min_pad)
    sx1, sy1 = clamp(cx - half, 0, w - 1), clamp(cy - half, 0, h - 1)
    sx2, sy2 = clamp(cx + half, 0, w - 1), clamp(cy + half, 0, h - 1)
    return img[sy1:sy2, sx1:sx2]

def crop_from_points(img, pts, pad_ratio=2.2, min_pad=12):
    """Square crop around a set of (x,y) points with padding."""
    if not pts:
        return None
    xs, ys = zip(*pts)
    x1, x2 = clamp(min(xs), 0, img.shape[1]-1), clamp(max(xs), 0, img.shape[1]-1)
    y1, y2 = clamp(min(ys), 0, img.shape[0]-1), clamp(max(ys), 0, img.shape[0]-1)
    return crop_square(img, x1, y1, x2, y2, pad_ratio=pad_ratio, min_pad=min_pad)

def resize_224(crop):
    """Resize an image to 244x244 (the required input by GazeNet)"""
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

def to_px(landmarks, iw, ih, idx_list):
    """Convert landmark coordinates into pixel coordinates"""
    return [(int(landmarks[i].x * iw), int(landmarks[i].y * ih)) for i in idx_list]

def make_facegrid(
    bbox_xywh, image_shape, grid_size=25, flatten=True, normalized=False, clip_outside=True
):
    """
    Build a binary facegrid mask marking cells overlapped by the face bbox.

    Args:
        bbox_xywh: (x, y, w, h) - bounding box with top-left corner (x, y) and size (w, h).
        image_shape: (H, W) - shape of the image (height, width).
        grid_size: Number of cells along one side of the grid (default 25).
        flatten: If True, returns the mask as a 1D array. If False, returns 2D.
        normalized: If True, bbox_xywh values are in [0,1] relative to image size.
        clip_outside: If True, clips bbox to be inside image bounds.

    Returns:
        A binary mask (grid_size x grid_size or flattened) with 1s where the bbox overlaps.
    """
    H, W = image_shape[:2]  # Get image height and width
    x, y, w, h = map(float, bbox_xywh)  # Ensure bbox values are floats

    # If bbox is normalized (0-1), scale to image size
    if normalized:
        x *= W
        y *= H
        w *= W
        h *= H

    # Optionally clip bbox so it doesn't go outside the image
    if clip_outside:
        x = np.clip(x, 0, W)
        y = np.clip(y, 0, H)
        w = np.clip(w, 0, W - x)
        h = np.clip(h, 0, H - y)

    # If bbox is empty or invalid, return an all-zeros mask
    if w <= 0 or h <= 0:
        mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        return mask.ravel() if flatten else mask

    # Calculate the size of each grid cell in pixels
    cell_w = W / grid_size
    cell_h = H / grid_size

    # Find which grid cells the bbox covers (left, right, top, bottom indices)
    left   = int(np.floor(x / cell_w))
    right  = int(np.ceil((x + w) / cell_w) - 1)
    top    = int(np.floor(y / cell_h))
    bottom = int(np.ceil((y + h) / cell_h) - 1)

    # Clamp indices to be within grid bounds
    left   = max(0, min(grid_size - 1, left))
    right  = max(0, min(grid_size - 1, right))
    top    = max(0, min(grid_size - 1, top))
    bottom = max(0, min(grid_size - 1, bottom))

    # Create the mask and set the covered cells to 1
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    mask[top:bottom+1, left:right+1] = 1

    # Return as a flat array if requested, else as a 2D mask
    return mask.ravel() if flatten else mask
