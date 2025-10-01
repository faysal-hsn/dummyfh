# peoplenet_module/tracker.py
import logging
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from peoplenet_module.utils import iou

logger = logging.getLogger("PeopleNet Tracker")


class PersonTrack:
    """Persistent storage for a single person across frames."""
    _next_id = 1

    def __init__(self, bbox: Sequence[float], score: float = 0.0):
        self.id = PersonTrack._next_id
        PersonTrack._next_id += 1

        # Core kinematics
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1,y1,x2,y2]
        self.score = float(score)
        self.missed = 0         # consecutive frames without match
        self.hits = 1           # number of matches

    # ----- Updates
    def update_bbox(self, bbox: Sequence[float], score: float | None = None):
        self.bbox = np.array(bbox, dtype=np.float32)
        if score is not None:
            self.score = float(score)
        self.missed = 0
        self.hits += 1
            
    # ----- Export
    def to_dict(self, score_override: float | None = None) -> dict:
        return {
            "id": int(self.id),
            "bbox": self.bbox.tolist(),
            "score": float(self.score if score_override is None else score_override)
        }


class IOUTracker:
    """
    Greedy IoU-based tracker for associating detections across frames.

    Key points:
    - Tracks are stored in a *dict* keyed by ID: self.tracks[id] -> PersonTrack
      * This makes pipeline lookups like tracks[tid] correct and O(1).
    - update(detections) expects a list of [x1,y1,x2,y2,score,(class_id...)].
    - Returns a list of dicts: {"id","bbox","score","age","gender"} for all active tracks.
    """

    def __init__(self, iou_match_threshold: float = 0.3, max_missed: int = 10):
        self.tracks: Dict[int, PersonTrack] = {}   # id -> track
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_missed = int(max_missed)

    # ---------- Public helpers ----------
    def get_track_by_id(self, tid: int) -> PersonTrack | None:
        return self.tracks.get(int(tid))

    # ---------- Internal utilities ----------
    @staticmethod
    def _iou_matrix(track_list: List[PersonTrack], boxes: List[Sequence[float]]) -> np.ndarray:
        if not track_list or not boxes:
            return np.zeros((len(track_list), len(boxes)), dtype=np.float32)
        M = np.zeros((len(track_list), len(boxes)), dtype=np.float32)
        for ti, t in enumerate(track_list):
            for di, db in enumerate(boxes):
                M[ti, di] = iou(t.bbox, db)
        return M

    @staticmethod
    def _to_list(d: Dict[int, PersonTrack]) -> List[PersonTrack]:
        # Stable order isn't required, but keep deterministic by id
        return [d[k] for k in sorted(d.keys())]

    # ---------- Main update ----------
    def update(self, detections: List[Sequence[float]]) -> List[dict]:
        """
        Update the tracker with current-frame detections using greedy IoU matching.

        Args:
            detections: list of [x1,y1,x2,y2,score,(class_id...)]
        Returns:
            List of dicts: [{"id","bbox","score","age","gender"}, ...]
        """
        det_boxes = [d[:4] for d in detections] if detections else []
        det_scores = [float(d[4]) for d in detections] if detections else []

        # --- No detections: age all tracks & prune stale
        if not det_boxes:
            remove_ids: List[int] = []
            for tid, t in self.tracks.items():
                t.missed += 1
                if t.missed > self.max_missed:
                    remove_ids.append(tid)
            for tid in remove_ids:
                del self.tracks[tid]
            return [t.to_dict() for t in self._to_list(self.tracks)]

        # --- Build IoU matrix (tracks x detections)
        track_list = self._to_list(self.tracks)
        iou_mat = self._iou_matrix(track_list, det_boxes)

        unmatched_track_idx = set(range(len(track_list)))
        unmatched_det_idx = set(range(len(det_boxes)))
        logger.info(f"Unmatched det idx: {list(unmatched_det_idx)}")
        matches: List[Tuple[int, int]] = []

        # Greedy matching
        while iou_mat.size:
            ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            best = iou_mat[ti, di]
            if best < self.iou_match_threshold:
                break
            matches.append((ti, di))
            # Invalidate row/col
            iou_mat[ti, :] = -1.0
            iou_mat[:, di] = -1.0
            unmatched_track_idx.discard(ti)
            unmatched_det_idx.discard(di)

        # --- Update matched tracks
        for ti, di in matches:
            t = track_list[ti]
            t.update_bbox(det_boxes[di], score=det_scores[di])

        # --- Age unmatched tracks
        for ti in unmatched_track_idx:
            track_list[ti].missed += 1

        # --- Remove stale tracks
        for t in list(self.tracks.values()):
            if t.missed > self.max_missed:
                del self.tracks[t.id]

        # --- Create new tracks for unmatched detections
        logger.info(f"Unmatched det idx: {list(unmatched_det_idx)}")
        for di in unmatched_det_idx:
            new_t = PersonTrack(det_boxes[di], score=det_scores[di])
            logger.debug(f"New detection: {new_t.id}")
            self.tracks[new_t.id] = new_t

        # --- Export
        return [t.to_dict() for t in self._to_list(self.tracks)]
