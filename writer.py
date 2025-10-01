import configparser
import logging
import os
from typing import Iterable

import cv2
import csv
from datetime import datetime

# Set up logger
logger = logging.getLogger("Writer")

# Config variables
config = configparser.ConfigParser()
config.read("config.cfg")
OUT_VIDEO_DIR  = config["writer"]["OUTDIR_VIDEO"]
OUT_CSV_DIR    = config["writer"]["OUTDIR_CSV"]
ONLY_GAZE_CSV   = config["writer"].getboolean("ONLY_GAZE_CSV", fallback=True)
OUT_VIDEO_NAME = config["writer"]["OUT_VIDEO_NAME"]
OUT_VIDEO_EXT = config["writer"]["OUT_VIDEO_EXT"]
DRAW_SCORE     = config["writer"].getboolean("DRAW_SCORE")
BOX_THICK      = config["writer"].getint("BOX_THICK")
FONT_SCALE     = config["writer"].getfloat("FONT_SCALE")
TEXT_THICK     = config["writer"].getint("TEXT_THICK")

# Toggle writing functions
WRITE_VIDEO = config["writer"].getboolean("WRITE_VIDEO", fallback=True)
WRITE_INSIGHTS_CSV = config["writer"].getboolean("WRITE_INSIGHTS_CSV", fallback=True)
WRITE_ANNO_CSV = config["writer"].getboolean("WRITE_ANNO_CSV", fallback=True)

# name of the input video
TEST_VIDEO_NAME = config["video"]["TEST_VIDEO_NAME"]


class Writer:
    """Writes annotated frames to an MP4 file."""

    def __init__(self, cap, fps, W, H):
        # Set up video writer
        if WRITE_VIDEO:
            os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_video_filename = f"{OUT_VIDEO_NAME}_{TEST_VIDEO_NAME}_{timestamp}.{OUT_VIDEO_EXT}"
            out_video_path = os.path.join(OUT_VIDEO_DIR, out_video_filename)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_video_path, fourcc, float(fps), (int(W), int(H)))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer at {out_video_path}")
            
            logger.info("Video writer initialized: %s", out_video_path)
            
        # Set up Annotation CSV writer (writes annotations per frame)
        if WRITE_ANNO_CSV:
            os.makedirs(OUT_CSV_DIR, exist_ok=True)
            anno_csv_filename = f"{OUT_VIDEO_NAME}_{TEST_VIDEO_NAME}_{timestamp}.csv"
            anno_csv_path = os.path.join(OUT_CSV_DIR, anno_csv_filename)

            self.anno_csvfile = open(anno_csv_path, "w", newline="")
            fieldnames = [
                "frame_idx",
                "person_id",
                "bbox",
                "age mode",
                "gender mode",
                "gaze pitch (degrees)",
                "gaze yaw (degrees)",
                "engagement"
                ]
            self.anno_csv_writer = csv.DictWriter(self.anno_csvfile, fieldnames=fieldnames)
            self.anno_csv_writer.writeheader()
            logger.info("Annotation CSV writer initialized: %s", out_video_path)


        # Set up insights csv (writes inights at the end of the video)
        if WRITE_INSIGHTS_CSV:
            self.insights_csv_filename = f"insights_{TEST_VIDEO_NAME}_{timestamp}.csv"

        if not WRITE_VIDEO and not WRITE_ANNO_CSV and not WRITE_INSIGHTS_CSV:
            logger.warning("Writer class called but writing functions are switched off. \
                           See 'WRITE_VIDEO' and 'WRITE CSV' in config variables")

    # ---- internals ----
    def _iter_persons(self, person_views) -> Iterable:
        """Accept list/iterable or dict {tid->PersonView}."""
        if isinstance(person_views, dict):
            return person_views.values()
        return person_views

    # ---- drawing helpers ----
    @staticmethod
    def _put_text(img, text, org, color=(0, 0, 0), font_scale=1, thickness=1):
        if not text:
            return
        cv2.putText(
            img,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE * font_scale,
            color,
            TEXT_THICK + thickness,
            cv2.LINE_AA,
        )

    def write_frame(self, frame, frame_idx, person_views):
        """Write a single annotated frame to the output video (no gaze arrow here)."""
        if not WRITE_VIDEO:
            logger.warning("Writer called but writing video disabled")
            return

        vis = frame.copy()

        # Draw frame index in the top-left corner
        if frame_idx is not None:
            self._put_text(vis,  f"Frame: {frame_idx}", (30, 70), color=(0, 0, 0), font_scale=2)

        for person in self._iter_persons(person_views):
            bbox = getattr(person, "bbox", None)
            if bbox is None:
                continue
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except Exception:
                continue

            # Person metadata
            tid   = getattr(person, "tid", None)
            score = float(getattr(person, "score", 0.0))

            # --- Person bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), BOX_THICK)

            # --- ID & detection score (top-left of person bbox)
            id_text = f"ID {tid}" if tid is not None else "ID ?"
            self._put_text(vis, id_text, (x1, max(0, y1 - 6)), color=(0, 0, 0))

            if DRAW_SCORE and score > 0:
                self._put_text(vis, f"conf: {score:.2f}", (x1, max(0, y1 - 6) + 16), color=(0, 0, 0))

            # --- Age/Gender (top-right of person bbox, in red)
            y_offset = max(0, y1 - 40)

            if person.age_mode is not None:
                age_text = f"Age: {person.age_mode}"
                size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICK + 1)[0]
                self._put_text(vis, age_text, (x2 - size[0], y_offset), color=(0, 0, 255))
                y_offset += size[1] + 8

            if person.gender_mode is not None:
                gender_text = f"Gender: {person.gender_mode}"
                gsz = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICK + 1)[0]
                self._put_text(vis, gender_text, (x2 - gsz[0], y_offset), color=(0, 0, 255))

            # --- Face bbox (blue), if any
            face = getattr(person, "face", None)
            if face is not None and getattr(face, "bbox", None) is not None:
                try:
                    fx1, fy1, fx2, fy2 = map(int, face.bbox)
                    cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (255, 0, 0), BOX_THICK)
                except Exception:
                    pass
                
            if person.gaze_yaw_deg is not None and person.gaze_pitch_deg is not None:
                # If the gaze can be counted as looking at the camera, make it stand out
                if person.engagement is True:
                    gaze_colour = (255, 0, 255)
                else:
                    gaze_colour = (255, 0, 0)

                face_angle = f"yaw {person.gaze_yaw_deg:.2f}, Pitch {person.gaze_pitch_deg:.2f}"
                gsz = cv2.getTextSize(face_angle, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICK + 1)[0]
                fx1, fy1, fx2, fy2 = map(int, person.face.bbox)
                fy_offset = fy2 + 36
                self._put_text(vis, face_angle, (fx2 - gsz[0], fy_offset), color=gaze_colour)

            # NOTE: Gaze arrow is intentionally NOT drawn here.
            # The pipeline draws it once via gaze360.draw(...) before calling write_frame().

        self.writer.write(vis)

    def write_annotations_csv(self, person_views):
        """Write person view data to the already open CSV file."""
        if not WRITE_ANNO_CSV:
            logger.warning("Writer called but writing Annotation CSV Disabled")
            return

        for person in self._iter_persons(person_views):
            if ONLY_GAZE_CSV:
                if person.gaze_pitch_deg is None or person.gaze_yaw_deg is None:
                    continue
            self.anno_csv_writer.writerow({
                "frame_idx": person.frame.idx,
                "person_id": person.tid,
                "bbox": person.bbox,
                "age mode": person.age_mode,
                "gender mode": person.gender_mode,
                "gaze pitch (degrees)": person.gaze_pitch_deg,
                "gaze yaw (degrees)": person.gaze_yaw_deg,
                "engagement": person.engagement
            })

    def write_insights_csv(self, person_records):
        """
        Write a csv of data that will be sent to the backend
        Only called once in the pipeline.
        """
        if not WRITE_INSIGHTS_CSV:
            logger.warning("Writer called but writing Insights CSV Disabled")
            return
    
        insights_csv_path = os.path.join(OUT_CSV_DIR, self.insights_csv_filename)
        with open(insights_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=person_records[0].keys())
            writer.writeheader()
            writer.writerows(person_records)

    def release(self):
        """Release video writer resources."""
        self.writer.release()
        self.anno_csvfile.close()
