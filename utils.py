import configparser
import logging
import os
import contextlib
import json
import numpy as np
import cv2

# Logging
logger = logging.getLogger("Pipeline")

# Config
config = configparser.ConfigParser()
config.read("config.cfg")
SAVE_CROPS = config["testing"].getboolean("SAVE_CROPS", fallback=False)
OUT_CROPS_DIR = config["testing"]["OUT_CROPS_DIR"]

def get_test_crops():
    """
    Get the list of test crops for the pipeline.
    """
    crops = []
    for crop in os.listdir("test_crops"):
        crops.append(os.path.join("test_crops", crop))
    return crops

@contextlib.contextmanager
def mute_native_stderr():
    """
    Mute native stderr output, to avoid annoying logs to the terminal by MediaPipe.
    """
    saved = os.dup(2)                     # real fd for stderr
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), 2) # redirect fd 2 -> /dev/null
            yield                        # <-- must yield!
    finally:
        os.dup2(saved, 2)                # restore
        os.close(saved)

def save_crops(frame, person_views):
    """
    If SAVE_CROPS is enabled, save the cropped person views to image files,
    along with an image of the full frame.
    For development and testing purposes.
    """
    if not SAVE_CROPS:
        return

    # make the output directory if it doesn't exist
    os.makedirs(OUT_CROPS_DIR, exist_ok=True)

    frame_idx = person_views[0].frame.idx if person_views else "no_people"

    # Create a subfolder for this frame
    frame_dir = os.path.join(OUT_CROPS_DIR, f"frame_{frame_idx:06d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Save the whole frame as an image (once per frame)
    frame_img_name = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
    if not os.path.exists(frame_img_name):
        cv2.imwrite(frame_img_name, frame)
        logger.info("Saved full frame image: %s", frame_img_name)

    # Save crops
    for person in person_views:
        # Create a subfolder for this person
        person_dir = os.path.join(frame_dir, f"person_{person.tid:03d}")
        os.makedirs(person_dir, exist_ok=True)

        # People crops
        person_crop_name = os.path.join(person_dir, f"id{person.tid}_s{person.score:.2f}.jpg")
        cv2.imwrite(person_crop_name, person.crop)
        logger.info("Saved crop: %s", person_crop_name)

        # Face crops
        if person.face is not None:
            if person.face.crop is not None:
                face_crop_name = os.path.join(person_dir, f"id{person.tid}_face.jpg")
                cv2.imwrite(face_crop_name, person.face.crop)
                logger.info("Saved face crop: %s", face_crop_name)

            # Eye crops
            if person.face.left_eye is not None:
                left_eye_name = os.path.join(person_dir, f"id{person.tid}_left_eye.jpg")
                cv2.imwrite(left_eye_name, person.face.left_eye)
                logger.info("Saved left eye crop: %s", left_eye_name)

            if person.face.right_eye is not None:
                right_eye_name = os.path.join(person_dir, f"id{person.tid}_right_eye.jpg")
                cv2.imwrite(right_eye_name, person.face.right_eye)
                logger.info("Saved right eye crop: %s", right_eye_name)

            # Save facegrid as txt
            if person.face.facegrid is not None:
                facegrid_name = os.path.join(person_dir, f"id{person.tid}_facegrid.txt")
                fg_2d = person.face.facegrid.reshape(25, 25).astype(np.uint8)    # (25,25) for inspection
                np.savetxt(facegrid_name, fg_2d, fmt="%d")  # easy to eyeball
                logger.info("Saved facegrid: %s", facegrid_name)

        x1, y1, x2, y2 = map(int, person.bbox)
        if person.face is not None:
            x1_f, y1_f, x2_f, y2_f = map(int, person.face.bbox)
            face_text = f"top left: ({x1_f}, {y1_f}), bottom right: ({x2_f}, {y2_f})"

            if person.face.left_eye is not None:
                eye_text = "detected"
            else:
                eye_text = "not detected"
        else:
            face_text = "not detected"
            eye_text = "not detected"

        # Save metadata as json
        person_dict = {
            "tid": person.tid,
            "score": float(round(person.score, 2)),
            "bbox": f"top left: ({x1}, {y1}), bottom right: ({x2}, {y2})",
            "face": face_text,
            "eyes": eye_text,
            "age_range": getattr(person, "age_range", "not detected"),
            "age_confidence": float(round(person.age_confidence, 2)) if getattr(
                person, "age_confidence", None) is not None else None,
            "gender": getattr(person, "gender", "not detected"),
            "gender_confidence": float(round(person.gender_confidence, 2)) if getattr(
                person, "gender_confidence", None) is not None else None
        }
        meta_name = os.path.join(person_dir, f"id{person.tid}_meta.json")
        with open(meta_name, "w", encoding="utf-8") as f:
            json.dump(person_dict, f, indent=2)
        logger.info("Saved metadata: %s", meta_name)

def _safe_face_crop(frame_bgr, face_bbox):
    """Create a BGR crop from face bbox, clamped to frame bounds."""
    if face_bbox is None:
        return None
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, face_bbox)
    x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]
