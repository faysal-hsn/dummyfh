"""
Utils for PeoplNet Module
"""
import configparser, logging
import time, cv2, numpy as np
from typing import Any, Iterable, Optional

# Set up logger
logger = logging.getLogger("PeopleNet Module")

# Config variables
config = configparser.ConfigParser()
config.read("config.cfg")

# PeopleNet (Transformer) input / postproc parameters
INPUT_W = config["peoplenet"].getint("INPUT_W")
INPUT_H = config["peoplenet"].getint("INPUT_H")
PERSON_CLASS_ID = config["peoplenet"].getint("PERSON_CLASS_ID")
BBOX_NORM = config["peoplenet"].getfloat("BBOX_NORM")
GRID_OFFSET = config["peoplenet"].getfloat("GRID_OFFSET")

CONF_THRESH = float(config["peoplenet"]["CONF_THRESH"])
IOU_THRESH_NMS = float(config["peoplenet"]["IOU_THRESH_NMS"])

SHOW_PREVIEW = config["peoplenet"].getboolean("SHOW_PREVIEW")
PREVIEW_EVERY_N_FRAMES = config["peoplenet"].getint("PREVIEW_EVERY_N_FRAMES")
PREVIEW_MAX_WIDTH = config["peoplenet"].getint("PREVIEW_MAX_WIDTH")
LIVE_WINDOW = config["peoplenet"].getboolean("LIVE_WINDOW")

# inline (notebook) preview
def _display_inline(img_bgr: np.ndarray):
    logger.debug("Displaying image inline.")
    from IPython.display import clear_output, display
    from PIL import Image
    h, w = img_bgr.shape[:2]
    max_w = PREVIEW_MAX_WIDTH
    if w > max_w:
        new_h = int(h * max_w / w)
        img_bgr = cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_LINEAR)

    # convert for notebook display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    clear_output(wait=True)
    display(Image.fromarray(img_rgb))
    # tiny sleep to let UI breathe
    time.sleep(0.001)

def _color_for_id(tid: int) -> tuple[int,int,int]:
    color = tuple(int(x) for x in np.random.default_rng(int(tid)).integers(60, 255, size=3))
    logger.debug(f"Color for id {tid}: {color}")
    return color

def _extract_xyxy_id_score(item: Any):
    logger.debug(f"Extracting bbox/id/score from item: {item}")
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("xyxy") or item.get("tlbr")
        tid  = item.get("track_id") or item.get("id") or item.get("tid")
        score= item.get("score") or item.get("conf") or 1.0
        if bbox is None: 
            logger.warning("No bbox found in dict item.")
            return None
        x1,y1,x2,y2 = map(float, bbox)
        return (x1,y1,x2,y2), (None if tid is None else int(tid)), float(score)

    arr = np.asarray(item).reshape(-1)
    if arr.size >= 4:
        x1,y1,x2,y2 = map(float, arr[:4])
        tid  = int(arr[4]) if arr.size >= 5 else None
        score= float(arr[5]) if arr.size >= 6 else 1.0
        return (x1,y1,x2,y2), tid, score

    logger.warning("Item could not be parsed for bbox/id/score.")
    return None

def draw_overlays(
    frame_bgr: np.ndarray,
    tracks: Optional[Iterable[Any]] = None,
    detections: Optional[Iterable[Any]] = None,
    show_scores: bool = True,
):
    logger.debug("Drawing overlays on frame.")
    source = tracks if tracks is not None else detections
    if source is None:
        logger.warning("No tracks or detections provided to draw_overlays.")
        return frame_bgr

    for item in source:
        parsed = _extract_xyxy_id_score(item)
        if not parsed: 
            logger.warning(f"Skipping item with invalid format: {item}")
            continue
        (x1,y1,x2,y2), tid, score = parsed
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        color = _color_for_id(tid if tid is not None else int(score*1000))
        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), color, 2)

        label = []
        if tid is not None: label.append(f"ID {tid}")
        if show_scores and score is not None: label.append(f"{score:.2f}")
        if label:
            txt = "  ".join(label)
            (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame_bgr, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return frame_bgr

def show_preview(img_bgr: np.ndarray):
    logger.debug("Showing preview.")
    if LIVE_WINDOW:
        cv2.imshow("Preview (processing…)", img_bgr)
        cv2.waitKey(1)  # non-blocking
    else:
        _display_inline(img_bgr)

def nms(boxes, scores, iou_thresh=0.5):
    logger.info(f"Running NMS on {len(boxes)} boxes.")
    if len(boxes) == 0:
        logger.warning("No boxes provided to NMS.")
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (np.maximum(0, x2 - x1)) * (np.maximum(0, y2 - y1))
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = (areas[i] + areas[order[1:]] - inter + 1e-6)
        ovr = inter / denom
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    logger.info(f"NMS kept {len(keep)} boxes.")
    return keep

def iou(a, b):
    logger.debug(f"Calculating IoU between {a} and {b}")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    iou_val = inter / (area_a + area_b - inter + 1e-6) if (area_a > 0 and area_b > 0) else 0.0
    logger.debug(f"IoU: {iou_val}")
    return iou_val
