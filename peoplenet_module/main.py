import configparser
import logging
import cv2
import numpy as np
import onnxruntime as ort

from datatypes import PersonView
from peoplenet_module.tracker import IOUTracker
from peoplenet_module.utils import nms

# Set up logger
logger = logging.getLogger("PeopleNet Module")

# Config variables
config = configparser.ConfigParser()
config.read("config.cfg")
ONNX_PATH   = config["peoplenet"]["ONNX_PATH"]
CONF_THRESH = float(config["peoplenet"]["CONF_THRESH"])

# post-processing config variables
IOU_THRESH_NMS = float(config["peoplenet"]["IOU_THRESH_NMS"])
INPUT_W = config["peoplenet"].getint("INPUT_W")
INPUT_H = config["peoplenet"].getint("INPUT_H")
PERSON_CLASS_ID = config["peoplenet"].getint("PERSON_CLASS_ID")
BBOX_NORM = config["peoplenet"].getfloat("BBOX_NORM")
GRID_OFFSET = config["peoplenet"].getfloat("GRID_OFFSET")

class PeopleNet:
    """Class for the PeopleNet module"""
    def __init__(self):
        # Set up model
        logger.debug("Loading ONNX model from: %s", ONNX_PATH)
        self.sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        self.in_name   = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        logger.debug("ONNX model loaded successfully.")

        # Set up tracker
        self.tracker = IOUTracker(iou_match_threshold=0.3, max_missed=10)

    def process_frame(self, frame, frame_data):
        """
        Runs the PeopleNet object detection and tracking pipeline on a given video frame.
        Returns: dict {tid -> PersonView}
        """
        H, W = frame_data.H, frame_data.W

        # Run the model on the frame
        inp = self._preprocess_bgr(frame)
        outs = self.sess.run(self.out_names, {self.in_name: inp})

        # Post-process to detections
        dets = self._postprocess_detectnet_v2(outs, W, H)
        logger.debug("Frame %d: %d detection(s) after postproc.", frame_data.idx, len(dets))

        # Track
        tracks = self.tracker.update(dets)  # list of dicts (id,bbox,score,age,gender)

        person_views = {}
        for tr in tracks:
            # Backward compatibility to tolerate either dict or (tid, bbox, score) tuples
            if isinstance(tr, dict):
                tid   = tr.get("id")
                bbox  = tr.get("bbox")
                score = tr.get("score", 0.0)
            else:
                tid, bbox, score = tr

            if score < CONF_THRESH:
                continue

            # Create a crop for the detected person
            try:
                crop = self._crop_frame(frame, bbox)
            except ValueError:
                continue

            # Build PersonView
            pv = PersonView(tid, frame_data, score, bbox, crop)

            person_views[tid] = pv

        return person_views

    def _preprocess_bgr(self, frame_bgr):
        resized = cv2.resize(frame_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = (rgb.astype(np.float32) / 255.0)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img

    def _postprocess_detectnet_v2(self, outputs, orig_w, orig_h):
        o0, o1 = outputs
        o0, o1 = np.array(o0), np.array(o1)
        bbox, conf = (o0, o1) if o0.shape[1] > o1.shape[1] else (o1, o0)

        _, bbox_ch, Hc, Wc = bbox.shape
        _, C, Hc2, Wc2 = conf.shape
        if (Hc != Hc2) or (Wc != Wc2) or (bbox_ch != 4 * C):
            logger.error("Unexpected output shapes: bbox %s, conf %s", bbox.shape, conf.shape)
            raise RuntimeError(f"Unexpected output shapes: bbox {bbox.shape}, conf {conf.shape}")

        sx, sy = orig_w / INPUT_W, orig_h / INPUT_H
        cell_w, cell_h = INPUT_W / Wc, INPUT_H / Hc

        dets = []
        c = PERSON_CLASS_ID
        conf_map = conf[0, c, :, :]
        ys, xs = np.where(conf_map >= CONF_THRESH)

        for y, x in zip(ys, xs):
            score = float(conf_map[y, x])

            # center location in grid-space (normalized)
            cx = (x * cell_w + GRID_OFFSET) / BBOX_NORM
            cy = (y * cell_h + GRID_OFFSET) / BBOX_NORM

            base = c * 4
            x1 = (bbox[0, base + 0, y, x] - cx) * -BBOX_NORM * sx
            y1 = (bbox[0, base + 1, y, x] - cy) * -BBOX_NORM * sy
            x2 = (bbox[0, base + 2, y, x] + cx) *  BBOX_NORM * sx
            y2 = (bbox[0, base + 3, y, x] + cy) *  BBOX_NORM * sy

            # clip + sanity
            x1 = max(0.0, min(float(x1), orig_w - 1))
            y1 = max(0.0, min(float(y1), orig_h - 1))
            x2 = max(0.0, min(float(x2), orig_w - 1))
            y2 = max(0.0, min(float(y2), orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                logger.debug("Skipping invalid bbox: %s", (x1, y1, x2, y2))
                continue

            dets.append([x1, y1, x2, y2, score, c])

        if not dets:
            return []

        dets = np.array(dets, dtype=np.float32)
        keep = nms(dets[:, :4], dets[:, 4], iou_thresh=IOU_THRESH_NMS)
        return dets[keep].tolist()

    def _crop_frame(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame.shape[:2]

        # Clamp to image boundaries
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        # Validate coordinates
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox for crop")

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Empty crop")

        return crop
