"""
The Face Detection module using MediaPipe.

It will take inputs from the Person Detection Module,
and will give outputs to the Gaze Detection Module.

Important:
This is meant to be run from pipeline.py
NOT from this file
because the baseline directory is the top-level directory for this repo
"""
import logging

import cv2
import mediapipe as mp
import numpy as np


from datatypes import FaceView
from mediapipe_module.utils import clamp, crop_square, crop_from_points, resize_224, make_facegrid

# Set up logger
logger = logging.getLogger("MediaPipe Module")

class MediaPipe:
    """Class for operations to do with the MediaPipe model"""
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        # Initialise MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_face(self, person):
        """Detect the face
        INPUT: Full-body image crop of a person
        OUTPUT: 
            If face: 
                - return TRUE
                - save an image crop of the person's face
                - save facegrids of a the location of the person's face in the original image
            If no face: return FALSE
        """
        # Get image height and width
        image = person.crop

        # get frame dimensions and person dimensions
        fH, fW = person.frame.H, person.frame.W
        pH, pW = image.shape[:2]

        # Create the face detection model
        results_fd = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Handle no face
        if not results_fd.detections:
            logger.debug("No face detected.")
            return person

        # Take the first face (the crop should only have one)
        det = results_fd.detections[0]
        bbox_rel = det.location_data.relative_bounding_box

        # Compute bbox in person crop coordinates (relative to crop)
        x_p = int(bbox_rel.xmin * pW)
        y_p = int(bbox_rel.ymin * pH)
        w_p = int(bbox_rel.width * pW)
        h_p = int(bbox_rel.height * pH)

        # Clamp to crop and compute corners (crop coordinates)
        x1_p, y1_p = clamp(x_p, 0, pW-1), clamp(y_p, 0, pH-1)
        x2_p, y2_p = clamp(x_p + w_p, 0, pW-1), clamp(y_p + h_p, 0, pH-1)

        # Compute bbox in frame coordinates (relative to full frame)
        x1_f = x_p + person.bbox[0]
        y1_f = y_p + person.bbox[1]
        x2_f = x_p + w_p + person.bbox[0]
        y2_f = y_p + h_p + person.bbox[1]
        w_f = x2_f - x1_f
        h_f = y2_f - y1_f

        # Pad to a square and crop face
        face_crop = crop_square(image, x1_p, y1_p, x2_p, y2_p, pad_ratio=1.25, min_pad=24)
        if face_crop is None or face_crop.size == 0:
            logger.error("Face detected but crop failed.")
            return person
        face_out = resize_224(face_crop)

        # Build facegrid from the ORIGINAL bbox on the full frame
        fg_flat = make_facegrid((x1_f, y1_f, h_f, w_f), (fH, fW),
                                grid_size=25, flatten=True, normalized=False)


        # # Common shapes used by TAO exporters:
        fg_625 = fg_flat.astype(np.float32)                 # (625,)
        fg_nchw = fg_625.reshape(1, 1, 625, 1)              # (1,1,625,1)
        fg_2d = fg_flat.reshape(25, 25).astype(np.uint8)    # (25,25) for inspection

        # # Save facegrids to disk for reuse
        # np.save(os.path.join(image_outdir, "facegrid_625.npy"), fg_625)
        # np.save(os.path.join(image_outdir, "facegrid_1x1x625x1.npy"), fg_nchw)
        # np.savetxt(os.path.join(image_outdir, "facegrid_25x25.txt"), fg_2d, fmt="%d")  # easy to eyeball
        # logger.info("Saved facegrid: flat=%s, nchw=%s, grid=%s", fg_625.shape, fg_nchw.shape, fg_2d.shape)

        # Make a FaceView (coordinates are relative to frame)
        face_view = FaceView(bbox=(x1_f, y1_f, x2_f, y2_f), crop=face_out, facegrid=fg_nchw)

        # Add face view to person
        person.face = face_view

        return person

    # Eye detection
    def detect_eyes(self, person):
        """
        Get a full_body image of a person and produce crops of the eyes
        INPUT: Full-body image crop of a person
        OUTPUTS: 
        """
        # Get image height and width
        image = person.crop
        H, W = image.shape[:2]

        LEFT_IRIS = [469, 470, 471, 472]
        RIGHT_IRIS = [474, 475, 476, 477]

        # Run the eye model
        results_fm = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Handle no eyes
        if not results_fm.multi_face_landmarks:
            logger.info("No iris landmarks found; only face_crop.png was saved.")
            return person

        logger.info("Iris landmarks detected in image")
        fl = results_fm.multi_face_landmarks[0]
        iw, ih = W, H

        def to_px(landmarks, iw, ih, idx_list):
            return [(int(landmarks[i].x * iw), int(landmarks[i].y * ih)) for i in idx_list]

        left_pts = to_px(fl.landmark, iw, ih, LEFT_IRIS)
        right_pts = to_px(fl.landmark, iw, ih, RIGHT_IRIS)

        left_eye = crop_from_points(image, left_pts, pad_ratio=2.6, min_pad=16)
        right_eye = crop_from_points(image, right_pts, pad_ratio=2.6, min_pad=16)

        # Add left and right eye crops to person
        if (right_eye is not None and right_eye.size != 0
        and left_eye is not None and left_eye.size != 0):
            person.face.right_eye = right_eye
            person.face.left_eye = left_eye

        return person
