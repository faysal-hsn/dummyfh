"""
Run the pipeline from here.
"""
import configparser
import logging
import sys
import os
import json
import time

import numpy as np
import cv2
from tqdm import tqdm

# Get config
config = configparser.ConfigParser()
config.read("config.cfg")
LOGDIR = config["logging"]["LOGDIR"]
DEBUG = config["logging"].getboolean("DEBUG")

TEST_VIDEO_DIR = config["video"]["TEST_VIDEO_DIR"]
TEST_VIDEO_NAME = config["video"]["TEST_VIDEO_NAME"]
max_frames_raw = config["video"].get("MAX_FRAMES", fallback="None")
MAX_FRAMES = None if max_frames_raw == "None" else int(max_frames_raw)
FRAME_STRIDE = config["video"].getint("FRAME_STRIDE")
TARGET_FPS_TO_PROCESS = config["video"]["TARGET_FPS_TO_PROCESS"]  # "None" or string int

SAVE_CROPS = config["testing"].getboolean("SAVE_CROPS", fallback=False)

# Toggle writing functions
TOGGLE_WRITER = config["writer"].getboolean("TOGGLE_WRITER", fallback=True)
WRITE_VIDEO = config["writer"].getboolean("WRITE_VIDEO", fallback=True)
WRITE_INSIGHTS_CSV = config["writer"].getboolean("WRITE_INSIGHTS_CSV", fallback=True)
WRITE_ANNO_CSV = config["writer"].getboolean("WRITE_ANNO_CSV", fallback=True)

# Toggle models
TOGGLE_FACE = config["face"].getboolean("TOGGLE_FACE", fallback=True)
TOGGLE_AGE_GENDER = config["caffe"].getboolean("TOGGLE_AGE_GENDER", fallback=True)
TOGGLE_GAZE = config["gaze360"].getboolean("TOGGLE_GAZE", fallback=True)

# Configure Logging
os.makedirs(LOGDIR, exist_ok=True)  # Create log directory if it doesn't exist
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    filename=os.path.join(LOGDIR, "pipeline.log"),
    filemode="w",
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger("Pipeline")

# Local imports
from datatypes import FrameData, PersonRecord
from peoplenet_module.main import PeopleNet
from mediapipe_module.main import MediaPipe
from caffe_module.main import Caffe
from gaze360.main import Gaze360
from writer import Writer
from utils import mute_native_stderr, save_crops, _safe_face_crop

def pipeline():
    """
    Main entry point for the pipeline.
    Loads the video and runs PeopleNet -> MediaPipe -> Caffe -> Gaze360,
    then writes annotated frames via Writer.
    """
    start_time = time.time()
    
    # helpful logs
    logger.debug("Pipeline started.")
    logger.debug(f"Face detection enabled: {TOGGLE_FACE}")
    logger.debug(f"Age/Gender detection enabled: {TOGGLE_AGE_GENDER}")
    logger.debug(f"Gaze detection enabled: {TOGGLE_AGE_GENDER}")
    logger.debug(f"Writer enabled: {TOGGLE_WRITER}")
    logger.debug(f"Output video enabled: {WRITE_VIDEO}")
    logger.debug(f"Output annotation csv enabled: {WRITE_ANNO_CSV}")
    logger.debug(f"Output insights csv enabled: {WRITE_INSIGHTS_CSV}")

    # open the video
    video_path = os.path.join(TEST_VIDEO_DIR, TEST_VIDEO_NAME)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        raise RuntimeError(f"Cannot open video: {video_path}")
    logger.info("Video opened: %s", video_path)

    # get video dimensions
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # configure stride
    try:
        if TARGET_FPS_TO_PROCESS.lower() != "none" and int(TARGET_FPS_TO_PROCESS) > 0:
            stride = max(1, int(round(fps / float(TARGET_FPS_TO_PROCESS))))
            logger.debug("Using stride %d for target FPS %s", stride, TARGET_FPS_TO_PROCESS)
        else:
            stride = max(1, int(FRAME_STRIDE))
            logger.debug("Using configured frame stride: %d", stride)
    except Exception:
        stride = max(1, int(FRAME_STRIDE))
        logger.debug("Using configured frame stride: %d", stride)

    # Initialise the models
    peoplenet = PeopleNet()
    with mute_native_stderr():  # mute annoying logs to the terminal by MediaPipe
        if TOGGLE_FACE: mediapipe = MediaPipe()
    if TOGGLE_AGE_GENDER: caffe = Caffe()
    if TOGGLE_GAZE: gaze360 = Gaze360()

    # Initialise the Writer, to write the output video
    if TOGGLE_WRITER: writer = Writer(cap, fps, W, H)

    # Set up person_records, which will store our permanent data
    person_records = {}

    # set up progress bar for the terminal
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar_total = min(total_frames, MAX_FRAMES) if MAX_FRAMES else total_frames
    pbar = tqdm(total=pbar_total, desc="Detect+Track", unit="frame", file=sys.stdout) if pbar_total else None

    # Loop through each frame of the video
    frame_idx = 0
    processed_idx = 0
    populated_frames = 0
    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if pbar is not None:
            pbar.update(1)

        # End of video or read error
        if not ret:
            logger.info("End of video or read error at frame %d.", frame_idx)
            break
        if MAX_FRAMES is not None and frame_idx > MAX_FRAMES:
            logger.info("Reached MAX_FRAMES limit: %d", MAX_FRAMES)
            break

        # skip this frame if not in stride
        if frame_idx % stride != 0:
            continue
        processed_idx += 1
        logger.debug("Processing frame %d", frame_idx)

        # Make a FrameData object
        frame_data = FrameData(frame_idx, W, H)

        # Detect people on the frame
        person_views = peoplenet.process_frame(frame, frame_data)

        # If no people were detected, skip to the next frame
        if len(person_views) == 0:
            logger.debug("next frame ===============================================")
            continue
        populated_frames += 1
        logger.info("%d people detected in frame %d", len(person_views), frame_idx)

        # Loop through each detection of a person
        for person in person_views.values():
            # Get some attributes
            tid = person.tid

            # Make new Person Records for new people
            if tid not in person_records:
                person_records[tid] = PersonRecord(tid, person.frame)
            logger.info(f"Person record: {person_records[tid]}")
            logger.debug(f"All person records: {person_records}")

            # Face detection (updates person.face and ideally person.face.bbox)
            with mute_native_stderr():
                if TOGGLE_FACE: person = mediapipe.detect_face(person)

            # If no face, clear any stale gaze and skip
            # Safety Note: person_views only lasts for the scope of the frame,
            # and all its fields are specified upon intialisation (empty fields as None), so this is safe
            if person.face is None:
                if hasattr(person, "gaze_vec3"):
                    delattr(person, "gaze_vec3")
                if hasattr(person, "gaze_frame_idx"):
                    delattr(person, "gaze_frame_idx")
                logger.debug("Frame %d, Person %d: Face detection failed", frame_idx, getattr(person, "tid", -1))
                continue
            logger.info("Frame %d, Person %d: Face detected.", frame_idx, getattr(person, "tid", -1))

            # Ensure person.crop is the current face crop every frame
            face_crop = _safe_face_crop(frame, person.face.bbox)
            if person.face is not None and face_crop is None or face_crop.size == 0:
                # As above: clear stale gaze and skip
                if hasattr(person, "gaze_vec3"):
                    delattr(person, "gaze_vec3")
                if hasattr(person, "gaze_frame_idx"):
                    delattr(person, "gaze_frame_idx")
                logger.info("Frame %d, Person %d: Invalid face crop.", frame_idx, getattr(person, "tid", -1))
                continue
            person.crop = face_crop  # keep fresh

            # Eye detection
            with mute_native_stderr():
                person = mediapipe.detect_eyes(person)

            # Age & Gender
            if TOGGLE_AGE_GENDER:
                person = caffe.detect_age_gender(person)

                # Count age and gender in the tracker
                person.age_mode = person_records[tid].count_age(person.age_range)
                person.gender_mode = person_records[tid].count_gender(person.gender)

            # Gaze (compute & tag with current frame)
            if TOGGLE_GAZE and (isinstance(person.face.right_eye, np.ndarray) 
                                and isinstance(person.face.left_eye, np.ndarray)):
                person = gaze360.detect_gaze(person)

                # Set engagement to tracker
                if person.engagement is True:
                    person_records[tid].engagement = True
                    logger.info(f"Person's engagement has been logged in the tracker. \
                                Person record: {person_records[tid]}")

            # Draw arrow ONLY if it was produced for THIS frame
            if TOGGLE_GAZE and getattr(person, "gaze_frame_idx", None) == frame_idx and hasattr(person, "gaze_vec3"):
                frame = gaze360.draw(frame, person)

        if TOGGLE_WRITER:
            # Write the processed frame to the output video
            # Writer can be optimised by making one call per person, rather than looping though persons again
            if WRITE_VIDEO: writer.write_frame(frame, frame_idx, person_views)

            # Add to the csv for the current frame
            if WRITE_ANNO_CSV: writer.write_annotations_csv(person_views)

        # Optionally save frame and crops for this frame
        if SAVE_CROPS:
            save_crops(frame, person_views)

        logger.info("next frame ===============================================")

    # Processing finished: log summary
    logger.info("Processed %d frames.", processed_idx)
    logger.info(
        "Detected a total of %d people across %d frames.",
        len(person_records),
        populated_frames,
    )

    # Convert person records to dictionaries and JSONs for sending
    insight_dicts = [] # used for writing the insights csv
    insight_jsons = [] # used for sending through API
    for record in person_records.values():
        insight_dict = record.export_dict()
        insight_dicts.append(insight_dict)
        person_json = json.dumps(insight_dict)
        insight_jsons.append(person_json)
        person_json_pretty = json.dumps(insight_dict, indent=1, sort_keys=True)
        logger.info("Final Record json: %s", person_json_pretty)

    # Write CSV for person records
    if TOGGLE_WRITER and WRITE_INSIGHTS_CSV and len(person_records) > 0:
        writer.write_insights_csv(insight_dicts)

    # Release resources
    cap.release()
    if TOGGLE_WRITER: writer.release()
    if pbar is not None:
        pbar.close()
    logger.info("Released video resources.")

    # Print time statistics
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    milliseconds = int((elapsed - int(elapsed)) * 1000)
    logger.info("Pipeline finished in %02d:%02d.%03d (mm:ss.ms)", minutes, seconds, milliseconds)

    # Return our insights
    return insight_jsons

if __name__ == "__main__":
    pipeline()
