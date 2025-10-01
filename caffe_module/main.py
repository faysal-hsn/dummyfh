"""
The Age/Gender Detection Module
"""
import configparser
import logging
import os
import cv2

# Set up logger
logger = logging.getLogger("Caffe Module")

# Get config
config = configparser.ConfigParser()
config.read("config.cfg")
MODEL_DIR = config["caffe"]["MODEL_DIR"]
AGE_BUCKETS = config["caffe"]["AGE_BUCKETS"].split(", ")
logger.debug(f"Age Buckets: {AGE_BUCKETS}")

class Caffe:
    """
    Caffe model for age and gender detection.
    """
    def __init__(self):
        """
        Sets up the models
        """
        age_proto = os.path.join(MODEL_DIR, "age_deploy.prototxt")  # network structure
        age_model = os.path.join(MODEL_DIR, "age_net.caffemodel")  # trained weights

        # Gender prediction model files (Caffe format)
        gen_proto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")  # network structure
        gen_model = os.path.join(MODEL_DIR, "gender_net.caffemodel")  # trained weights

        # Load age and gender networks (Caffe format)
        self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        self.gen_net = cv2.dnn.readNetFromCaffe(gen_proto, gen_model)

    def detect_age_gender(self, person):
        """
        Detect age and gender from the person's face.
        Returns the person object, either with or without age and gender fields filled.
        """
        logger.info("Frame %d, Person %d: Running Caffe", person.frame.idx, person.tid)

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # mean values for age
        GENDERS = ["Male", "Female"]

        # Prepare the cropped face as blob for Caffe networks
        face_blob = cv2.dnn.blobFromImage(person.face.crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False)

        # Gender prediction
        self.gen_net.setInput(face_blob)
        gender_scores = self.gen_net.forward()[0]
        gender_idx = int(gender_scores.argmax())
        gender = GENDERS[gender_idx]
        gender_conf = float(gender_scores[gender_idx])

        # Age prediction
        self.age_net.setInput(face_blob)
        age_scores = self.age_net.forward()[0]
        age_idx = int(age_scores.argmax())
        age_range = AGE_BUCKETS[age_idx]
        age_conf = float(age_scores[age_idx])

        # Record age and gender information if confidence is high enough
        if gender_conf > 0.95:
            person.gender = gender
            person.gender_confidence = gender_conf
        else:
            person.gender = "unsure"
            person.gender_confidence = gender_conf

        if age_conf > 0.95:
            person.age_range = age_range
            person.age_confidence = age_conf
        else:
            person.age_range = "unsure"
            person.age_confidence = age_conf

        return person
