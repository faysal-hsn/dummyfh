"""
Custom Dataclasses used to transfer data between different modules in the pipeline.
"""
import configparser
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import numpy as np

# Get config
config = configparser.ConfigParser()
config.read("config.cfg")
AGE_BUCKETS = config["caffe"]["AGE_BUCKETS"].split(", ")

@dataclass(frozen=True)
class FrameData:
    """
    A frozen dataclass representing a single frame of video data.
    """
    idx: int
    W: int
    H: int

@dataclass()
class PersonView:
    """
    A dataclass representing a detection of a person,
    for passing data along the pipeline and between modules.

    This is temporary storage that only lasts for the scope of a frame.
    For permament storage, see PersonRecord below.
    """
    # Attributes set upon initalisation
    tid: int
    frame: FrameData
    score: float
    bbox: np.ndarray
    crop: np.ndarray
    engagement: bool = False # if this person view shows an engagement with the target area, set to True

    # Face detection
    face: Optional["FaceView"] = None

    # Gaze detection
    gaze_yaw_deg: float = None
    gaze_pitch_deg: float = None

    # age and gender
    age_range: Optional[str] = None
    age_confidence: float = None
    gender: Optional[Literal["Male", "Female", "unsure"]] = None
    gender_confidence: float = None
    age_mode: Optional[str] = None
    gender_mode: Optional[Literal["Male", "Female"]] = None

@dataclass(slots=True)
class FaceView:
    """
    A frozen dataclass representing a detection of a face,
    for passing data along the pipeline and between modules.
    It lasts for the scope of a single frame.

    The bbox to the person crop.
    """
    bbox: np.ndarray  # (x1, y1, x2, y2)
    crop: np.ndarray
    facegrid: np.ndarray
    right_eye: np.ndarray = None
    left_eye: np.ndarray = None

@dataclass(slots=True)
class PersonRecord:
    """
    A permanent dataclass that represents a unique person.
    """
    tid: int
    frame: FrameData

    # Demographic tallies (label-agnostic; tolerate any string labels)
    _age_counts: dict = field(default_factory=lambda: dict.fromkeys(AGE_BUCKETS, 0))
    _gender_counts: dict = field(default_factory=lambda: dict.fromkeys(["Male", "Female"], 0))
    age_mode: Optional[str] = None
    gender_mode: Optional[Literal["Male", "Female"]] = None

    # this tracks if the person has engaged in *any* frame
    engagement: bool = False

    def count_age(self, age_label: str | None):
        """Increment age bucket and refresh mode (ignored if None/empty)."""
        if not age_label:
            return self.age_mode
        if age_label == "unsure":
            return self.age_mode
        if not self._age_counts.get(age_label, False):
            self._age_counts[age_label] = 0
            
        self._age_counts[age_label] += 1
        if self.age_mode is None or (self._age_counts[age_label] >= self._age_counts[self.age_mode]):
            self.age_mode = age_label

        return self.age_mode

    def count_gender(self, gender_label: str | None):
        """Increment gender bucket and refresh mode (ignored if None/empty)."""
        if not gender_label:
            return self.gender_mode
        if gender_label == "unsure":
            return self.gender_mode
        
        self._gender_counts[gender_label] += 1
        if self.gender_mode is None or (self._gender_counts[gender_label] >= self._gender_counts[self.gender_mode]):
            self.gender_mode = gender_label

        return self.gender_mode

    def export_dict(self):
        return asdict(self)
