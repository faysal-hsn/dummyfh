# Gaze Model Labeling Instructions (Video -> Face crops + .label spec)

## Purpose
Please create face-crop images from our videos and a plain-text label file (.label) with the gaze direction for each crop. Our training code reads the crops and their (pitch, yaw) angles to train a gaze-estimation model.

## Folder structure to deliver
Please send a single zip with the following structure (these names are preferred):
project_root/
  faces/                       # REQUIRED: face crops (JPG/PNG)
    <videoId>_<frame>_<id>.jpg
    ...
  labels/
    train.label                # REQUIRED: label file (UTF-8, plain text)
    val.label                  # OPTIONAL: same format for validation
  frames/                      # OPTIONAL: extracted full frames for reference

## Face crop requirements
1. One person per image. If a frame has 3 people, make 3 crops.
2. Crop tightly around the face with a small margin (include forehead and chin).
3. Face should be upright (no heavy rotation) and sharp (not blurry).
4. Preferable size 224x224 px; 256x256 or larger is better.
5. File format: JPG (quality 90%+) or PNG, sRGB color.
6. Filenames: {videoId}_{frameIndex}_{subjectId}.jpg (example: cam01_000123_A.jpg).

## Angle definition
We label pitch and yaw in radians.
- Yaw: left is positive, right is negative.
- Pitch: up is negative, down is positive.
Typical ranges:
- Yaw in [-pi/2, +pi/2]
- Pitch in [-pi/4, +pi/4]
Benchmarks:
- Yaw 0; Pitch 0: A person gazing directly forward (in the direction of the camera), with their head level.
- Yaw pi/2 (90 degrees): A person gazing directly to the left, perpendicular to the camera. 
- Yaw -pi/2 (-90 degrees): A person gazing directly to the right, perpendicular to the camera.
- Pitch pi/4 (45 degrees): A person gazing upwards at a 45 degree angle.
- Pitch -pi/4 (-45 degrees): A person gazing downwards at a 45 degree angle.
- Yaw and pitch are calculated relative to the camera's perspective. Ie. Regardless of the position of a person's body, a person looking towards the left side of the image will have a positive yaw, and a person looking towards the right side of the image will have a negative yaw.
- As yaw and pitch are 3D angles, please use the ground in the image as a benchmark to estimate them. They will not line up exactly with 2D angles on the image.
If you work in degrees, convert to radians before writing the labels: `radians = degrees x pi / 180`

## .label file format
Plain-text UTF-8, space-separated, with a single header line (ignored by our loader). Each subsequent line represents one face crop (one person).

Columns (space-separated):
1. face_rel_path        -> path to the face crop relative to the image root we provide (usually the faces/ directory)
2. left_eye_rel         -> use "-" (placeholder; not used)
3. right_eye_rel        -> use "-" (placeholder; not used)
4. subject_id_or_name   -> free text for your bookkeeping
5. unused               -> use "_" (placeholder)
6. pitch_rad,yaw_rad    -> two floats separated by a comma (no spaces)

Notes:
- Only columns 1 and 6 are used by training; the others are placeholders.
- The first line is a header and will be ignored.
- Keep all image paths relative to the agreed image root (we use faces/ unless we tell you otherwise).
- Do not insert a space after the comma in pitch_rad,yaw_rad (example: 0.0873,-0.3491).

## Example directory tree (minimal)
project_root/
  faces/
    cam01_000121_A.jpg
    cam01_000121_B.jpg
    cam01_000122_A.jpg
    cam02_000451_A.jpg
  labels/
    train.label

## Example: train.label (copy-paste ready)
Header line (ignored by the loader):
# face lefteye righteye subject unused pitch_rad,yaw_rad

Option A (image root = project_root/faces -> face_rel_path is just the filename):
``` 
cam01_000121_A.jpg - - subjA _ 0.0873,-0.3491
cam01_000121_B.jpg - - subjB _ -0.0524,0.2618
cam01_000122_A.jpg - - subjA _ 0.0000,0.0000
cam02_000451_A.jpg - - subjX _ 0.1222,0.1745
cam02_000452_A.jpg - - subjX _ -0.0349,-0.1047
cam02_000452_B.jpg - - subjY _ 0.0698,0.0873
cam02_000453_A.jpg - - subjX _ -0.1571,0.0000
cam02_000453_B.jpg - - subjY _ 0.0000,-0.2618
cam02_000454_A.jpg - - subjX _ 0.1047,0.2094
cam02_000454_B.jpg - - subjY _ -0.0873,0.0698
```
Option B 
(image root = project_root -> keep the faces/ prefix in face_rel_path; eye paths shown for completeness but are ignored by training):
``` 
faces/cam01_000121_A.jpg eyes_left/cam01_000121_A_L.jpg eyes_right/cam01_000121_A_R.jpg subjA _ 0.0873,-0.3491
faces/cam01_000121_B.jpg eyes_left/cam01_000121_B_L.jpg eyes_right/cam01_000121_B_R.jpg subjB _ -0.0524,0.2618
```
## Multiple subjects in the same frame
If one frame has multiple people, create one face crop per person and add one line per crop (for example, suffixes _A and _B). Do not reuse the same full-frame image path for multiple people; we need person-specific crops.

## Quality and consistency checklist
- One person per crop; face clearly visible, minimal occlusion.
- Crops are tight and consistently framed across the dataset.
- Paths in train.label match files exactly and are relative to the agreed image root.
- Angles are in radians and follow the sign convention (yaw: L-/R+, pitch: up-/down+).
- Header line present on the first line of train.label.
- Encoding is UTF-8; LF or CRLF line endings are both fine.
- No trailing spaces; use a comma with no space for angles (example: 0.0873,-0.3491).

## What to deliver
1. faces/ directory with all face crops (JPG/PNG).
2. labels/train.label (and labels/val.label if you split a validation set) in the exact format above.
3. Optional: a short README describing how you derived angles (tools, annotators).