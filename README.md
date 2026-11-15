# Automated Classroom Attendance System using Face Recognition

A lightweight, offline face recognition attendance system built using Python, OpenCV, SQLite, and Pandas for Semester 5 Computer Vision & Gen AI (Assignment 3).

The system replaces manual attendance by capturing video or image frames of a classroom, detecting and identifying students against a local biometric database, and recording attendance records to SQLite and dated CSV files.

---

## Technical Approach

Most Python face recognition tutorials rely on `dlib` or `face_recognition`, which require CMake and MSVC C++ build tools on Windows and frequently fail during installation. 

This project avoids external heavy dependencies by relying entirely on OpenCV's native deep learning modules (`cv2.dnn`):
- **Face Detection:** YuNet (`cv2.FaceDetectorYN`) detects multiple faces per frame and outputs 5 facial landmarks (eyes, nose, mouth corners).
- **Face Recognition:** SFace (`cv2.FaceRecognizerSF`) aligns each detected face to a 112x112 canonical crop and extracts a 128-dimensional L2-normalized embedding.
- **Biometric Matching:** Computes cosine similarity between live query vectors and stored embeddings. Matches above the threshold (default 0.363) are marked present.
- **Temporal Voting Filter:** A student must be detected in at least 3 separate frames during a scanning session to be marked present, filtering out transient misdetections and motion blur.
- **Dual Persistence:** Stores historical data in SQLite (`data/attendance.db`) and automatically generates a CSV file and proof snapshot per session.

---

## System Workflow

```
[ Classroom Camera / Video Feed / Photo ]
                   │
                   ▼
       [ FaceDetectorYN (YuNet) ]
  Locates bounding boxes and 5 facial landmarks
                   │
                   ▼
      [ FaceRecognizerSF (SFace) ]
  Aligns face crop and generates 128-d feature embedding
                   │
                   ▼
     [ Vectorized Cosine Matcher ]
  Compares embedding against enrolled SQLite database
                   │
                   ▼
     [ Multi-Frame Temporal Voting ]
  Requires >= 3 positive detections during session window
                   │
                   ▼
      [ Attendance Persistence ]
  - SQLite: relational record with session ID & timestamp
  - CSV: administrative log in data/exports/
  - JPG: annotated snapshot saved as verification proof
```

---

## Project Structure

```
face-recognition-attendance-system/
├── config.py                 # Paths, model parameters, and similarity thresholds
├── db.py                     # SQLite database schema, CRUD operations, and queries
├── download_models.py        # Script to download official OpenCV Zoo ONNX weights
├── recognizer.py             # Core wrapper for YuNet and SFace
├── enroll.py                 # CLI tool to register students via webcam, folder, or sample seed
├── run_session.py            # Main attendance scanner for live webcams, videos, or photos
├── report.py                 # CLI to view session summaries and attendance percentages
├── evaluate.py               # Benchmark script testing accuracy, thresholds, and stress cases
├── demo_detection.py         # Visual demo testing YuNet face detection alone
├── demo_recognition.py       # Visual demo testing SFace matching alone
├── learnings.md              # Technical notes, design decisions, and system trade-offs
├── requirements.txt          # Python dependencies
├── models/                   # ONNX model files (downloaded via download_models.py)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── data/                     # Local data directory
│   ├── attendance.db         # SQLite database
│   ├── known_faces/          # Stored enrollment photos
│   ├── test_images/          # Sample images used for testing
│   └── exports/              # Generated attendance CSVs, proof images, and plots
└── tests/                    # Pytest test suite (24 unit tests)
    ├── test_db_and_enroll.py
    ├── test_evaluate.py
    ├── test_recognition.py
    ├── test_recognizer.py
    └── test_session_and_report.py
```

---

## Setup Instructions

### 1. Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

Download the official YuNet and SFace ONNX models from the OpenCV Zoo:

```bash
python download_models.py
```

This downloads the two model files into the `models/` directory and verifies their loading integrity.

### 3. Student Enrollment

The system needs registered students in the database before running attendance.

```bash
# Seed 10 sample student profiles for testing
python enroll.py --seed-sample-students

# Or enroll a new student using your webcam (captures varied poses)
python enroll.py --webcam --roll CS201 --name "Rohan Saini"

# Or enroll from a folder containing photos of a student
python enroll.py --folder path/to/images --roll CS101 --name "Aarav Sharma"

# View all currently enrolled students
python enroll.py --list
```

---

## How to Run Attendance

### Running a Session

```bash
# Live webcam scan (15-second scanning window, requiring 3 detections to confirm)
python run_session.py --webcam --course "CS-301 Computer Vision" --duration 15 --min-frames 3

# Run on a recorded video file
python run_session.py --video data/test_images/classroom_sim.mp4 --course "CS-301 Computer Vision"

# Run on a single classroom image
python run_session.py --image data/test_images/sample_face.jpg --course "CS-301 Computer Vision" --no-display

# For large lecture halls with distant faces, use tiling and upscaling
python run_session.py --image classroom_group.jpg --course "CS-301" --tile-size 640 --upscale 1.5
```

When a session completes, it prints a terminal summary, writes records to `data/attendance.db`, saves a CSV log in `data/exports/`, and saves an annotated verification image.

### Viewing Reports

```bash
# View the summary of the most recent session
python report.py --latest

# View a list of all historical sessions
python report.py --sessions

# View cumulative attendance percentages per student (flags <75% attendance)
python report.py --student-summary
```

---

## System Evaluation & Benchmark Results

The system includes an evaluation module (`evaluate.py`) that tests recognition accuracy across 91 labeled probes (50 genuine student attempts and 41 unregistered impostor attempts).

### Verification Metrics (Threshold = 0.363)

| Metric | Result | Description |
| :--- | :--- | :--- |
| Accuracy | 98.9% | Correct predictions across genuine and impostor attempts |
| Precision | 98.0% | Ratio of true matches over all positive identifications |
| Recall | 100.0% | All enrolled students correctly detected |
| F1-Score | 0.9901 | Harmonic mean of precision and recall |
| False Accept Rate (FAR) | 2.4% | Percentage of unregistered faces incorrectly matched |
| False Reject Rate (FRR) | 0.0% | Percentage of registered faces rejected |
| Average CPU Latency | 26.6 ms | ~37.6 FPS throughput on a standard laptop CPU |

### Error Trade-Off Curve

Run the threshold sweep:

```bash
python evaluate.py
```

This exports `data/exports/threshold_sweep.png`, showing False Accept Rate vs False Reject Rate across similarity thresholds from 0.10 to 0.90.

### Stress Testing & Limitations

| Condition | Result | Behavior |
| :--- | :--- | :--- |
| Standard Frontal View | PASS | Clear detection and matching (confidence ~1.00) |
| Low Light (0.35x brightness) | PASS | Face detected and matched (confidence ~0.94) |
| High Glare / Overexposure | FAIL | Landmarks washed out; detector skips image |
| Minor Head Tilt (+/- 15 deg) | PASS | 5-point affine alignment corrects rotation |
| Moderate Head Tilt (+/- 30 deg) | FAIL | Landmarks become occluded below detector threshold |
| Distant Small Face (~40px) | PASS | Detectable; confidence improves with `--upscale 1.5` |
| Face Mask Occlusion | FAIL | Lower landmarks covered; prevents false matching |
| Dark Sunglasses Occlusion | FAIL | Eye landmarks covered; skipped by detector |
| Multi-Face Frame | PASS | Detects and identifies multiple students concurrently |
| Unenrolled Impostor | PASS | Cosine similarity stays below 0.363; labeled Unknown |

---

## Unit Tests

Run the test suite to verify the database, detection, embedding, and reporting functions:

```bash
python -m pytest tests/ -v
```

All 24 unit tests pass without requiring a GPU or camera device.

---

## Assignment Deliverables Summary

- **Database with >= 10 Students:** 11 students enrolled in SQLite with 128-d biometric embeddings.
- **Automated CSV Log:** Generates timestamped logs with roll number, name, status, confidence, and date.
- **Evaluation of Accuracy and Limitations:** Documented with metrics, threshold sweeps, and stress-test failure modes.
- **Tech Stack:** Python, OpenCV (YuNet + SFace), SQLite3, Pandas.

---

## Author

Rohan Saini  
GitHub: [rohansaini-02](https://github.com/rohansaini-02)
