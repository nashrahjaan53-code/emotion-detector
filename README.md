# 🎭 Real-Time Emotion Detector

A lightweight, real-time emotion detection system using your webcam. Built with Python, OpenCV, and the DeepFace library, it analyzes facial expressions to classify emotions into happy, sad, angry, surprised, neutral, fearful, and disgusted states in real-time.

## Features
- **Real-Time Analysis**: Processes webcam frames dynamically and predicts emotion with minimal latency.
- **Adaptive Lighting Correction**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in the LAB color space to stabilize detection under poor or uneven lighting conditions.
- **Temporal Smoothing Buffer**: Employs a sliding buffer (voting system) to aggregate predictions over multiple frames, eliminating flickering or jitter in emotion labels.
- **On-Screen Dashboard**:
  - Colored bounding box around the detected face matching the emotion's representative color.
  - Confidence percentage bar graphs for all 7 standard emotions displayed on-screen.
  - A low-confidence warning to suggest better lighting if the model is unsure.
  - Interactive buffer/stability visualization.
- **Dual Support**: Includes a command-line script ([live_emotion.py](file:///c:/emotion-detector/live_emotion.py)) and an interactive Jupyter Notebook ([emotion_detector.ipynb](file:///c:/emotion-detector/emotion_detector.ipynb)) for exploration.

---

## Installation

### Prerequisites
- Python 3.8 or higher.
- A webcam.

### Setup

1. **Clone the repository or navigate to the directory**:
   ```bash
   cd emotion-detector
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Installing `deepface` will automatically download the backend deep learning framework (e.g., TensorFlow) and model weights on the first run.*

---

## Usage

### 1. Command-Line Script
Run the real-time webcam detector:
```bash
python live_emotion.py
```
- The camera window will pop up.
- Press **Q** on your keyboard inside the window to exit.

### 2. Jupyter Notebook
If you want to run the detector interactively cell-by-cell or explore the implementation:
```bash
pip install notebook
jupyter notebook
```
Open `emotion_detector.ipynb` and run the cells.

---

## Configuration

In [live_emotion.py](file:///c:/emotion-detector/live_emotion.py), you can fine-tune the parameters near the top of the file:
```python
BUFFER_SIZE   = 15      # Number of frames to store for smoothing (higher = more stable, slower response)
ANALYZE_EVERY = 3       # Run prediction on 1 out of every N frames (lower = more frequent, higher CPU usage)
MIN_CONFIDENCE = 60     # Minimum confidence score (0-100) required to register an emotion (otherwise falls back to neutral)
```

---

## How It Works
1. **Frame Capture**: Pulls frames from the default webcam via OpenCV.
2. **Preprocessing**: Boosts brightness and contrast of the face region using the CLAHE algorithm to handle varying light levels.
3. **Inference**: Feeds the image into `DeepFace.analyze` which utilizes deep learning models (like VGG-Face) with the fast OpenCV face detector backend.
4. **Filtering**: Stores the output in a FIFO queue and computes the mode (most frequent emotion) to determine the "stable emotion".
5. **Rendering**: Overlays bounding boxes, labels, real-time confidence scores, and a buffer indicator on the video stream.

---

## License
MIT License. Feel free to use and customize!
