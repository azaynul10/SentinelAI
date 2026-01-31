# SentinelAI: Robust Fall Detection System

**SentinelAI** is a privacy-first, real-time fall detection system designed for elderly care monitoring. It uses a **Hybrid Geometric-ML** approach to achieve high accuracy (`>95%`) while running efficiently on local hardware (CPU/GPU).

## 🚀 Key Features

*   **Hybrid Detection Engine**: Combines **MediaPipe Pose Physics** (Velocity, Angles, Aspect Ratio) with **XGBoost** classification.
*   **False Positive Mitigation**: 
    *   **Anti-Walking Guard**: Vertical aspect ratio checks prevent walking from triggering falls.
    *   **Sitting vs. Bending**: Strict knee-angle logic distinguishes safe bending from crumpling.
    *   **Jitter Filter**: 30-frame temporal smoothing eliminates sensor noise.
*   **Actionable Alerts**:
    *   **Instant Replay**: Automatically captures and saves the last 15 seconds of a fall event for review.
    *   **Stillness Gating**: Only alarms after confirmed post-fall stillness (~1 sec).
*   **Privacy First**: All processing happens **Locally**. No video is sent to the cloud.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Node.js 18+ (for Dashboard)
*   Webcam

### 1. Backend Setup (Python)
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup (Next.js)
```bash
cd fall-detection-dashboard
npm install
```

---

## 🏃‍♂️ Usage

### Start the System
1.  **Start Backend** (Root Directory):
    ```bash
    python app.py
    ```
    *Server runs on http://localhost:5000*

2.  **Start Dashboard** (New Terminal):
    ```bash
    cd fall-detection-dashboard
    npm run dev
    ```
    *UI available at http://localhost:3000*

---

## 🧠 System Architecture

*   **Input**: Real-time Webcam Feed at 30 FPS.
*   **Processing (`pose_fall_detection.py`)**:
    1.  **Pose Extraction**: MediaPipe extracts 33 keypoints.
    2.  **Metric Calculation**: Computes velocity (`dy`), height ratios, and joint angles.
    3.  **Voting Window**: A 30-frame buffer weighs evidence (Visibility + Physics).
    4.  **State Machine**: Transitions between `STANDING` -> `FALLING` -> `FALLEN` based on strict gates.
*   **Output**: JSON stream to Frontend + `.mp4` Replay generation.

---

## 🔒 License
Private Repository.
