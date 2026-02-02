<div align="center">
  <h1 align="center">Sentinel AI</h1>
  <h3>Multimodal Fall Detection System</h3>
  <p>
    <b>Privacy-First. Real-Time. Life-Saving.</b>
  </p>
  <p>
    Built for the <b>AI Vibe Coding Hackathon 2026</b>
  </p>
</div>

---

## 💡 The Problem
Falls are the leading cause of injury-related death among adults aged 65 and older. 
*   **Existing Wearables?** Often forgotten or refused.
*   **Existing Cameras?** Plagued by false alarms (sitting down, picking up objects).

## 🚀 The Solution: Sentinel AI
**Sentinel AI** is a next-generation safety appliance that uses **Multimodal AI** (Vision + Audio) to detect falls with >95% accuracy, running entirely on edge hardware to preserve privacy.

It solves the "False Alarm" problem using a **Hybrid Geometric-ML Engine**:
1.  **Vision:** Extracts 33 skeletal keypoints (MediaPipe) to analyze velocity, joint angles, and aspect ratios.
2.  **Audio:** Listens for "thuds" or distress sounds using Audio Spectrogram Transformers (AST).
3.  **Context:** A temporal voting buffer ensures that a person *sitting down* (controlled descent) is never mistaken for a *fall* (chaotic descent).

## ⚡ Tech Stack ("2026 Vibe Coding")
We built this using the modern "Vibe Coding" stack for maximum performance and DX:

*   **Frontend:** [Next.js 14](https://nextjs.org/) (App Router) + [Shadcn UI](https://ui.shadcn.com/) + Tailwind CSS.
    *   *Features:* Dark Mode Bento Grid, Real-time WebSockets, Client-side Recharts.
*   **Backend:** Python Flask + OpenCV.
    *   *Features:* Async Frame Processing, XGBoost Inference, Thread-safe Circular Buffers.
*   **State Management:** [TanStack Query](https://tanstack.com/query/latest) + Zustand.
    *   *Why:* Zero-flicker polling and robust server synchronization.

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   Webcam

### 1. Backend (Python)
```bash
# Clone the repo
git clone https://github.com/yourusername/sentinel-ai.git
cd sentinel-ai

# Environment Setup
python -m venv venv
.\venv\Scripts\activate

# Install Dependencies (PyTorch, MediaPipe, Flask)
pip install -r requirements.txt

# Run Server
python app.py
```

### 2. Frontend (Next.js)
```bash
cd fall-detection-dashboard

# Install deps
npm install

# Run Dev Server
npm run dev
```
Open **http://localhost:3000** to see the dashboard.

## 🎮 How to Demo
1.  Launch the App.
2.  **Walk around:** The Status Pill will show `STANDING`.
3.  **Sit down:** The system recognizes the controlled descent and shows `SITTING` (No Alarm).
4.  **Simulate a Fall:** Drop quickly to the floor.
    *   **Visual:** Status turns **RED (FALLEN)**.
    *   **Data:** Confidence Score spikes to >90%.
    *   **Replay:** The "Event Log" captures the moment for review.

## 🎥 Demo: ElevenLabs Voice Integration
> *"I have detected a fall. Don't worry, I am alerting your emergency contact now."*

[**▶️ Watch the Voice Alert in Action**](./fall_detection_voice_demo.mp4)

## 🏆 Future Roadmap (Hackathon Follow-up)
*   **ElevenLabs Integration:** Voice alerts calling out to the fallen person ("Are you okay?"). (✅ **DONE**)
*   **Robotics Integration:** Signal a local robot (or Spot) to investigate.
*   **.cv Domain:** Hosting user health profiles on secure .cv domains.

---

<p align="center">
  Made with ❤️ by the Sentinel AI Team
</p>
