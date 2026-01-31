import React, { useRef, useEffect, useState, useCallback } from "react";
import axios from "axios";

// --- Inline Styles ---
const cssStyles = `
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background-color: #f3f4f6;
  color: #1f2937;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background-color: #0f172a;
  color: white;
  padding: 1rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.title {
  font-size: 1.25rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
}

.icon-red { color: #ef4444; }

.stats {
  font-family: monospace;
  font-size: 0.8rem;
  color: #cbd5e1;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.stats span {
  background: rgba(255,255,255,0.1);
  padding: 2px 8px;
  border-radius: 4px;
}

.main-content {
  flex: 1;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
}

.control-bar {
  background: white;
  padding: 1rem;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  display: flex;
  gap: 1rem;
  width: 100%;
  max-width: 700px;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
}

.button-group, .status-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  border: none;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background-color: #2563eb;
  color: white;
}
.btn-primary:hover:not(:disabled) { background-color: #1d4ed8; }

.btn-secondary {
  background-color: #e5e7eb;
  color: #374151;
}
.btn-secondary:hover:not(:disabled) { background-color: #d1d5db; }

.btn-dark {
  background-color: #1f2937;
  color: white;
}
.btn-dark:hover:not(:disabled) { background-color: #000; }

/* Status Badges - ALL STATES */
.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 700;
  text-transform: uppercase;
}

.status-standing { background-color: #dcfce7; color: #166534; }
.status-sitting { background-color: #dbeafe; color: #1e40af; }
.status-bending { background-color: #fef3c7; color: #92400e; }
.status-falling { background-color: #ffedd5; color: #9a3412; }
.status-fallen { background-color: #fee2e2; color: #991b1b; }
.status-lying { background-color: #f3e8ff; color: #7c3aed; }

.detection-badge {
  font-size: 0.7rem;
  padding: 2px 8px;
  background: #e5e7eb;
  border-radius: 4px;
  color: #374151;
}

.video-wrapper {
  position: relative;
  background-color: black;
  border-radius: 0.5rem;
  overflow: hidden;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
  width: 640px;
  height: 480px;
}

.hidden-video {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.main-canvas {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.replay-media {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: black;
}

.overlay-alert {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(239, 68, 68, 0.2);
  backdrop-filter: blur(4px);
  z-index: 10;
}

.alert-box {
  background-color: rgba(255, 255, 255, 0.95);
  padding: 1.5rem 2rem;
  border-radius: 0.75rem;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  text-align: center;
  max-width: 300px;
}

.alert-box h2 {
  color: #dc2626;
  margin: 0 0 0.5rem 0;
  font-size: 1.5rem;
}

.alert-box p {
  color: #4b5563;
  margin: 0;
}

.overlay-badge {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background-color: #dc2626;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 700;
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  z-index: 20;
}

.info-panel {
  background: white;
  padding: 1rem;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 700px;
}

.info-panel h3 {
  margin: 0 0 0.5rem 0;
  font-size: 0.9rem;
  color: #374151;
}

.state-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.state-legend span {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 4px;
}

.instructions {
  color: #6b7280;
  font-size: 0.8rem;
  line-height: 1.5;
  margin: 0;
}

.error-message {
  color: #dc2626;
  background: #fee2e2;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  border: 1px solid #fecaca;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e5e7eb;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .5; }
}

.frame-slideshow {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.frame-slideshow img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

@media (max-width: 700px) {
  .video-wrapper {
    width: 100%;
    height: auto;
    aspect-ratio: 4/3;
  }
  
  .control-bar, .info-panel {
    flex-direction: column;
    align-items: stretch;
  }
  
  .button-group, .status-group {
    justify-content: center;
  }
}
`;

// Icons
const Activity = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
  </svg>
);

const Upload = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const RotateCcw = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
  </svg>
);

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const processingCanvasRef = useRef(document.createElement("canvas")); // Offscreen canvas
  const replayMediaRef = useRef(null);
  const fileInputRef = useRef(null);

  const [sourceType, setSourceType] = useState("camera");
  const [isProcessing, setIsProcessing] = useState(false);
  const [fallDetected, setFallDetected] = useState(false);
  const [currentState, setCurrentState] = useState("STANDING");
  const [detectionMethod, setDetectionMethod] = useState("initializing");
  const [isReplaying, setIsReplaying] = useState(false);
  const [isLoadingReplay, setIsLoadingReplay] = useState(false);
  const [replayData, setReplayData] = useState(null);
  const [replayFrameIndex, setReplayFrameIndex] = useState(0);
  const [stats, setStats] = useState({ fps: 0, detectionTime: 0 });
  const [error, setError] = useState(null);

  const POST_FALL_DELAY_MS = 2000;
  const BACKEND_URL = "http://localhost:5000";

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.src = "";
        await videoRef.current.play();
        setSourceType("camera");
        setIsProcessing(true);
        resetSystem();
      }
    } catch (err) {
      setError("Could not access camera. Please check permissions.");
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setError(null);
      const url = URL.createObjectURL(file);
      setSourceType("video");

      if (videoRef.current) {
        if (videoRef.current.srcObject) {
          videoRef.current.srcObject
            .getTracks()
            .forEach((track) => track.stop());
        }
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        videoRef.current.loop = true;
        videoRef.current
          .play()
          .catch((e) => console.error("Video play error:", e));
        setIsProcessing(true);
        resetSystem();
      }
      // Reset file input to allow selecting the same file again
      event.target.value = "";
    }
  };

  const resetSystem = async () => {
    setFallDetected(false);
    setIsReplaying(false);
    setIsLoadingReplay(false);
    setReplayData(null);
    setReplayFrameIndex(0);
    setCurrentState("STANDING");
    setDetectionMethod("initializing");
    setError(null);
    try {
      await axios.post(`${BACKEND_URL}/reset`);
    } catch (e) {
      console.error("Reset error:", e);
    }
  };

  const fetchReplay = useCallback(async () => {
    try {
      setIsLoadingReplay(true);
      setIsProcessing(false);

      if (videoRef.current) videoRef.current.pause();

      const response = await axios.get(`${BACKEND_URL}/get_replay`, {
        timeout: 30000,
      });
      const data = response.data;

      if (data.type === "video" || data.type === "gif") {
        setReplayData({
          type: data.type,
          src: data.video,
          frameCount: data.frame_count,
        });
        setIsReplaying(true);
      } else if (data.type === "frames" && data.frames?.length > 0) {
        setReplayData({
          type: "frames",
          frames: data.frames,
          frameCount: data.frame_count,
        });
        setReplayFrameIndex(0);
        setIsReplaying(true);
      } else {
        setError(data.error || "No replay available");
        resumeLive();
      }
    } catch (error) {
      setError("Failed to load replay");
      resumeLive();
    } finally {
      setIsLoadingReplay(false);
    }
  }, []);

  const resumeLive = useCallback(() => {
    setIsReplaying(false);
    setIsLoadingReplay(false);
    setReplayData(null);
    setReplayFrameIndex(0);
    setFallDetected(false);
    setError(null);
    if (videoRef.current) videoRef.current.play();
    setIsProcessing(true);
  }, []);

  // Trigger replay fetch after fall detected and delay
  useEffect(() => {
    let timeoutId;
    if (fallDetected && !isReplaying && !isLoadingReplay) {
      timeoutId = setTimeout(() => {
        fetchReplay();
      }, POST_FALL_DELAY_MS);
    }
    return () => clearTimeout(timeoutId);
  }, [fallDetected, isReplaying, isLoadingReplay, fetchReplay]);

  // Main detection loop
  useEffect(() => {
    let intervalId;
    let timeoutId;
    let isActive = true;

    const processFrame = async () => {
      if (!isActive || !isProcessing || isReplaying || isLoadingReplay) return;
      if (!videoRef.current || !canvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      // We do not draw to the visible canvas here to avoid flickering/vibration
      // context.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

      const startTime = performance.now();

      try {
        // Capture frame using offscreen canvas
        const pCanvas = processingCanvasRef.current;
        pCanvas.width = canvas.width;
        pCanvas.height = canvas.height;
        const pContext = pCanvas.getContext("2d");
        pContext.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frameData = pCanvas.toDataURL("image/jpeg", 0.7);

        const response = await axios.post(
          `${BACKEND_URL}/detect_fall`,
          { frame: frameData },
          { timeout: 5000 }
        );

        if (!isActive) return;

        if (response.data.annotated_frame) {
          const img = new Image();
          img.onload = () => {
            if (isActive && canvasRef.current) {
              canvasRef.current
                .getContext("2d")
                .drawImage(img, 0, 0, canvas.width, canvas.height);
            }
          };
          img.src = response.data.annotated_frame;
        }

        setCurrentState(response.data.state || "STANDING");
        setDetectionMethod(response.data.detection_method || "unknown");

        if (response.data.fall_detected && !fallDetected) {
          setFallDetected(true);
          // Timeout logic moved to separate useEffect to prevent cleanup race condition
        }

        const frameTime = performance.now() - startTime;
        setStats({
          fps: Math.round(1000 / Math.max(frameTime, 1)),
          detectionTime: Math.round(frameTime),
        });
      } catch (error) {
        if (error.code !== "ECONNABORTED") {
          console.error("Detection error:", error.message);
        }
      }
    };

    if (isProcessing && !isReplaying && !isLoadingReplay) {
      intervalId = setInterval(processFrame, 100);
    }

    return () => {
      isActive = false;
      clearInterval(intervalId);
      clearTimeout(timeoutId);
    };
  }, [isProcessing, isReplaying, isLoadingReplay, fallDetected, fetchReplay]);

  // Auto-play replay
  useEffect(() => {
    if (
      isReplaying &&
      replayMediaRef.current &&
      replayData &&
      (replayData.type === "video" || replayData.type === "gif")
    ) {
      replayMediaRef.current.play().catch(() => {});
    }
  }, [isReplaying, replayData]);

  // Frame slideshow
  useEffect(() => {
    let frameInterval;
    if (isReplaying && replayData?.type === "frames") {
      frameInterval = setInterval(() => {
        setReplayFrameIndex((prev) => (prev + 1) % replayData.frames.length);
      }, 100);
    }
    return () => clearInterval(frameInterval);
  }, [isReplaying, replayData]);

  const getStatusClass = (state) => {
    if (state && state.startsWith("BUFFERING")) return "status-bending"; // Use yellow for buffering

    const map = {
      STANDING: "status-standing",
      SITTING: "status-sitting",
      BENDING: "status-bending",
      FALLING: "status-falling",
      FALLEN: "status-fallen",
      LYING: "status-lying",
    };
    return map[state] || "status-standing";
  };

  return (
    <>
      <style>{cssStyles}</style>
      <div className="app-container">
        <header className="app-header">
          <div className="header-content">
            <h1 className="title">
              <Activity className="icon-red" />
              Fall Detection System
            </h1>
            <div className="stats">
              <span>FPS: {stats.fps}</span>
              <span>Latency: {stats.detectionTime}ms</span>
              <span>Method: {detectionMethod}</span>
            </div>
          </div>
        </header>

        <main className="main-content">
          {error && <div className="error-message">{error}</div>}

          <div className="control-bar">
            <div className="button-group">
              <button
                onClick={startCamera}
                className={`btn ${
                  sourceType === "camera" && isProcessing
                    ? "btn-primary"
                    : "btn-secondary"
                }`}
                disabled={isLoadingReplay}
              >
                Camera
              </button>
              <button
                onClick={() => fileInputRef.current.click()}
                className={`btn ${
                  sourceType === "video" ? "btn-primary" : "btn-secondary"
                }`}
                disabled={isLoadingReplay}
              >
                <Upload />
                Load Video
              </button>
              <input
                type="file"
                ref={fileInputRef}
                accept="video/*"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
            </div>

            <div className="status-group">
              <div className={`status-badge ${getStatusClass(currentState)}`}>
                {currentState}
              </div>
              <span className="detection-badge">{detectionMethod}</span>

              {(isReplaying || isLoadingReplay) && (
                <button
                  onClick={resumeLive}
                  className="btn btn-dark"
                  disabled={isLoadingReplay}
                >
                  <RotateCcw /> Resume
                </button>
              )}
            </div>
          </div>

          <div className="video-wrapper">
            <video
              ref={videoRef}
              playsInline
              muted
              className="hidden-video"
              width={640}
              height={480}
            />

            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="main-canvas"
              style={{ display: isReplaying ? "none" : "block" }}
            />

            {isReplaying && replayData && (
              <>
                {(replayData.type === "video" || replayData.type === "gif") && (
                  <video
                    ref={replayMediaRef}
                    src={replayData.src}
                    className="replay-media"
                    controls
                    loop
                    muted
                    playsInline
                  />
                )}
                {replayData.type === "frames" && replayData.frames && (
                  <div className="frame-slideshow">
                    <img
                      src={replayData.frames[replayFrameIndex]}
                      alt={`Frame ${replayFrameIndex + 1}`}
                    />
                  </div>
                )}
              </>
            )}

            {fallDetected && !isReplaying && !isLoadingReplay && (
              <div className="overlay-alert">
                <div className="alert-box">
                  <h2>⚠️ FALL DETECTED</h2>
                  <p>Capturing footage...</p>
                </div>
              </div>
            )}

            {isLoadingReplay && (
              <div className="overlay-alert">
                <div className="alert-box">
                  <div className="spinner"></div>
                  <h2>Loading Replay</h2>
                </div>
              </div>
            )}

            {isReplaying && <div className="overlay-badge">🔄 REPLAY</div>}
          </div>

          <div className="info-panel">
            <h3>State Legend</h3>
            <div className="state-legend">
              <span className="status-standing">STANDING</span>
              <span className="status-sitting">SITTING</span>
              <span className="status-bending">BENDING</span>
              <span className="status-falling">FALLING</span>
              <span className="status-fallen">FALLEN</span>
              <span className="status-lying">LYING</span>
            </div>
            <p className="instructions">
              <strong>Hybrid Detection:</strong> Uses MediaPipe pose estimation
              when available, falls back to contour analysis for overhead
              cameras.
              <br />
              <strong>LYING state:</strong> Detected when person starts video
              already in horizontal position (not a fall event).
              <br />
              <strong>FALLEN state:</strong> Triggered after detecting rapid
              descent followed by sustained horizontal position.
            </p>
          </div>
        </main>
      </div>
    </>
  );
};

export default App;
