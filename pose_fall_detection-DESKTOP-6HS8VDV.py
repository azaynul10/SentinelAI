import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import importlib.util
import tempfile
import os
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: xgboost could not be imported. ML confirmation will be disabled.")
import time
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

# Load external feature engineering modules dynamically
def load_module(mod_name, file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Module {mod_name} not found at {file_path}")
        return None
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Assume files are in current directory
per_frame = load_module("per_frame_best", "per-frame-best.py")
fe_eng = load_module("final_feature_eng_best", "final-feature-eng-best.py")

class AdvancedPoseFallDetector:
    """
    Hybrid Fall Detector features:
    1. Real-time heuristic detection (immediate response)
    2. XGBoost ML model with EXACT Feature Engineering (Async & Zero-Copy)
    """
    
    def __init__(self, model_path='xgb_final_model.json'):
        # thread lock for safety
        self.lock = threading.Lock()
        
        # 1. MediaPipe Setup (Heavy Mode for Accuracy)
        # 1. MediaPipe Setup (Heavy Mode for Accuracy)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = None # Lazy Init
        self.pose_options = {
            'static_image_mode': False,
            'model_complexity': 0, # Changed to 0 (Lite) for performance/memory
            'min_detection_confidence': 0.65,
            'min_tracking_confidence': 0.65
        }
        
        # 2. Try to load XGBoost Model
        self.xgb_clf = None
        self.feature_names = None
        print(f"Loading XGBoost model from {model_path}...")
        try:
            if xgb is None:
                raise ImportError("xgboost library is not installed or failed to import.")
            
            self.xgb_clf = xgb.XGBClassifier()
            self.xgb_clf.load_model(model_path)
            self.feature_names = self.xgb_clf.get_booster().feature_names
            print(f"Model loaded successfully with {len(self.feature_names)} features.")
        except Exception as e:
            print(f"XGBoost model not loaded: {e}. Using heuristic detection only.")
            self.xgb_clf = None

        # 3. Buffers
        self.fps = 30 # CRITICAL: Must match capture FPS for velocity calculation
        # ACCURACY CONFIG: 1.5 second window (45 frames at 30fps)
        # Reducing window size to ensure we capture the fall event 'tightly'
        self.window_size_sec = 1.5
        self.window_frames = int(self.window_size_sec * self.fps)
        
        # Ingestion Hardening: Async Inference Executor
        # Max workers = 1 ensures we don't spawn threads uncontrollably.
        # This keeps the main loop (Ingestion) free.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ml_future = None
        self.latest_ml_prob = 0.0
        
        # Processing buffer (stores raw frames for feature extraction)
        self.processing_buffer = [] 
        
        # Video frame buffer for replay (circular)
        self.replay_buffer_size = 150  # 15 seconds
        self.frame_buffer = deque(maxlen=self.replay_buffer_size)
        
        # State variables
        self.state = "STANDING"
        self.paused = False
        self.frame_count = 0
        self.last_inference_frame = 0
        self.stride_frames = 5 # Run ML every 5 frames (0.5s)
        self.fall_cooldown = 0
        self.consecutive_fall_frames = 0
        
        # Check if modules loaded
        if per_frame is None or fe_eng is None:
            print("CRITICAL WARNING: Feature engineering modules not loaded. ML will not work.")
            self.xgb_clf = None # Disable ML if no FE

    def _safe_heuristic_check(self, frame):
        return True

    def _run_strict_ml_inference(self, buffer_snapshot, width, height):
        """
        Runs feature extraction and prediction on a SNAPSHOT of the buffer.
        buffer_snapshot: List of (landmarks, frame_width, frame_height) tuples.
        """
        if not self.xgb_clf or not per_frame or not fe_eng:
            return 0.0
            
        try:
            # 1. Zero-Copy: Pass landmarks directly to refactored extract_from_landmarks
            # buffer_snapshot is a list of MediaPipe landmark objects.
            
            # Extract landmarks list to pass
            # We assume all frames have same W/H, or we use the latest. 
            # Ideally store W/H or pass it. For now, assuming fixed capture size or taking from last.
            
            landmarks_only = [item for item in buffer_snapshot] 
            
            # 2. Extract Per-Frame features (In-Memory, CPU only, no new MP instance)
            # We hardcode 1920x1080 as fallback if not tracked, or we should track it.
            # Let's assume standard HD for feature normalization (it's normalized anyway).
            
            df_pf = per_frame.extract_from_landmarks(landmarks_only, width=width, height=height, fps=self.fps)
            
            if df_pf is None or df_pf.empty:
                return 0.0
                
            # 3. Extract Advanced Features
            feats = fe_eng.extract_advanced_features(df_pf)
            
            # 4. Align features with Model
            row = pd.DataFrame([feats])
            if hasattr(self.xgb_clf, "feature_names_in_"):
                names = list(self.xgb_clf.feature_names_in_)
                for c in names:
                    if c not in row:
                        row[c] = 0.0
                row = row.reindex(columns=names)
            
            # 5. Predict
            prob = self.xgb_clf.predict_proba(row)[0][1]
            return prob
            
        except Exception as e:
            print(f"Async ML Inference Error: {e}")
            return 0.0

    def process_frame(self, frame):
        with self.lock:
            if self.paused or frame is None or frame.size == 0:
                return None, False, self.state, "paused", 0.0

            self.frame_count += 1
            h, w = frame.shape[:2]
            
            # 1. MediaPipe Draw (Visuals only) AND Capture Landmarks
            annotated_frame = frame.copy()
            current_landmarks = None
            
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # Lazy initialization
                if self.pose is None: # Changed from self.mp_pose to self.pose for consistency with original code
                    
                    # INFRA FIX: Load model into buffer to avoid OneDrive/Pathing errors
                    try:
                        # Assuming self.mp_model_path is defined in __init__ for the MediaPipe model
                        # For now, using a placeholder or assuming it's passed.
                        # If not defined, this will cause an AttributeError.
                        # For this edit, we'll assume a default or that it's set elsewhere.
                        # Let's use a common model name as a placeholder if not explicitly set.
                        mp_model_path = getattr(self, 'mp_model_path', 'pose_landmarker_heavy.task')
                        if not os.path.exists(mp_model_path):
                            print(f"MediaPipe model not found at {mp_model_path}. Please ensure it's available.")
                            # Fallback to original mp.solutions.pose if model not found
                            self.pose = mp.solutions.pose.Pose(**self.pose_options)
                            print("Falling back to mp.solutions.pose due to missing model file.")
                        else:
                            with open(mp_model_path, "rb") as f:
                                model_data = f.read()
                            
                            base_options = python.BaseOptions(model_asset_buffer=model_data)
                            options = vision.PoseLandmarkerOptions(
                                base_options=base_options,
                                running_mode=vision.RunningMode.VIDEO,
                                num_poses=1,
                                min_pose_detection_confidence=0.5,
                                min_pose_presence_confidence=0.5,
                                min_tracking_confidence=0.5,
                                output_segmentation_masks=False
                            )
                            self.pose = vision.PoseLandmarker.create_from_options(options) # Assign to self.pose
                            print("MediaPipe Pose initialized (Buffered Mode)")
                    except Exception as e:
                        print(f"Error initializing MediaPipe (Buffered): {e}")
                        # Fallback to original mp.solutions.pose if buffered init fails
                        self.pose = mp.solutions.pose.Pose(**self.pose_options)
                        print("Falling back to mp.solutions.pose due to initialization error.")
                        
                # Process the frame using the initialized pose object
                if isinstance(self.pose, mp.solutions.pose.Pose):
                    # Original MediaPipe solutions API
                    results = self.pose.process(rgb)
                    if results.pose_landmarks:
                        current_landmarks = results.pose_landmarks
                        self.mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS
                        )
                elif isinstance(self.pose, vision.PoseLandmarker):
                    # New MediaPipe Tasks API
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    detection_result = self.pose.detect_for_video(mp_image, self.frame_count) # Use frame_count as timestamp_ms
                    if detection_result.pose_landmarks:
                        current_landmarks = detection_result.pose_landmarks[0] # Assuming single pose
                        # Convert normalized landmarks to pixel coordinates for drawing
                        h, w, _ = frame.shape
                        for landmark in current_landmarks:
                            landmark.x *= w
                            landmark.y *= h
                        self.mp_drawing.draw_landmarks(
                            annotated_frame,
                            current_landmarks,
                            self.mp_pose.POSE_CONNECTIONS, # mp.solutions.pose.POSE_CONNECTIONS still valid
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
            except Exception as e:
                 print(f"MP Error: {e}")

            # 2. Buffer Management - STORE LANDMARKS ONLY
            # We store landmarks to avoid re-running MP in ML thread
            self.processing_buffer.append(current_landmarks)
            
            # Keep small buffer (Window + Margin)
            if len(self.processing_buffer) > self.window_frames + 10: 
                 overflow = len(self.processing_buffer) - self.window_frames
                 self.processing_buffer = self.processing_buffer[overflow:]

            # 3. ML Inference Logic (SYNC MODE for Stability)
            # Now much faster because NO second inference.
            if len(self.processing_buffer) >= self.window_frames and \
               (self.frame_count - self.last_inference_frame) >= self.stride_frames:
               
                 self.last_inference_frame = self.frame_count
                 snapshot = list(self.processing_buffer)[-self.window_frames:]
                 
                 # Run synchronously (now lightweight)
                 self.latest_ml_prob = self._run_strict_ml_inference(snapshot, w, h)
                 if self.latest_ml_prob > 0.5:
                      print(f"Strict ML Prob: {self.latest_ml_prob:.2f}")

            # 6. Hybrid System: Geometric Voting Buffer
            # We calculate a "Fall Vote" for THIS frame based on Physics/Geometry.
            # Then we use a temporal buffer to confirm the fall.
            
            fall_detected = False
            vote_score = 0.0 # CRITICAL FIX: Initialize variable
            
            if current_landmarks:
                lm = current_landmarks.landmark
                
                # --- A. Metrics Calculation ---
                # 1. Height Ratio
                shoulder_y = (lm[11].y + lm[12].y) / 2
                ankle_y = (lm[27].y + lm[28].y) / 2
                current_height = abs(ankle_y - shoulder_y)
                
                # Auto-Calibrate
                if not hasattr(self, 'max_standing_height'): self.max_standing_height = 0.1
                if self.state == "STANDING": 
                    self.max_standing_height = max(self.max_standing_height, current_height)
                height_ratio = current_height / (self.max_standing_height + 1e-6)
                
                # 2. Joint Angles
                def get_angle(a, b, c):
                    a = np.array([a.x, a.y])
                    b = np.array([b.x, b.y])
                    c = np.array([c.x, c.y])
                    ba = a - b; bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                avg_hip_angle = (get_angle(lm[11], lm[23], lm[25]) + get_angle(lm[12], lm[24], lm[26])) / 2
                avg_knee_angle = (get_angle(lm[23], lm[25], lm[27]) + get_angle(lm[24], lm[26], lm[28])) / 2
                
                # 3. Velocity & AR
                dy = self._calculate_vertical_velocity()
                xs = [l.x for l in lm]; ys = [l.y for l in lm]
                w_bb = max(xs) - min(xs); h_bb = max(ys) - min(ys)
                current_ar = w_bb / (h_bb + 1e-6)

                # --- Advanced Verification (Grade A) ---
                
                # 1. Edge Guard
                # Ignore detections if key joints are too close to the edge (partial body)
                def _is_edge_safe(landmarks):
                    margin = 0.05
                    for i in [11, 12, 23, 24, 25, 26]: # Shoulders, Hips, Knees
                        x, y = landmarks[i].x, landmarks[i].y
                        if x < margin or x > (1 - margin) or y < margin or y > (1 - margin):
                            return False
                    return True
                
                edge_safe = _is_edge_safe(lm)

                # 2. Zone Calibration (Perspective Fix)
                # People further away (higher Y) look smaller. 
                avg_foot_y = (lm[27].y + lm[28].y) / 2
                perspective_scale = 1.0
                if avg_foot_y < 0.5: # User is in the "back" of the room
                     perspective_scale = 0.85 # Expect them to be 15% smaller
                
                effective_height_ratio = height_ratio / perspective_scale

                # --- B. Frame-Level Classification (The Vote) ---
                
                # Guard: Detect Sitting explicitly first
                # Enhanced Sitting Logic: Includes "Legs Extended" sitting
                # If Hips are bent (70-140) OR (Height is Low AND Knees Straight)
                is_standard_sitting = (70 < avg_hip_angle < 140) and (70 < avg_knee_angle < 140)
                is_legs_extended_sitting = (effective_height_ratio < 0.65) and (avg_knee_angle > 140) and (avg_hip_angle < 120)
                
                is_sitting_guard = is_standard_sitting or is_legs_extended_sitting
                
                is_fall_frame = 0
                
                # Skip voting if unsafe edge
                if edge_safe:
                    # 1. Dynamic Fall
                    threshold_dy = 0.1 if is_sitting_guard else 0.05
                    if dy > threshold_dy:
                        is_fall_frame = 1
                        
                    # 2. Impact/Post-Fall
                    if not is_sitting_guard:
                        # Use EFFECTIVE height ratio (Perspective Corrected)
                        if current_ar > 1.1 and effective_height_ratio < 0.8:
                            is_fall_frame = 1
                    
                    # 3. XGBoost Model Vote
                    ml_thresh = 0.95 if is_sitting_guard else 0.85
                    if self.latest_ml_prob > ml_thresh:
                         is_fall_frame = 1
                         print(f"ML Override: Prob={self.latest_ml_prob:.2f}")

                # Update Voting Buffer
                self.prediction_history = getattr(self, 'prediction_history', deque(maxlen=20))
                self.prediction_history.append(is_fall_frame)
                
                # --- C. Temporal Decision ---
                vote_score = sum(self.prediction_history) / len(self.prediction_history) if self.prediction_history else 0
                
                # --- D. State Machine & Alarm Gate ---
                
                # 0. Stillness Tracker
                if current_ar > 1.0 and abs(dy) < 0.02:
                     self.stillness_counter = getattr(self, 'stillness_counter', 0) + 1
                else:
                     self.stillness_counter = 0

                # 1. Pre-Alarm (FALLEN -> VERIFYING)
                if vote_score > 0.6:
                    self.state = "FALLEN"
                    self.fall_cooldown = 45
                    
                    # 2. ALARM GATE: Only trigger confirmed ALARM if stillness is observed
                    # OR if the Vote is overwhelming (> 0.9) implying a violent crash
                    if self.stillness_counter > 30 or vote_score > 0.9:
                        fall_detected = True 
                        self.state = "ALARM"
                
                # 3. Non-Alarm States
                elif self.fall_cooldown == 0:
                    # 1. FALLING
                    if dy > 0.05: self.state = "FALLING"
                    # 2. LYING
                    elif self.stillness_counter > 30: self.state = "LYING"
                    
                    # 3. SITTING (Priority over Standing for ambiguous "legs extended" cases)
                    elif is_sitting_guard: self.state = "SITTING"
                        
                    # 4. STANDING
                    elif avg_knee_angle > 150 and effective_height_ratio > 0.75: self.state = "STANDING"
                    
                    # 5. BENDING
                    elif avg_hip_angle < 110 and avg_knee_angle > 120: self.state = "BENDING"
                    
                    # Fallback
                    elif effective_height_ratio > 0.8: self.state = "STANDING"
                
                if self.fall_cooldown > 0:
                     self.fall_cooldown -= 1
                     if self.state == "ALARM": fall_detected = True 

                # DEBUG
                if self.frame_count % 15 == 0 or is_fall_frame:
                     print(f"DBG: St={self.state} Vote={vote_score:.2f} | dy={dy:.3f} AR={current_ar:.2f}")
                     
            # Replay Buffer
            try:
                rz = cv2.resize(annotated_frame, (320, 240))
                self.frame_buffer.append(rz)
            except: pass
            
            # Trigger Replay Save if Fall Detected (Confirmed ALARM only)
            if fall_detected:
                 import threading
                 import time
                 threading.Thread(target=self._save_replay, daemon=True).start()
            
            # Info overlay
            ml_prob = self.latest_ml_prob
            mode = "Hybrid Geom-Vote"
            cv2.putText(annotated_frame, f"Mode: {mode} | State: {self.state}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.ml_future is not None:
                 cv2.putText(annotated_frame, "Processing...", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            color = (0,0,255) if vote_score > 0.6 else (255,255,0)
            cv2.putText(annotated_frame, f"Vote Score: {vote_score:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detection_method = mode
            return annotated_frame, fall_detected, self.state, detection_method, vote_score

    def _save_replay(self):
        """Save the current buffer to a video file."""
        try:
            import time
            import os
            
            if not self.frame_buffer: return
            
            timestamp = int(time.time())
            filename = f"fall_replay_{timestamp}.mp4"
            save_dir = "./results/replays"
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            
            # Convert deque to list
            frames = list(self.frame_buffer)
            
            if not frames: return
            
            # Settings
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
            out = cv2.VideoWriter(filepath, fourcc, 15.0, (w, h))
            
            for f in frames:
                out.write(f)
            
            out.release()
            print(f"Replay saved: {filepath}")
        except Exception as e:
            print(f"Error saving replay: {e}")

    def _calculate_vertical_velocity(self):
        """Simple heuristic to get vertical velocity of Hip center over last 5 frames"""
        if len(self.processing_buffer) < 5: return 0.0
        
        curr = self.processing_buffer[-1]
        prev = self.processing_buffer[-5]
        
        if not curr or not prev: return 0.0
        
        # Hip Center Y
        curr_y = (curr.landmark[23].y + curr.landmark[24].y) / 2
        prev_y = (prev.landmark[23].y + prev.landmark[24].y) / 2
        
        # Velocity = displacement / time
        # Time for 5 frames at 30fps is 5/30 = 0.16s
        # Only care about DOWNWARD velocity (positive Y increase)
        dy = curr_y - prev_y
        return dy # We just return raw displacement proxy for speed


    def get_replay_clip(self):
        with self.lock:
            return list(self.frame_buffer)[::2]

    def save_replay(self, output_dir='./checkpoints'):
        """Save the current replay buffer as a video file (Video Only)"""
        with self.lock:
            if len(self.frame_buffer) == 0:
                return None
            frames = list(self.frame_buffer)
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'fall_detection_{timestamp}.mp4')
            os.makedirs(output_dir, exist_ok=True)
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"[REPLAY SAVED] {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving replay: {e}")
            return None
            
    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused

    def reset(self):
        with self.lock:
            self.processing_buffer = []
            self.frame_buffer.clear()
            self.state = "STANDING"
            self.fall_cooldown = 0
            
    def close(self):
        """Explicitly release resources"""
        with self.lock:
            if self.pose:
                self.pose.close()
                self.pose = None
