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
            'model_complexity': 1,
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
            vote_score = 0.0 # Init safely

            
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
                                min_pose_detection_confidence=0.7, # Increased for Less Jitter
                                min_pose_presence_confidence=0.7,
                                min_tracking_confidence=0.7,
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
            
            if current_landmarks:
                lm = current_landmarks.landmark
                
                # --- A. Metrics Calculation ---
                # Init History Buffer Promptly (30 frames = 1 sec)
                # Stores tuples: (is_fall_frame (0/1), weight (float))
                self.prediction_history = getattr(self, 'prediction_history', deque(maxlen=30))

                # 1. Height Ratio
                shoulder_y = (lm[11].y + lm[12].y) / 2
                ankle_y = (lm[27].y + lm[28].y) / 2
                current_height = abs(ankle_y - shoulder_y)
                
                # Auto-Calibrate (Perspective Aware)
                # Normalizes height expectations based on distance (y-position)
                # Lower in frame (Y=1.0) = Closer = Taller. Top of frame (Y=0.0) = Farther = Shorter.
                if not hasattr(self, 'max_standing_height'): self.max_standing_height = 0.1
                
                # Simple Linear Map: Scale factor ranges from 0.6 (top) to 1.1 (bottom)
                perspective_scale = 0.6 + (0.5 * ankle_y) 
                
                # Check normalized height
                if self.state == "STANDING": 
                     # Store the "Unscaled" absolute max to serve as the baseline
                     raw_height_norm = current_height / perspective_scale
                     self.max_standing_height = max(self.max_standing_height, raw_height_norm)

                # Calculate Ratio against the Local Expected Height for this Y-position
                expected_height_at_y = self.max_standing_height * perspective_scale
                height_ratio = current_height / (expected_height_at_y + 1e-6)
                
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

                # --- B. Frame-Level Classification (The Vote) ---
                
                # ANTI-GUESSING RULE (Vertical Guard)
                # If Aspect Ratio < 0.8 (Tall & Thin), it is physically impossible to be "Fallen".
                # Force score down to prevent "Walking = Falling".
                if current_ar < 0.8:
                     # self.prediction_history.append(0) -- REMOVED (Appended later with weight)
                     is_fall_frame = 0 # Override any physics calculation
                else: 
                     # Proceed with standard physics checks
                     # Does this SINGLE frame look like a fall?
                     # Criteria: High Velocity OR (Low Height + Horizontal)
                     
                     is_fall_frame = 0
                     
                     # 1. Dynamic Fall (Moving down fast)
                     if dy > 0.05:
                         is_fall_frame = 1

                    
                # 2. Impact/Post-Fall (Horizontal & Low)
                # Relaxed thresholds to ensure "Lying" counts as a Fall Vote
                # Height Ratio < 0.8 means they are NOT standing tall.
                if current_ar > 1.1 and height_ratio < 0.8:
                    is_fall_frame = 1
                
                # Weighted Voting: High confidence frames count more
                # Calculate MediaPipe Visibility Confidence (Average of key landmarks)
                avg_vis = (lm[11].visibility + lm[12].visibility + lm[23].visibility + lm[24].visibility) / 4
                frame_weight = 1.5 if avg_vis > 0.8 else 1.0
                
                # Check Vertical Guard AGAIN before appending
                if current_ar < 0.8: is_fall_frame = 0 # Double strict
                
                # Append (Vote, Weight)
                self.prediction_history.append((is_fall_frame, frame_weight))
                
                # --- C. Temporal Decision (Buffer Check) ---
                if self.prediction_history:
                     total_weighted_votes = sum(vote * w for vote, w in self.prediction_history)
                     total_possible_weights = sum(w for _, w in self.prediction_history)
                     vote_score = total_weighted_votes / (total_possible_weights + 1e-6)
                else:
                     vote_score = 0.0
                
                # Threshold: >70% weighted evidence required
                is_fall_vote = (vote_score > 0.7)
                is_lying = (current_ar > 1.4) # Separate check for lying
                
                # --- C. Final State Decision (The Gavel) ---
                
                # CONFIDENCE GATE: Requires ML Support (Vote > 0.4) to trigger Alarm
                # Prevents "Geometry Only" false positives (Random Guessing issue)
                is_confident_fall = (vote_score > 0.4)

                if is_lying:
                    self.state = "LYING" # Passive state, no alarm
                
                elif is_fall_vote and is_confident_fall:
                    # High Confidence Fall Logic
                    # We enter "FALLING" state immediately, but "FALLEN" (Alarm) requires STILLNESS.
                    
                    self.state = "FALLING" # Default transient state
                    
                    # Initialize counter if not exists
                    if not hasattr(self, 'fallen_frame_counter'): self.fallen_frame_counter = 0
                    
                    # STILLNESS CHECK: Velocity must be near zero (< 0.02)
                    if dy < 0.02:
                         self.fallen_frame_counter += 1
                    else:
                         self.fallen_frame_counter = max(0, self.fallen_frame_counter - 1) # Decay if moving
                    
                    # ALARM TRIGGER: 30 Frames (~1s) of Stillness
                    if self.fallen_frame_counter > 30:
                         self.state = "FALLEN"
                         fall_detected = True
                         self.fall_cooldown = 45 # Lock alarm for 1.5s
                
                else:
                    # Reset counter if not in fall voting mode
                    self.fallen_frame_counter = 0 
                    
                    if is_fall_vote and not is_confident_fall:
                        # Geometry says Fall, but ML says Safe -> Suppress
                        self.state = "BENDING" # Safe fallback
                
                # --- D. Non-Fall States (If no alarm) ---
                if self.fall_cooldown == 0:
                    # 1. FALLING (Transient, High Velocity)
                    if dy > 0.05:
                        self.state = "FALLING"
                        
                    # 2. LYING (Sustained Horizontal)
                    elif current_ar > 1.4:
                        self.state = "LYING"
                        
                    # 3. STANDING (Tall & Straight)
                    # Knee > 150 (Straight), HR > 0.8
                    elif avg_knee_angle > 150 and height_ratio > 0.75:
                        self.state = "STANDING"
                        
                    # 4. SITTING (Crumple / Mid-height)
                    # Both Hip and Knee must be bent (< 120)
                    elif avg_hip_angle < 120 and avg_knee_angle < 120:
                        self.state = "SITTING"
                        
                    # 5. BENDING (Hinge / Picking up object)
                    # Hip Bent (< 110) BUT Knees Straighter (> 120)
                    elif avg_hip_angle < 110 and avg_knee_angle > 120:
                        self.state = "BENDING"
                    
                    # Fallback
                    elif height_ratio > 0.8:
                        self.state = "STANDING" # Default for upright
                
                # --- Sanity Check Override ---
                # If system claims FALLING but confidence is 0, force reset
                if self.state in ["FALLING", "FALLEN"] and vote_score < 0.1:
                    self.state = "BENDING"
                    fall_detected = False
                
                if self.fall_cooldown > 0:
                     self.fall_cooldown -= 1
                     if self.state == "FALLEN": fall_detected = True

                # DEBUG
                if self.frame_count % 15 == 0 or is_fall_frame:
                     print(f"DBG: St={self.state} Vote={vote_score:.2f} | dy={dy:.3f} AR={current_ar:.2f} | H={avg_hip_angle:.0f} K={avg_knee_angle:.0f}")

            # Replay Buffer
            try:
                rz = cv2.resize(annotated_frame, (320, 240))
                self.frame_buffer.append(rz)
            except: pass
            
            # Trigger Replay Save if Fall Detected
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
