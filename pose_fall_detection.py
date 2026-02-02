import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import importlib.util
import tempfile
import os
import math
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
import winsound # Added for Audio Alerts

# Load external feature engineering modules dynamically
def load_module(mod_name, file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Module {mod_name} not found at {file_path}")
        return None
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

per_frame = load_module("per_frame_best", "per-frame-best.py")
fe_eng = load_module("final_feature_eng_best", "final-feature-eng-best.py")

# --- SKILL: Biomechanical State Latcher ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class AdvancedPoseFallDetector:
    def __init__(self, model_path='xgb_final_model.json'):
        self.lock = threading.Lock()
        
        # 1. MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = None 
        self.pose_options = {
            'static_image_mode': False,
            'model_complexity': 1,
            'min_detection_confidence': 0.65,
            'min_tracking_confidence': 0.65
        }
        
        # 2. XGBoost
        self.xgb_clf = None
        self.feature_names = None
        print(f"Loading XGBoost model from {model_path}...")
        try:
            if xgb:
                self.xgb_clf = xgb.XGBClassifier()
                self.xgb_clf.load_model(model_path)
                self.feature_names = self.xgb_clf.get_booster().feature_names
                print(f"Model loaded successfully.")
            else:
                print("XGBoost not available.")
        except Exception as e:
            print(f"Model load failed: {e}")
            self.xgb_clf = None

        # 3. Buffers
        self.fps = 30 
        self.window_size_sec = 1.5
        self.window_frames = int(self.window_size_sec * self.fps)
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ml_future = None
        self.latest_ml_prob = 0.0
        
        self.processing_buffer = [] 
        self.replay_buffer_size = 150 
        self.frame_buffer = deque(maxlen=self.replay_buffer_size)
        
        self.state = "STANDING"
        self.paused = False
        self.frame_count = 0
        self.last_inference_frame = 0
        self.stride_frames = 5
        self.fall_cooldown = 0
        self.fallen_frame_counter = 0

        # SKILL: State Latching Counters
        self.stable_standing_counter = 0
        
        # SKILL: Filters (Dict of OneEuroFilter)
        # Keys: 'x_0', 'y_0', ... 'x_32', 'y_32'
        self.filters = {}
        
        # AUDIO ALERT SETUP
        self.alert_sound_path = os.path.abspath("fall_alert.wav")
        self.last_alert_time = 0

        if per_frame is None or fe_eng is None:
            self.xgb_clf = None

    def _get_filtered_landmark(self, index, raw_val, t, axis):
        key = f"{axis}_{index}"
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(t, raw_val, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
            return raw_val
        return self.filters[key](t, raw_val)

    def _run_strict_ml_inference(self, buffer_snapshot, width, height):
        if not self.xgb_clf or not per_frame or not fe_eng: return 0.0
        try:
            landmarks_only = [item for item in buffer_snapshot] 
            df_pf = per_frame.extract_from_landmarks(landmarks_only, width=width, height=height, fps=self.fps)
            if df_pf is None or df_pf.empty: return 0.0
            feats = fe_eng.extract_advanced_features(df_pf)
            row = pd.DataFrame([feats])
            if hasattr(self.xgb_clf, "feature_names_in_"):
                names = list(self.xgb_clf.feature_names_in_)
                for c in names:
                    if c not in row: row[c] = 0.0
                row = row.reindex(columns=names)
            return self.xgb_clf.predict_proba(row)[0][1]
        except Exception:
            return 0.0
            
    def _play_alert(self):
        """Plays the ElevenLabs alert sound asynchronously."""
        try:
            if os.path.exists(self.alert_sound_path):
                # SND_ASYNC = 1, SND_FILENAME = 131072
                # We use bitwise OR to combine flags
                winsound.PlaySound(self.alert_sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                print(f"Alert file not found: {self.alert_sound_path}")
        except Exception as e:
            print(f"Failed to play alert: {e}")

    def process_frame(self, frame):
        with self.lock:
            if self.paused or frame is None or frame.size == 0:
                return None, False, self.state, "paused", 0.0

            self.frame_count += 1
            t_curr = time.time()
            h, w = frame.shape[:2]
            annotated_frame = frame.copy()
            current_landmarks = None
            vote_score = 0.0
            
            # --- 1. MediaPipe & Filtering ---
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # Lazy Init
                if self.pose is None:
                   try:
                       self.pose = mp.solutions.pose.Pose(**self.pose_options)
                       # Simplified init for robustness
                   except: 
                       self.pose = mp.solutions.pose.Pose(**self.pose_options)
                       
                if isinstance(self.pose, mp.solutions.pose.Pose):
                    results = self.pose.process(rgb)
                    if results.pose_landmarks:
                        current_landmarks = results.pose_landmarks
                        
                        # SKILL: Apply One-Euro Filter in-place
                        for idx, lm in enumerate(current_landmarks.landmark):
                            lm.x = self._get_filtered_landmark(idx, lm.x, t_curr, 'x')
                            lm.y = self._get_filtered_landmark(idx, lm.y, t_curr, 'y')
                            # Z and Vis usually don't need heavy filtering or are less critical for 2D logic
                        
                        self.mp_drawing.draw_landmarks(annotated_frame, current_landmarks, self.mp_pose.POSE_CONNECTIONS)
            except Exception as e:
                 print(f"MP Error: {e}")

            # 2. Buffer Logic
            self.processing_buffer.append(current_landmarks)
            if len(self.processing_buffer) > self.window_frames + 10: 
                 overflow = len(self.processing_buffer) - self.window_frames
                 self.processing_buffer = self.processing_buffer[overflow:]

            # 3. ML Inference
            if len(self.processing_buffer) >= self.window_frames and \
               (self.frame_count - self.last_inference_frame) >= self.stride_frames:
                 self.last_inference_frame = self.frame_count
                 snapshot = list(self.processing_buffer)[-self.window_frames:]
                 self.latest_ml_prob = self._run_strict_ml_inference(snapshot, w, h)

            # 6. Hybrid System
            fall_detected = False
            final_score = 0.0 
            
            if current_landmarks:
                lm = current_landmarks.landmark
                self.prediction_history = getattr(self, 'prediction_history', deque(maxlen=30))

                # Metrics
                shoulder_y = (lm[11].y + lm[12].y) / 2
                ankle_y = (lm[27].y + lm[28].y) / 2
                current_height = abs(ankle_y - shoulder_y)
                
                if not hasattr(self, 'max_standing_height'): self.max_standing_height = 0.1
                perspective_scale = 0.6 + (0.5 * ankle_y) 
                
                if self.state == "STANDING": 
                     raw_height_norm = current_height / perspective_scale
                     self.max_standing_height = max(self.max_standing_height, raw_height_norm)

                expected_height_at_y = self.max_standing_height * perspective_scale
                height_ratio = current_height / (expected_height_at_y + 1e-6)
                
                dy = self._calculate_vertical_velocity()
                xs = [l.x for l in lm]; ys = [l.y for l in lm]
                w_bb = max(xs) - min(xs); h_bb = max(ys) - min(ys)
                current_ar = w_bb / (h_bb + 1e-6)

                # Angles
                def get_angle(a, b, c):
                    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
                    ba = a - b; bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                avg_hip_angle = (get_angle(lm[11], lm[23], lm[25]) + get_angle(lm[12], lm[24], lm[26])) / 2
                avg_knee_angle = (get_angle(lm[23], lm[25], lm[27]) + get_angle(lm[24], lm[26], lm[28])) / 2
                
                # Heuristic Vote
                is_fall_frame = 0
                if current_ar > 1.1 and height_ratio < 0.8: is_fall_frame = 1
                if dy > 0.05: is_fall_frame = 1
                if current_ar < 0.8: is_fall_frame = 0 
                
                frame_weight = 1.0
                self.prediction_history.append((is_fall_frame, frame_weight))
                
                total_weighted_votes = sum(vote * w for vote, w in self.prediction_history)
                total_possible_weights = sum(w for _, w in self.prediction_history) if self.prediction_history else 1
                vote_score = total_weighted_votes / (total_possible_weights + 1e-6)
                
                is_fall_vote = (vote_score > 0.7)
                is_lying = (current_ar > 1.4) 
                
                # --- Final Decision Logic ---
                final_score = self.latest_ml_prob if self.latest_ml_prob > 0 else vote_score
                if self.xgb_clf and self.latest_ml_prob > 0: final_score = self.latest_ml_prob
                is_confident_fall = (final_score > 0.75) 

                # SKILL: State Hysteresis (Latch)
                # Helper: Can we unlock from Fallen/Lying?
                can_unlock = False
                if height_ratio > 0.75:
                    self.stable_standing_counter += 1
                else:
                    self.stable_standing_counter = 0 # Reset if any low frame
                
                if self.stable_standing_counter > 30: # 1 second stability
                    can_unlock = True

                # State Transitions
                if self.state in ["FALLEN", "LYING"] and not can_unlock:
                    # LATCHED: Can only switch between Fallen/Lying
                    if is_confident_fall: 
                         # Check stillness to confirm FALLEN vs FALLING
                         if dy < 0.05: self.fallen_frame_counter += 1
                         else: self.fallen_frame_counter = max(0, self.fallen_frame_counter - 1)
                         
                         if self.fallen_frame_counter > 30:
                             if self.state != "FALLEN":
                                 # Transitioning to FALLEN just now
                                 self._play_alert() # TRIGGER AUDIO
                             self.state = "FALLEN"
                             fall_detected = True
                             self.fall_cooldown = 45
                         else:
                             self.state = "FALLING"
                    else:
                         self.state = "LYING" # Default if latched and not falling
                
                elif is_confident_fall:
                    # Logic for entering Fall
                    self.state = "FALLING"
                    
                    # SKILL: Stillness Verification
                    if abs(dy) < 0.05:
                         self.fallen_frame_counter += 1
                    else:
                         self.fallen_frame_counter = max(0, self.fallen_frame_counter - 1)
                    
                    if self.fallen_frame_counter > 30:
                         if self.state != "FALLEN":
                             # Transitioning to FALLEN just now
                             self._play_alert() # TRIGGER AUDIO
                         self.state = "FALLEN"
                         fall_detected = True
                         self.fall_cooldown = 45 # Lock alarm for 1.5s
                
                else:
                    self.fallen_frame_counter = 0 
                    
                    if is_fall_vote and not is_confident_fall:
                        self.state = "BENDING" 
                    
                    # Non-Fall State Logic
                    if self.fall_cooldown == 0:
                        # SKILL: Bending Override
                        is_horizontal = (current_ar > 1.2)
                        
                        if dy > 0.05: self.state = "FALLING"
                        elif current_ar > 1.4: self.state = "LYING"
                        elif avg_knee_angle > 150 and height_ratio > 0.75: self.state = "STANDING"
                        elif avg_hip_angle < 120 and avg_knee_angle < 120: self.state = "SITTING"
                        elif avg_hip_angle < 110 and avg_knee_angle > 120: 
                             if is_horizontal: self.state = "LYING" # Override
                             else: self.state = "BENDING"
                        elif height_ratio > 0.8: self.state = "STANDING"

                if self.state in ["FALLING", "FALLEN"] and final_score < 0.4 and self.fall_cooldown == 0:
                    self.state = "BENDING"
                    fall_detected = False
                
                if self.fall_cooldown > 0:
                     self.fall_cooldown -= 1
                     if self.state == "FALLEN": fall_detected = True

                # Audio Alert Re-trigger logic (every 5 seconds if still fallen)
                if self.state == "FALLEN" and (time.time() - self.last_alert_time > 5.0):
                     self._play_alert()
                     self.last_alert_time = time.time()

                if self.frame_count % 15 == 0:
                     print(f"DBG: St={self.state} ML={final_score:.2f} AR={current_ar:.2f} HR={height_ratio:.2f}")

            # Replay Buffer
            try:
                rz = cv2.resize(annotated_frame, (320, 240))
                self.frame_buffer.append(rz)
            except: pass
            
            if fall_detected:
                 threading.Thread(target=self._save_replay, daemon=True).start()
            
            mode = "Hybrid ML-Latcher"
            color = (0,0,255) if final_score > 0.6 else (0, 255, 0)
            cv2.putText(annotated_frame, f"Mode: {mode} | State: {self.state}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {final_score:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return annotated_frame, fall_detected, self.state, mode, final_score

    def _save_replay(self):
        try:
            import time
            import os
            if not self.frame_buffer: return
            timestamp = int(time.time())
            filename = f"fall_replay_{timestamp}.mp4"
            save_dir = "./results/replays"
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            frames = list(self.frame_buffer)
            if not frames: return
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(filepath, fourcc, 15.0, (w, h))
            for f in frames: out.write(f)
            out.release()
        except: pass

    def _calculate_vertical_velocity(self):
        if len(self.processing_buffer) < 5: return 0.0
        curr = self.processing_buffer[-1]
        prev = self.processing_buffer[-5]
        if not curr or not prev: return 0.0
        curr_y = (curr.landmark[23].y + curr.landmark[24].y) / 2
        prev_y = (prev.landmark[23].y + prev.landmark[24].y) / 2
        return curr_y - prev_y

    def get_replay_clip(self):
        with self.lock: return list(self.frame_buffer)[::2]

    def save_replay(self, output_dir='./checkpoints'):
        with self.lock:
            if len(self.frame_buffer) == 0: return None
            frames = list(self.frame_buffer)
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'fall_detection_{timestamp}.mp4')
            os.makedirs(output_dir, exist_ok=True)
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames: out.write(frame)
            out.release()
            return output_path
        except: return None
            
    def toggle_pause(self):
        with self.lock: self.paused = not self.paused

    def reset(self):
        with self.lock:
            self.processing_buffer = []
            self.frame_buffer.clear()
            self.state = "STANDING"
            self.fall_cooldown = 0
            self.last_alert_time = 0
            
    def close(self):
        with self.lock:
            if self.pose:
                self.pose.close()
                self.pose = None
