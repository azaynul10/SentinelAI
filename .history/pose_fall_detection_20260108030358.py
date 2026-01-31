import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: xgboost could not be imported. ML confirmation will be disabled.")
import time
import os
from datetime import datetime
from collections import deque

class AdvancedPoseFallDetector:
    """
    Hybrid Fall Detector combining:
    1. Real-time heuristic detection (immediate response)
    2. XGBoost ML model (when available, for confirmation)
    """
    
    def __init__(self, model_path='xgb_final_model.json'):
        # 1. MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.4, # Lowered for better low-quality detection
            min_tracking_confidence=0.4 # Lowered for better low-quality detection
        )
        
        # 2. Try to load XGBoost Model (optional - fallback to heuristics if fails)
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
        self.fps = 30
        self.window_size = 45  # 1.5 seconds window
        self.buffer_size = 450  # 15 seconds for replay
        
        # Metrics buffer for feature calculation
        self.metrics_buffer = deque(maxlen=self.window_size)
        
        # Video frame buffer for replay
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        # State variables
        self.state = "STANDING"
        self.paused = False
        self.frame_count = 0
        self.prev_landmarks = None
        self.prev_com = None
        self.prev_com_speed = 0
        self.prev_torso_angle = 0
        self.fall_cooldown = 0
        
        # Heuristic thresholds (tuned for webcam detection - aggressive mode)
        self.VELOCITY_THRESHOLD = 80  # pixels/sec (lowered from 150 for better sensitivity)
        self.ANGLE_THRESHOLD = 45  # degrees from vertical
        self.ASPECT_RATIO_THRESHOLD = 1.0  # width/height (Increased from 0.9 to reduce false positives)
        self.FALL_CONFIRM_FRAMES = 5  # (Increased from 3 to 5 to avoid transient false positives)
        self.consecutive_fall_frames = 0

    def _extract_metrics(self, landmarks, shape):
        """Extract per-frame metrics for both heuristic and ML detection"""
        h, w, _ = shape
        
        def get_xy(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        # Key landmark indices
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        NOSE = 0

        # 1. Center of Mass (hip midpoint)
        hip_l, hip_r = get_xy(L_HIP), get_xy(R_HIP)
        com = (hip_l + hip_r) / 2.0
        
        # 2. Velocity
        com_speed = 0.0
        com_acc = 0.0
        vertical_speed = 0.0  # Positive = Downwards
        
        if self.prev_com is not None:
            displacement_vec = com - self.prev_com
            displacement = np.linalg.norm(displacement_vec)
            
            # Calculate vertical speed (y is index 1)
            vertical_speed = displacement_vec[1] * self.fps
            
            com_speed = displacement * self.fps
            com_acc = (com_speed - self.prev_com_speed) * self.fps
        
        # 3. Torso angle (from vertical)
        shoulder_mid = (get_xy(L_SHOULDER) + get_xy(R_SHOULDER)) / 2.0
        torso_vec = shoulder_mid - com
        vertical = np.array([0, -1])
        norm_torso = np.linalg.norm(torso_vec)
        
        if norm_torso > 1e-6:
            unit_torso = torso_vec / norm_torso
            dot = np.dot(unit_torso, vertical)
            torso_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        else:
            torso_angle = 0.0
        
        # 4. Angular velocity
        angular_velocity = abs(torso_angle - self.prev_torso_angle) * self.fps if self.prev_torso_angle else 0
        
        # 5. Aspect ratio (bounding box)
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        box_w = (max(xs) - min(xs)) * w
        box_h = (max(ys) - min(ys)) * h
        aspect_ratio = box_w / (box_h + 1e-6)
        
        # 6. Head-hip vertical distance (normalized)
        nose = get_xy(NOSE)
        head_hip_dist = abs(nose[1] - com[1]) / h
        
        # 7. Knee angle (average)
        try:
            knee_l, knee_r = get_xy(L_KNEE), get_xy(R_KNEE)
            ankle_l, ankle_r = get_xy(27), get_xy(28)
            
            def calc_angle(a, b, c):
                ba = a - b
                bc = c - b
                cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            knee_angle = (calc_angle(hip_l, knee_l, ankle_l) + calc_angle(hip_r, knee_r, ankle_r)) / 2
        except:
            knee_angle = 180.0
        
        # Update previous values
        self.prev_com = com
        self.prev_com_speed = com_speed
        self.prev_torso_angle = torso_angle
        
        return {
            'com_speed': com_speed,
            'com_acc': com_acc,
            'vertical_speed': vertical_speed,
            'torso_angle_deg': torso_angle,
            'angular_velocity_deg_s': angular_velocity,
            'aspect_ratio': aspect_ratio,
            'head_hip_dist': head_hip_dist,
            'knee_angle': knee_angle,
            'log_com_speed': np.log1p(com_speed),
            'log_upper_body_angle': np.log1p(torso_angle),
            'com': com
        }

    def _heuristic_fall_detection(self, metrics):
        """Simple rule-based fall detection for immediate response"""
        com_speed = metrics['com_speed']
        vertical_speed = metrics['vertical_speed']
        torso_angle = metrics['torso_angle_deg']
        aspect_ratio = metrics['aspect_ratio']
        angular_velocity = metrics['angular_velocity_deg_s']
        
        # 1. UPWARD MOTION REJECTION (Fix for "standing up" false positives)
        # If moving UP significantly (vertical_speed negative), it's not a fall
        if vertical_speed < -40:  # Moving up faster than 40px/sec
             self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 2)
             return False

        # Fall indicators
        rapid_motion = com_speed > self.VELOCITY_THRESHOLD
        tilted = torso_angle > self.ANGLE_THRESHOLD
        horizontal_body = aspect_ratio > self.ASPECT_RATIO_THRESHOLD
        rapid_angle_change = angular_velocity > 100  # degrees/sec
        
        # Check for fall pattern
        # Added requirement that rapid motion must not be purely horizontal/upward
        is_falling_motion = rapid_motion and (vertical_speed > 20) 
        
        if (is_falling_motion and tilted) or (rapid_angle_change and tilted and vertical_speed > 0) or horizontal_body:
            self.consecutive_fall_frames += 1
        else:
            self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
        
        # Confirm fall after sustained detection
        if self.consecutive_fall_frames >= self.FALL_CONFIRM_FRAMES:
            return True
        
        return False

    def _compute_ml_features(self):
        """Compute features for XGBoost model"""
        if len(self.metrics_buffer) < self.window_size or self.xgb_clf is None:
            return None
        
        df = pd.DataFrame(list(self.metrics_buffer))
        feats = {}
        
        try:
            # Peak features
            feats['peak_com_speed'] = df['com_speed'].max()
            feats['t_peak_com_speed'] = df['com_speed'].idxmax() / len(df) if len(df) > 0 else 0
            feats['peak_com_acc'] = df['com_acc'].max()
            feats['t_peak_com_acc'] = df['com_acc'].idxmax() / len(df) if len(df) > 0 else 0
            
            # Compute jerk
            df['com_jerk'] = df['com_acc'].diff().fillna(0) * self.fps
            feats['peak_com_jerk'] = df['com_jerk'].max()
            feats['t_peak_com_jerk'] = df['com_jerk'].idxmax() / len(df) if len(df) > 0 else 0
            
            feats['peak_angular_velocity_deg_s'] = df['angular_velocity_deg_s'].max()
            feats['t_peak_angular_velocity_deg_s'] = df['angular_velocity_deg_s'].idxmax() / len(df) if len(df) > 0 else 0
            feats['peak_log_upper_body_angle'] = df['log_upper_body_angle'].max()
            feats['t_peak_log_upper_body_angle'] = df['log_upper_body_angle'].idxmax() / len(df) if len(df) > 0 else 0
            feats['peak_log_com_speed'] = df['log_com_speed'].max()
            feats['t_peak_log_com_speed'] = df['log_com_speed'].idxmax() / len(df) if len(df) > 0 else 0
            
            # Delta features (first half vs second half)
            mid = len(df) // 2
            first_half = df.iloc[:mid]
            second_half = df.iloc[mid:]
            
            feats['delta_mean_torso_angle_deg'] = second_half['torso_angle_deg'].mean() - first_half['torso_angle_deg'].mean()
            feats['delta_mean_angular_velocity_deg_s'] = second_half['angular_velocity_deg_s'].mean() - first_half['angular_velocity_deg_s'].mean()
            feats['delta_mean_angular_velocity'] = feats['delta_mean_angular_velocity_deg_s']
            feats['delta_mean_aspect_ratio'] = second_half['aspect_ratio'].mean() - first_half['aspect_ratio'].mean()
            feats['delta_mean_knee_angle'] = second_half['knee_angle'].mean() - first_half['knee_angle'].mean()
            feats['delta_mean_com_speed'] = second_half['com_speed'].mean() - first_half['com_speed'].mean()
            feats['delta_mean_com_acc'] = second_half['com_acc'].mean() - first_half['com_acc'].mean()
            feats['delta_mean_com_jerk'] = second_half['com_jerk'].mean() - first_half['com_jerk'].mean()
            feats['delta_mean_log_upper_body_angle'] = second_half['log_upper_body_angle'].mean() - first_half['log_upper_body_angle'].mean()
            feats['delta_mean_log_com_speed'] = second_half['log_com_speed'].mean() - first_half['log_com_speed'].mean()
            
            # Segment features (split into 3 segments)
            seg_size = len(df) // 3
            for i, seg_name in enumerate(['seg1', 'seg2', 'seg3']):
                seg = df.iloc[i*seg_size:(i+1)*seg_size]
                if len(seg) > 0:
                    feats[f'd1_com_speed_{seg_name}'] = seg['com_speed'].diff().mean()
                    feats[f'mean_com_speed_{seg_name}'] = seg['com_speed'].mean()
                    feats[f'std_com_speed_{seg_name}'] = seg['com_speed'].std()
                    feats[f'range_com_speed_{seg_name}'] = seg['com_speed'].max() - seg['com_speed'].min()
                    
                    feats[f'd1_com_acc_{seg_name}'] = seg['com_acc'].diff().mean()
                    feats[f'mean_com_acc_{seg_name}'] = seg['com_acc'].mean()
                    feats[f'std_com_acc_{seg_name}'] = seg['com_acc'].std()
                    feats[f'range_com_acc_{seg_name}'] = seg['com_acc'].max() - seg['com_acc'].min()
                    
                    feats[f'd1_angular_velocity_deg_s_{seg_name}'] = seg['angular_velocity_deg_s'].diff().mean()
                    feats[f'mean_angular_velocity_deg_s_{seg_name}'] = seg['angular_velocity_deg_s'].mean()
                    feats[f'std_angular_velocity_deg_s_{seg_name}'] = seg['angular_velocity_deg_s'].std()
                    feats[f'range_angular_velocity_deg_s_{seg_name}'] = seg['angular_velocity_deg_s'].max() - seg['angular_velocity_deg_s'].min()
                    
                    # Slopes
                    x = np.arange(len(seg)).reshape(-1, 1)
                    for col in ['torso_angle_deg', 'com_speed', 'log_upper_body_angle', 'log_com_speed']:
                        if len(seg) > 1:
                            slope = np.polyfit(x.flatten(), seg[col].values, 1)[0]
                            feats[f'slope_{col}_{seg_name}'] = slope
                        else:
                            feats[f'slope_{col}_{seg_name}'] = 0
            
            # Additional features (simplified)
            feats['mean_interval_acc'] = df['com_acc'].mean()
            feats['std_interval_acc'] = df['com_acc'].std()
            
            # Change point features (simplified)
            angle_diff = df['torso_angle_deg'].diff().abs()
            feats['cp_magnitude_ang'] = angle_diff.max()
            feats['cp_time_ang'] = angle_diff.idxmax() / len(df) if len(df) > 0 else 0
            
            # Wavelet entropy (simplified - using variance as proxy)
            for level, d_name in enumerate(['d1', 'd2', 'd3'], 1):
                feats[f'wavelet_entropy_com_speed_{d_name}'] = df['com_speed'].rolling(2**level).var().mean()
                feats[f'wavelet_entropy_log_com_speed_{d_name}'] = df['log_com_speed'].rolling(2**level).var().mean()
            
            # Sample entropy (simplified)
            feats['sampen_com_speed'] = df['com_speed'].std() / (df['com_speed'].mean() + 1e-6)
            feats['sampen_log_com_speed'] = df['log_com_speed'].std() / (df['log_com_speed'].mean() + 1e-6)
            
            # Head-hip and correlation
            feats['slope_head_hip'] = np.polyfit(np.arange(len(df)), df['head_hip_dist'].values, 1)[0] if len(df) > 1 else 0
            feats['corr_torso_hhg'] = df['torso_angle_deg'].corr(df['head_hip_dist'])
            
            # Build feature vector in correct order
            model_input = []
            for col in self.feature_names:
                val = feats.get(col, 0.0)
                if pd.isna(val):
                    val = 0.0
                model_input.append(val)
            
            return np.array([model_input])
            
        except Exception as e:
            print(f"Feature computation error: {e}")
            return None

    def process_frame(self, frame):
        if self.paused or frame is None or frame.size == 0:
            return None, False, self.state

        self.frame_count += 1
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            annotated_frame = frame.copy()
        except:
            return frame, False, self.state

        current_fall_detected = False
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw pose
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Extract metrics
            metrics = self._extract_metrics(landmarks, frame.shape)
            self.metrics_buffer.append(metrics)
            self.prev_landmarks = landmarks
            
            # Heuristic detection (every frame)
            heuristic_fall = self._heuristic_fall_detection(metrics)
            
            # ML detection (every 5 frames when buffer is full)
            ml_fall = False
            ml_prob = 0.0
            if self.xgb_clf and len(self.metrics_buffer) == self.window_size and self.frame_count % 5 == 0:
                features = self._compute_ml_features()
                if features is not None:
                    try:
                        ml_prob = self.xgb_clf.predict_proba(features)[0][1]
                        ml_fall = ml_prob > 0.5
                    except Exception as e:
                        print(f"ML prediction error: {e}")
            
            # Combine detections
            if self.fall_cooldown == 0:
                if heuristic_fall or ml_fall:
                    self.state = "FALLEN"
                    current_fall_detected = True
                    self.fall_cooldown = 90  # 3 second cooldown
                    print(f"[FALL DETECTED] Heuristic: {heuristic_fall}, ML: {ml_fall} (prob: {ml_prob:.2f})")
                else:
                    self.state = "STANDING"
            
            if self.fall_cooldown > 0:
                self.fall_cooldown -= 1

            # Debug overlay
            mode = "XGBoost+Heuristic" if self.xgb_clf else "Heuristic"
            cv2.putText(annotated_frame, f"Mode: {mode} | State: {self.state}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Speed: {metrics['com_speed']:.0f} | Angle: {metrics['torso_angle_deg']:.0f}°", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Buffer for replay
        try:
            replay_frame = cv2.resize(annotated_frame, (320, 240))
            self.frame_buffer.append(replay_frame)
        except:
            pass

        return annotated_frame, current_fall_detected, self.state

    def get_replay_clip(self):
        return list(self.frame_buffer)[::2]

    def save_replay(self, output_dir='./checkpoints'):
        """Save the current replay buffer as a video file"""
        if len(self.frame_buffer) == 0:
            return None
        
        try:
            # Get frames from buffer
            frames = list(self.frame_buffer)
            if not frames:
                return None
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'fall_detection_{timestamp}.mp4')
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get frame dimensions (frames are already resized to 320x240)
            h, w = frames[0].shape[:2]
            
            # Create VideoWriter (use mp4v codec)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10  # Save at 10 FPS
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Write frames
            for frame in frames:
                out.write(frame)
            out.release()
            
            print(f"[REPLAY SAVED] {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving replay: {e}")
            return None

    def toggle_pause(self):
        self.paused = not self.paused

    def reset(self):
        self.frame_buffer.clear()
        self.metrics_buffer.clear()
        self.state = "STANDING"
        self.fall_cooldown = 0
        self.consecutive_fall_frames = 0
        self.prev_com = None
        self.prev_com_speed = 0
        self.prev_torso_angle = 0