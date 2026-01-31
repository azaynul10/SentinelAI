# pose_fall_detection.py - Hybrid Detection for Overhead Cameras
# Combines: MediaPipe Pose + Contour Analysis + Motion Detection
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from threading import Lock
import json


class IoTAlertManager:
    """IoT Alert Manager for fall notifications"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.webhook_url = self.config.get('webhook_url', None)
        self.mqtt_broker = self.config.get('mqtt_broker', None)
        self.alert_history = []
        self.last_alert_time = 0
        self.cooldown_seconds = 30
        
    def send_alert(self, fall_data):
        """Send fall alert through configured channels"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown_seconds:
            return False
            
        self.last_alert_time = current_time
        alert = {
            'type': 'FALL_DETECTED',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'unix_time': current_time,
            'data': fall_data,
            'device_id': self.config.get('device_id', 'fall_detector_001')
        }
        self.alert_history.append(alert)
        
        if self.webhook_url:
            try:
                import requests
                requests.post(self.webhook_url, json=alert, timeout=5)
            except Exception as e:
                print(f"[IoT] Webhook failed: {e}")
                
        print(f"[IoT ALERT] Fall detected at {alert['timestamp']}")
        return True
        
    def get_alert_history(self):
        return self.alert_history[-10:]


class PoseFallDetector:
    """
    Hybrid Fall Detector - Works with overhead/angled cameras
    
    Detection Methods:
    1. MediaPipe Pose (when visible)
    2. Contour Analysis (shape-based detection)
    3. Motion Detection (sudden movements)
    4. Background Subtraction (person segmentation)
    """
    
    # States
    STATE_STANDING = "STANDING"
    STATE_SITTING = "SITTING"
    STATE_BENDING = "BENDING"
    STATE_FALLING = "FALLING"
    STATE_FALLEN = "FALLEN"
    STATE_LYING = "LYING"  # New state for already lying down
    
    def __init__(self, buffer_seconds=10, target_fps=15, iot_config=None):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,  # Lower threshold for difficult angles
            min_tracking_confidence=0.3,
            enable_segmentation=True  # Enable for better person detection
        )
        
        # Buffer configuration
        self.buffer_fps = target_fps
        self.buffer_size = buffer_seconds * target_fps
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = Lock()
        self.buffer_width = 320
        self.buffer_height = 240
        
        # State variables
        self.state = self.STATE_STANDING
        self.prev_state = self.STATE_STANDING
        self.paused = False
        self.frame_count = 0
        self.fps = 30
        
        # Background subtractor for motion/person detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Tracking history
        self.aspect_ratio_history = deque(maxlen=30)
        self.centroid_history = deque(maxlen=30)
        self.area_history = deque(maxlen=30)
        self.pose_detected_history = deque(maxlen=10)
        
        # Fall detection state
        self.fall_start_time = 0
        self.consecutive_fallen_frames = 0
        self.lying_frames = 0
        
        # Thresholds - tuned for webcam detection
        self.ASPECT_RATIO_LYING = 1.2      # Width/Height when lying (lowered for better detection)
        self.ASPECT_RATIO_STANDING = 0.8   # Width/Height when standing
        self.MIN_PERSON_AREA = 3000        # Minimum contour area for person
        self.VERTICAL_VELOCITY_THRESHOLD = 15  # Pixels per frame (lowered for sensitivity)
        self.FALLEN_CONFIRM_FRAMES = 8     # Faster fall confirmation
        self.LYING_CONFIRM_FRAMES = 15     # Faster lying detection
        
        # Debug mode
        self.debug = True
        
        # Frame skip for buffer
        self.frame_skip = max(1, self.fps // self.buffer_fps)
        
        # Reference frame for motion detection
        self.prev_gray = None
        
        # Drawing specs
        self.landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3
        )
        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 255), thickness=2, circle_radius=2
        )
        
        # IoT
        self.iot_manager = IoTAlertManager(iot_config)
        
        # Detection method used
        self.detection_method = "initializing"

    def process_frame(self, frame):
        """Process frame with hybrid detection"""
        if self.paused:
            return None, False, self.state
            
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None, False, self.state

        self.frame_count += 1
        current_time = time.time()
        
        try:
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Convert for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try MediaPipe first
            pose_results = self.pose.process(rgb_frame)
            pose_detected = pose_results.pose_landmarks is not None
            self.pose_detected_history.append(pose_detected)
            
            # Initialize metrics
            metrics = {
                'aspect_ratio': 0,
                'centroid': None,
                'area': 0,
                'pose_detected': pose_detected,
                'velocity': 0,
                'is_horizontal': False,
                'detection_method': 'none'
            }
            
            if pose_detected:
                # Use MediaPipe detection
                metrics = self._process_pose_detection(pose_results, annotated_frame, h, w)
                metrics['detection_method'] = 'pose'
                self.detection_method = "MediaPipe Pose"
            else:
                # Fallback to contour-based detection
                metrics = self._process_contour_detection(frame, gray_frame, annotated_frame, h, w)
                metrics['detection_method'] = 'contour'
                self.detection_method = "Contour Analysis"
            
            # Update history
            self._update_history(metrics)
            
            # State machine
            fall_confirmed = self._update_state(metrics, current_time)
            
            # IoT alert
            if fall_confirmed:
                self.iot_manager.send_alert({
                    'state': self.state,
                    'detection_method': metrics['detection_method'],
                    'metrics': {
                        'aspect_ratio': metrics['aspect_ratio'],
                        'is_horizontal': metrics['is_horizontal']
                    }
                })
            
            # Draw status overlay
            self._draw_status(annotated_frame, metrics)
            
            # Buffer
            if self.frame_count % self.frame_skip == 0:
                self._add_to_buffer(annotated_frame)
            
            # Update previous frame
            self.prev_gray = gray_frame.copy()
            
            return annotated_frame, fall_confirmed, self.state
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame, False, self.state

    def _process_pose_detection(self, results, annotated_frame, h, w):
        """Process MediaPipe pose detection results"""
        landmarks = results.pose_landmarks.landmark
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.landmark_spec,
            self.connection_spec
        )
        
        # Calculate bounding box from landmarks
        visible_landmarks = [(lm.x * w, lm.y * h) for lm in landmarks if lm.visibility > 0.3]
        
        if not visible_landmarks:
            return {'aspect_ratio': 0, 'centroid': None, 'area': 0, 
                    'pose_detected': True, 'velocity': 0, 'is_horizontal': False}
        
        x_coords = [p[0] for p in visible_landmarks]
        y_coords = [p[1] for p in visible_landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        aspect_ratio = box_width / max(box_height, 1)
        centroid = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        area = box_width * box_height
        
        # Calculate velocity from centroid history
        velocity = 0
        if self.centroid_history and self.centroid_history[-1]:
            prev_centroid = self.centroid_history[-1]
            velocity = abs(centroid[1] - prev_centroid[1])
        
        # Determine if horizontal (lying down)
        is_horizontal = aspect_ratio > self.ASPECT_RATIO_LYING
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, 
                     (int(min_x), int(min_y)), 
                     (int(max_x), int(max_y)), 
                     (0, 255, 0) if not is_horizontal else (0, 0, 255), 2)
        
        return {
            'aspect_ratio': aspect_ratio,
            'centroid': centroid,
            'area': area,
            'pose_detected': True,
            'velocity': velocity,
            'is_horizontal': is_horizontal
        }

    def _process_contour_detection(self, frame, gray_frame, annotated_frame, h, w):
        """Fallback: Contour-based person detection for overhead cameras"""
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove shadows (gray pixels)
        fg_mask[fg_mask == 127] = 0
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No motion detected - try edge detection on full frame
            return self._process_edge_detection(gray_frame, annotated_frame, h, w)
        
        # Find largest contour (assumed to be person)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.MIN_PERSON_AREA:
            return self._process_edge_detection(gray_frame, annotated_frame, h, w)
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        
        aspect_ratio = bw / max(bh, 1)
        centroid = (x + bw/2, y + bh/2)
        
        # Calculate velocity
        velocity = 0
        if self.centroid_history and self.centroid_history[-1]:
            prev_centroid = self.centroid_history[-1]
            velocity = abs(centroid[1] - prev_centroid[1])
        
        is_horizontal = aspect_ratio > self.ASPECT_RATIO_LYING
        
        # Draw contour and bounding box
        cv2.drawContours(annotated_frame, [largest_contour], -1, (255, 255, 0), 2)
        color = (0, 0, 255) if is_horizontal else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x, y), (x + bw, y + bh), color, 2)
        
        # Label
        cv2.putText(annotated_frame, f"Contour Detection", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return {
            'aspect_ratio': aspect_ratio,
            'centroid': centroid,
            'area': area,
            'pose_detected': False,
            'velocity': velocity,
            'is_horizontal': is_horizontal
        }

    def _process_edge_detection(self, gray_frame, annotated_frame, h, w):
        """Last resort: Edge-based detection for static scenes"""
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'aspect_ratio': 0, 'centroid': None, 'area': 0,
                    'pose_detected': False, 'velocity': 0, 'is_horizontal': False}
        
        # Filter contours by area and find most person-like shape
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_PERSON_AREA]
        
        if not valid_contours:
            return {'aspect_ratio': 0, 'centroid': None, 'area': 0,
                    'pose_detected': False, 'velocity': 0, 'is_horizontal': False}
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        
        aspect_ratio = bw / max(bh, 1)
        centroid = (x + bw/2, y + bh/2)
        area = cv2.contourArea(largest_contour)
        
        velocity = 0
        if self.centroid_history and self.centroid_history[-1]:
            prev_centroid = self.centroid_history[-1]
            velocity = abs(centroid[1] - prev_centroid[1])
        
        is_horizontal = aspect_ratio > self.ASPECT_RATIO_LYING
        
        # Draw
        cv2.drawContours(annotated_frame, [largest_contour], -1, (255, 0, 255), 2)
        color = (0, 0, 255) if is_horizontal else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(annotated_frame, f"Edge Detection", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return {
            'aspect_ratio': aspect_ratio,
            'centroid': centroid,
            'area': area,
            'pose_detected': False,
            'velocity': velocity,
            'is_horizontal': is_horizontal
        }

    def _update_history(self, metrics):
        """Update tracking history"""
        self.aspect_ratio_history.append(metrics['aspect_ratio'])
        self.centroid_history.append(metrics['centroid'])
        self.area_history.append(metrics['area'])

    def _update_state(self, metrics, current_time):
        """State machine with better lying detection"""
        self.prev_state = self.state
        fall_confirmed = False
        
        aspect_ratio = metrics['aspect_ratio']
        is_horizontal = metrics['is_horizontal']
        velocity = metrics['velocity']
        
        # Debug logging
        if self.debug and self.frame_count % 30 == 0:  # Log every 30 frames
            print(f"[DEBUG] State={self.state}, AR={aspect_ratio:.2f}, V={velocity:.1f}, Horiz={is_horizontal}")
        
        # Check for rapid descent (actual falling motion)
        rapid_motion = velocity > self.VERTICAL_VELOCITY_THRESHOLD
        
        # Check consistent horizontal position
        if len(self.aspect_ratio_history) >= 5:
            recent_ratios = list(self.aspect_ratio_history)[-5:]
            avg_ratio = np.mean(recent_ratios)
            consistently_horizontal = avg_ratio > self.ASPECT_RATIO_LYING
        else:
            consistently_horizontal = is_horizontal
        
        if self.state == self.STATE_STANDING:
            if rapid_motion:
                self.state = self.STATE_FALLING
                self.fall_start_time = current_time
                self.consecutive_fallen_frames = 0
                print(f"[FALL] Rapid motion detected! Velocity={velocity:.1f}")
            elif consistently_horizontal:
                # Person is already lying - might have been lying when video started
                self.lying_frames += 1
                if self.lying_frames > self.LYING_CONFIRM_FRAMES:
                    self.state = self.STATE_LYING
                    print(f"[STATE] Person is lying down (no fall detected)")
            else:
                self.lying_frames = 0
                
        elif self.state == self.STATE_LYING:
            # Already lying - this is the initial state detection
            if not is_horizontal and aspect_ratio < self.ASPECT_RATIO_STANDING:
                # Person stood up
                self.state = self.STATE_STANDING
                self.lying_frames = 0
            elif rapid_motion:
                # Motion while lying could be struggling or secondary fall
                pass
                
        elif self.state == self.STATE_FALLING:
            if is_horizontal:
                self.consecutive_fallen_frames += 1
            else:
                self.consecutive_fallen_frames = max(0, self.consecutive_fallen_frames - 2)
            
            if self.consecutive_fallen_frames > self.FALLEN_CONFIRM_FRAMES:
                self.state = self.STATE_FALLEN
                fall_confirmed = True
                print(f"[ALERT] FALL CONFIRMED!")
            
            if current_time - self.fall_start_time > 3.0 and not fall_confirmed:
                self.state = self.STATE_STANDING
                self.consecutive_fallen_frames = 0
                print(f"[STATE] Fall not confirmed, returning to STANDING")
                
        elif self.state == self.STATE_FALLEN:
            if not is_horizontal and aspect_ratio < 0.8:
                self.state = self.STATE_STANDING
                self.consecutive_fallen_frames = 0
                self.lying_frames = 0
        
        return fall_confirmed

    def _draw_status(self, frame, metrics):
        """Draw status overlay"""
        h, w = frame.shape[:2]
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # State colors
        state_colors = {
            self.STATE_STANDING: (0, 255, 0),
            self.STATE_SITTING: (255, 255, 0),
            self.STATE_BENDING: (0, 165, 255),
            self.STATE_FALLING: (0, 100, 255),
            self.STATE_FALLEN: (0, 0, 255),
            self.STATE_LYING: (255, 0, 255)  # Magenta for lying
        }
        color = state_colors.get(self.state, (255, 255, 255))
        
        # Draw info
        cv2.putText(frame, f"State: {self.state}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Detection: {self.detection_method}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Aspect Ratio: {metrics['aspect_ratio']:.2f} (>{self.ASPECT_RATIO_LYING:.1f}=lying)", 
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Horizontal: {metrics['is_horizontal']} | Velocity: {metrics['velocity']:.1f}", 
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Fallen Frames: {self.consecutive_fallen_frames}/{self.FALLEN_CONFIRM_FRAMES}", 
                    (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _add_to_buffer(self, frame):
        """Add frame to replay buffer"""
        try:
            small_frame = cv2.resize(frame, (self.buffer_width, self.buffer_height),
                                     interpolation=cv2.INTER_AREA)
            with self.buffer_lock:
                self.frame_buffer.append(small_frame)
        except Exception:
            pass

    def get_replay_clip(self):
        """Get buffered frames"""
        with self.buffer_lock:
            return list(self.frame_buffer)

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def reset(self):
        """Reset detector"""
        with self.buffer_lock:
            self.frame_buffer.clear()
        
        self.state = self.STATE_STANDING
        self.prev_state = self.STATE_STANDING
        self.consecutive_fallen_frames = 0
        self.lying_frames = 0
        self.frame_count = 0
        self.aspect_ratio_history.clear()
        self.centroid_history.clear()
        self.area_history.clear()
        self.pose_detected_history.clear()
        self.prev_gray = None
        
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    def get_stats(self):
        return {
            'state': self.state,
            'paused': self.paused,
            'frame_count': self.frame_count,
            'buffer_frames': len(self.frame_buffer),
            'detection_method': self.detection_method,
            'iot_alerts': len(self.iot_manager.alert_history)
        }
    
    def get_iot_alerts(self):
        return self.iot_manager.get_alert_history()