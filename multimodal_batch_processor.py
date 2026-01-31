
import os
import sys
import time
import json
import tempfile
import subprocess
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import torch

# Import existing modules
# We assume these files are in the same directory
import audio_inference
from audio_inference import AudioFallDetector

# Configuration Defaults
DEFAULT_XGB_MODEL = 'xgb_final_model.json'
DEFAULT_AUDIO_MODEL = 'checkpoints/fold_1_best/model_state.pt'

def load_module(mod_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_xgb(model_path):
    import xgboost as xgb
    if not os.path.exists(model_path):
        print(f"XGBoost model not found at {model_path}")
        return None
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        return clf
    except Exception as e:
        print(f"Error loading XGBoost: {e}")
        return None

def extract_audio_to_array(video_path, sr=16000):
    """Extract audio using ffmpeg to a temp wav file, then load with librosa."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        # ffmpeg: extract audio, mono, 16kHz
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", tmp_path
        ]
        # Suppress output unless error
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load with librosa (lazy loaded inside AudioFallDetector usually, but we need it here)
        import librosa
        wav, _ = librosa.load(tmp_path, sr=sr, mono=True)
        os.remove(tmp_path)
        return wav
    except Exception as e:
        print(f"Audio extraction failed (video might be silent): {e}")
        return np.zeros(sr * 5) # Return 5s silence

def soft_voting_ensemble(audio_prob, vision_prob, audio_weight=0.5, vision_weight=0.5):
    """
    Weighted Soft Voting.
    Returns: fused_prob, prediction_index (0=Normal, 1=Fall), fused_prob_scalar
    """
    total = audio_weight + vision_weight
    aw = audio_weight / total
    vw = vision_weight / total
    
    # Prob is [Normal_Prob, Fall_Prob]
    # We assume inputs are arrays/tuples of 2 floats
    
    # Audio/Vision probs might be single floats (Fall Conf) or [p0, p1]
    # Normalize to [p0, p1]
    if isinstance(audio_prob, (float, np.floating)):
        a_p = np.array([1.0 - audio_prob, audio_prob])
    else:
        a_p = np.array(audio_prob)
        
    if isinstance(vision_prob, (float, np.floating)):
        v_p = np.array([1.0 - vision_prob, vision_prob])
    else:
        v_p = np.array(vision_prob)
        
    ens = aw * a_p + vw * v_p
    idx = int(np.argmax(ens))
    return ens, idx, float(ens[1]) # Return Fall Probability

def analyze_scene_brightness(frame):
    """Calculate average brightness to detect low-light conditions."""
    if frame is None: return 128.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

class MultimodalBatchProcessor:
    def __init__(self, xgb_path=DEFAULT_XGB_MODEL, audio_path=DEFAULT_AUDIO_MODEL):
        print("Initializing Logic...")
        self.xgb_clf = load_xgb(xgb_path)
        
        # Load helper modules (assume in CWD)
        try:
            self.per_frame = load_module("per_frame_best", "per-frame-best.py")
            self.fe_eng = load_module("final_feature_eng_best", "final-feature-eng-best.py")
        except Exception as e:
            print(f"Error loading feature engineering modules: {e}")
            self.per_frame = None
            self.fe_eng = None

        # Initialize Audio Detector
        print("Initializing Audio Detector...")
        self.audio_detector = AudioFallDetector(model_path=audio_path, threshold=0.5)

    def process_folder(self, folder_path, output_dir):
        """Process all videos in a folder."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Found {len(video_files)} videos in {folder_path}")
        
        for v_file in video_files:
            v_path = os.path.join(folder_path, v_file)
            out_name = f"{os.path.splitext(v_file)[0]}_annotated.mp4"
            out_path = os.path.join(output_dir, out_name)
            
            print(f"Processing {v_file}...")
            fall_events = self.process_single_video(v_path, out_path)
            
            summary = {
                "file": v_file,
                "fall_detected": len(fall_events) > 0,
                "events": fall_events,
                "output_video": out_path
            }
            results.append(summary)
            
        # Save JSON Report
        json_path = os.path.join(output_dir, f"report_{int(time.time())}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Batch processing complete. Report saved to {json_path}")
        return json_path

    def process_single_video(self, video_path, out_path, win_s=3.0):
        # 1. Extract Audio
        sr = 16000
        full_audio = extract_audio_to_array(video_path, sr=sr)
        
        # 2. Open Video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Temp video for annotation (no audio yet)
        tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = cv2.VideoWriter(tmp_vid, fourcc, fps, (width, height))
        
        win_frames = int(win_s * fps)
        frames_buffer = []
        
        fall_events = []
        is_fall_sticky = False
        
        frame_idx = 0
        seg_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frames_buffer.append(frame)
                
                # We process in sliding WINDOWS, but we also need to write every frame.
                # To keep it simple and match user's logic efficiently:
                # We'll run inference periodically (every 'win_frames' or overlapping?)
                # user's logic: read whole window, process, write whole window.
                # This implies non-overlapping windows for efficiency in the user's script.
                
                if len(frames_buffer) >= win_frames:
                    # PROCESS WINDOW
                    window_config = self._process_window_logic(
                        frames_buffer.copy(), 
                        full_audio, 
                        seg_idx, 
                        win_s, 
                        sr, 
                        fps
                    )
                    
                    # Annotate & Write
                    for i, fr in enumerate(frames_buffer):
                        annotated = self._annotate_frame(fr, window_config)
                        writer.write(annotated)
                        
                        # Record event if fall
                        if window_config['is_fall'] and window_config['ens_sf'] > 0.8:
                            event_time = (frame_idx - win_frames + i) / fps
                            # De-duplicate events (simple 1s coalescing)
                            if not fall_events or (event_time - fall_events[-1]['timestamp'] > 2.0):
                                fall_events.append({
                                    "timestamp": float(f"{event_time:.2f}"),
                                    "confidence": float(f"{window_config['ens_sf']:.2f}"),
                                    "type": "FALL"
                                })
                    
                    frames_buffer = [] # Clear buffer
                    seg_idx += 1
                
                frame_idx += 1
                
            # Process remaining frames
            if frames_buffer:
                 # Last partial window
                 window_config = self._process_window_logic(
                        frames_buffer, full_audio, seg_idx, win_s, sr, fps
                 )
                 for fr in frames_buffer:
                     writer.write(self._annotate_frame(fr, window_config))
        
        finally:
            cap.release()
            writer.release()
            
        # 3. Mux Audio Back
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_vid,
                "-i", video_path,
                "-c:v", "copy",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", "192k",
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmp_vid)
        except Exception:
            # Fallback if ffmpeg mux fails (e.g., no audio stream in source)
            # Just rename temp video
            if os.path.exists(tmp_vid):
                import shutil
                shutil.move(tmp_vid, out_path)
                
        return fall_events

    def _process_window_logic(self, frames, full_audio, seg_idx, win_s, sr, fps):
        # 1. Vision Inference (XGB)
        # We need to extract features from these frames using per_frame logic
        # per_frame expects a video file path usually, or lists.
        # User's script writes to temp file then extracts.
        
        v_prob = [0.5, 0.5]
        brightness = 128.0
        
        if self.per_frame and self.fe_eng and self.xgb_clf:
            try:
                # Calc brightness of middle frame
                brightness = analyze_scene_brightness(frames[len(frames)//2])
                
                # Write temp chunk for feature extraction
                tmp_chunk = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(tmp_chunk, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for f in frames: vw.write(f)
                vw.release()
                
                # Extract
                df_pf = self.per_frame.extract_per_frame(tmp_chunk, fps=fps, n=len(frames))
                os.remove(tmp_chunk)
                
                if df_pf is not None and not df_pf.empty:
                    feats = self.fe_eng.extract_advanced_features(df_pf)
                    # Align cols
                    row = pd.DataFrame([feats])
                    if hasattr(self.xgb_clf, "feature_names_in_"):
                        names = list(self.xgb_clf.feature_names_in_)
                        for c in names:
                            if c not in row: row[c] = 0.0
                        row = row.reindex(columns=names)
                    
                    v_prob = self.xgb_clf.predict_proba(row)[0] # [p_norm, p_fall]
            except Exception as e:
                print(f"Vision Error: {e}")

        # 2. Audio Inference (AST)
        # Extract segment from full audio
        start_sample = int(seg_idx * win_s * sr)
        end_sample = int((seg_idx + 1) * win_s * sr)
        
        # Padding
        audio_seg = full_audio[start_sample:end_sample] if start_sample < len(full_audio) else np.zeros(int(win_s*sr))
        if len(audio_seg) < int(win_s*sr):
            audio_seg = np.pad(audio_seg, (0, int(win_s*sr) - len(audio_seg)), 'constant')
            
        # Detect
        # AudioFallDetector expects 4s clip usually, we have 3s window.
        # We pad it to 4s if needed or let detector handle it (it pads/crops).
        is_fall_a, conf_a, label_a = self.audio_detector.detect(audio_seg)
        a_prob = [1.0 - conf_a, conf_a] # Normalize to [norm, fall]

        # 3. Dynamic Weighting
        # "if it is low light then audio is preferred"
        w_v, w_a = 0.5, 0.5
        if brightness < 60: # Low light
            w_a = 0.8
            w_v = 0.2
            
        # 4. Ensemble
        ens_prob, idx, ens_sf = soft_voting_ensemble(a_prob, v_prob, w_a, w_v)
        
        return {
            "is_fall": idx == 1,
            "ens_sf": ens_sf,
            "txt_v": f"VIS: {v_prob[1]:.2f}",
            "txt_a": f"AUD: {a_prob[1]:.2f}",
            "txt_e": f"ENS: {ens_sf:.2f}",
            "mode": "LowLight" if brightness < 60 else "Normal"
        }

    def _annotate_frame(self, frame, config):
        disp = frame.copy()
        
        # Draw Box
        cv2.rectangle(disp, (5, 5), (300, 100), (0, 0, 0), -1)
        
        # Colors
        c_norm = (0, 255, 0)
        c_fall = (0, 0, 255)
        
        # Text
        cv2.putText(disp, config['txt_v'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(disp, config['txt_a'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        c_ens = c_fall if config['is_fall'] else c_norm
        cv2.putText(disp, config['txt_e'] + f" [{config['mode']}]", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_ens, 2)
        
        return disp

if __name__ == "__main__":
    # Command Line Interface
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Input folder of videos")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    args = parser.parse_args()
    
    processor = MultimodalBatchProcessor()
    processor.process_folder(args.folder, args.output)
