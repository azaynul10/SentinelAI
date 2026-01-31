import cv2
import numpy as np
import tempfile
import sys
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

# Import your local modules
from pose_fall_detection import AdvancedPoseFallDetector
from audio_inference import AudioFallDetector

def extract_audio_to_array(video_path, sr=16000):
    """Extract audio from video using FFmpeg and load as numpy array"""
    print(f"Extracting audio from {video_path}...")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", tmp_path
        ]
        # Run ffmpeg with reduced verbosity
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Lazy import librosa here to save time if not needed earlier
        import librosa
        wav, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return wav
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return np.zeros(int(sr * 10)) # Return 10s of silence on error
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def soft_voting_ensemble(audio_prob, vision_prob, audio_weight=0.5, vision_weight=0.5):
    """Combine probabilities using weighted average"""
    # Probabilities are for the "FALL" class (index 1)
    
    ensemble_prob = (audio_weight * audio_prob + vision_weight * vision_prob) / (audio_weight + vision_weight)
    
    # Heuristic: If either is very high (>0.85), trust it more
    if audio_prob > 0.85 or vision_prob > 0.85:
        ensemble_prob = max(audio_prob, vision_prob)
        
    is_fall = ensemble_prob > 0.5
    return ensemble_prob, is_fall

def run_analysis(video_path, output_path=None):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_analyzed{ext}"

    print(f"Initializing models...")
    
    # 1. Initialize Pose (Vision) Detector
    # Uses the local XGBoost model
    pose_detector = AdvancedPoseFallDetector(model_path='xgb_final_model.json')
    
    # 2. Initialize Audio Detector
    # Uses the local AST model checkpoint
    audio_detector = None
    if os.path.exists('./checkpoints/fold_1_best/model_state.pt'):
        audio_detector = AudioFallDetector(model_path='./checkpoints/fold_1_best/model_state.pt')
    else:
        print("Warning: Audio model checkpoint not found. Audio detection will be disabled.")

    # 3. Audio Extraction
    sr = 16000
    full_audio = extract_audio_to_array(video_path, sr=sr)
    
    # Video Setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Writer (temp file first)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_video_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(tmp_video_out, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames at {fps:.2f} FPS...")
    
    frame_idx = 0
    audio_buffer_size = int(sr * 4.0) # 4 seconds buffer for audio model
    
    # Status tracking
    fall_detected_sticky = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. VISION PROCESSING
        # Update detector state
        # We access internal logic to separate display drawing from detection if needed,
        # but process_frame gives us a nice annotated frame already.
        # However, we want to overlay our OWN ensemble text.
        # So we use process_frame but ignore its internal annotation if we want clean text,
        # OR we just draw over it. 
        # Let's use the detector's internal annotation but add our stats.
        
        annotated_frame, vision_fall, state = pose_detector.process_frame(frame)
        
        # Get Vision Probability (Manually peek into detector)
        vision_prob = 0.0
        ml_features = pose_detector._compute_ml_features()
        if ml_features is not None and pose_detector.xgb_clf:
            try:
                # Features are [1, 84] shape
                vision_prob = pose_detector.xgb_clf.predict_proba(ml_features)[0][1]
            except:
                vision_prob = 0.5 if vision_fall else 0.1
        else:
            # Fallback if ML not ready (buffer filling) uses heuristic
            vision_prob = 0.9 if vision_fall else 0.1

        # 2. AUDIO PROCESSING
        audio_prob = 0.0
        if audio_detector and full_audio is not None:
            # Calculate current audio window corresponding to video time
            current_time = frame_idx / fps
            sample_idx = int(current_time * sr)
            
            # Simple version: Take window centered on current time, or ending at current time
            # Fall sounds are sudden, so looking at [now-2s, now+2s] or [now-4s, now] is good.
            # AST model is trained on specific window clicks.
            # Let's use a sliding window ending at current time (simulating real-time)
            
            start_samp = max(0, sample_idx - audio_buffer_size + int(0.5*sr)) # 4s window, slightly shifted
            end_samp = start_samp + audio_buffer_size
            
            if end_samp <= len(full_audio):
                audio_chunk = full_audio[start_samp:end_samp]
                # Run inference
                _, audio_prob, _ = audio_detector.detect(audio_chunk)
        
        # 3. ENSEMBLE
        ensemble_prob, ensemble_fall = soft_voting_ensemble(audio_prob, vision_prob)
        
        if ensemble_fall:
            fall_detected_sticky = True
            
        # 4. DRAW OVERLAY
        # We draw on the annotated frame returned by pose_detector
        disp = annotated_frame.copy() if annotated_frame is not None else frame.copy()
        
        # Overlay Box
        h, w = disp.shape[:2]
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color_safe = (0, 255, 0)
        color_alert = (0, 0, 255)
        
        lines = [
            f"TIME: {frame_idx/fps:.1f}s | STATE: {state}",
            f"VISION: {vision_prob*100:.1f}% ({'FALL' if vision_prob>0.5 else 'OK'})",
            f"AUDIO:  {audio_prob*100:.1f}% ({'FALL' if audio_prob>0.5 else 'OK'})",
            f"FUSED:  {ensemble_prob*100:.1f}% ({'FALL' if ensemble_fall else 'OK'})"
        ]
        
        # Draw background box
        cv2.rectangle(disp, (10, 10), (350, 130), (0, 0, 0), -1)
        
        for i, line in enumerate(lines):
            c = color_alert if "FALL" in line else color_safe
            if i == 0: c = (255, 255, 255) # White for time
            
            cv2.putText(disp, line, (20, 40 + i*25), font, scale, c, 2)
            
        if fall_detected_sticky:
             cv2.putText(disp, "FALL DETECTED", (w//2 - 150, h//2), font, 1.5, (0, 0, 255), 3)

        writer.write(disp)
        
        # print progress every 30 frames
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

        frame_idx += 1
    
    # Cleanup
    cap.release()
    writer.release()
    print("Video processing complete.")
    
    # 5. MUX AUDIO BACK
    # Add original audio to the output video
    print("Merging audio...")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_video_out,
            "-i", video_path,
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0", # Map video from temp, audio from original
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Success! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error merging audio (ffmpeg): {e}")
        print(f"Video saved without audio at: {tmp_video_out}")
    finally:
        if os.path.exists(tmp_video_out) and os.path.exists(output_path):
            os.remove(tmp_video_out)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_multimodal_analysis.py <path_to_video.mp4>")
        print("Example: python run_multimodal_analysis.py my_test_video.mp4")
    else:
        video_file = sys.argv[1]
        run_analysis(video_file)
