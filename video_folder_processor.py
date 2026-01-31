import os
import glob
import cv2
import numpy as np
import time
import json
import threading
import traceback

try:
    from moviepy import VideoFileClip
except ImportError:
    print("MoviePy not installed. Audio extraction will fail.")
    VideoFileClip = None

from multimodal_detector import MultimodalFallDetector

class VideoFolderProcessor:
    """
    Bio-pipeline for processing videos (Single or Batch).
    1. Iterates through video files.
    2. Extracts audio (if available).
    3. Feeds video/audio to MultimodalFallDetector.
    4. Collects results (fall events).
    """

    def __init__(self, detector_factory):
        self.detector_factory = detector_factory
        self.processing = False
        self.current_job_id = None
        self.progress = {} # {job_id: {total: N, current: M, current_file: "..."}}

    # --- Public Async Methods ---

    def process_folder_async(self, folder_path, job_id=None):
        """Starts processing a folder in background"""
        try:
            supported_exts = ('.mp4', '.avi', '.mov', '.mkv')
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]
            files = [os.path.join(folder_path, f) for f in files]
        except Exception as e:
            print(f"Error accessing folder: {e}")
            files = []
            
        return self._start_job(files, job_id)

    def process_video_async(self, video_path, job_id=None):
        """Starts processing a single video in background"""
        files = [video_path]
        return self._start_job(files, job_id)

    def _start_job(self, files, job_id=None):
        if not job_id:
            job_id = str(int(time.time()))
        
        self.current_job_id = job_id
        self.progress[job_id] = {"total": len(files), "current": 0, "status": "initializing"}
        
        t = threading.Thread(target=self._worker_logic, args=(files, job_id))
        t.start()
        return job_id

    # --- Worker Logic ---

    def _worker_logic(self, files, job_id):
        self.processing = True
        print(f"Starting job {job_id} with {len(files)} files...")
        
        # Instantiate PRIVATE detector for this thread
        print(f"Job {job_id}: Instantiating local detector...")
        local_detector = self.detector_factory()
        
        try:
            self.progress[job_id]["total"] = len(files)
            self.progress[job_id]["status"] = "processing"
            
            # Start Detector Threads (Internal queues)
            local_detector.start_threads()
            
            results = []
            
            for idx, video_path in enumerate(files):
                self.progress[job_id]["current"] = idx + 1
                self.progress[job_id]["current_file"] = os.path.basename(video_path)
                print(f"Processing: {video_path}")
                
                try:
                    if not os.path.exists(video_path):
                         print(f"File not found: {video_path}")
                         continue
                         
                    file_events = self._process_single_video(local_detector, video_path)
                    if file_events:
                        results.append({
                            "file": os.path.basename(video_path),
                            "events": file_events
                        })
                except Exception as e:
                    print(f"Error processing file {video_path}: {e}")
                    traceback.print_exc()
            
            self.progress[job_id]["status"] = "completed"
            self.progress[job_id]["results"] = results
            print(f"Job {job_id} completed.")
            
            # Save JSON Report
            try:
                os.makedirs("./results", exist_ok=True)
                with open(f"./results/report_{job_id}.json", "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"Error saving report: {e}")

        except Exception as e:
            print(f"Batch Processing CRITICAL FAILURE: {e}")
            traceback.print_exc()
            self.progress[job_id]["status"] = "failed"
            self.progress[job_id]["error"] = str(e)
            
        finally:
            # Always clean up
            self.processing = False
            try:
                print(f"Job {job_id}: Cleaning up local detector...")
                local_detector.stop_threads()
                
                # Explicitly close pose resources if possible
                if hasattr(local_detector, 'pose_detector'):
                    local_detector.pose_detector.close()
                    
            except Exception as e:
                print(f"Error stopping threads: {e}")

    def _process_single_video(self, detector, video_path):
        """feeds a single video into the detector frame-by-frame + audio extraction."""
        detected_events = []
        
        # Capture events via callback hook injection
        def local_callback(event_data):
            detected_events.append(event_data)
        
        # Swap existing callback temporarily (NOT thread safe for concurrent jobs)
        # Since we have a unique detector per job, this IS safe now!
        original_callback = detector.alert_callback
        detector.alert_callback = local_callback
        
        # 1. Extract Audio
        audio_clip = None
        sr = 16000
        try:
            if VideoFileClip:
                clip = VideoFileClip(video_path)
                if clip.audio is not None:
                    try:
                        audio_array = clip.audio.to_soundarray(fps=sr)
                        if audio_array.ndim > 1:
                            audio_array = audio_array.mean(axis=1) # Mono
                        audio_clip = audio_array
                    except Exception as e:
                         print(f"Audio read error: {e}")
                clip.close()
                del clip # Explicit cleanup for MoviePy
        except Exception as e:
            print(f"Audio extraction warning: {e}")
            audio_clip = None

        # 2. Open Video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        # Process loop
        frame_idx = 0
        audio_idx = 0
        chunk_size = int(sr * 0.5) # 0.5s chunks for feeding
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            
            # Feed Video
            detector.process_frame(frame, timestamp)
            
            # Feed Audio
            if audio_clip is not None:
                current_time = timestamp
                audio_covered_time = audio_idx / sr
                
                if current_time >= audio_covered_time + 0.5:
                    end_idx = min(len(audio_clip), audio_idx + chunk_size)
                    chunk = audio_clip[audio_idx:end_idx]
                    if len(chunk) > 0:
                        detector.process_audio(chunk, timestamp, duration=len(chunk)/sr)
                    audio_idx = end_idx
            
            # Sync Throttling
            # Since we are in SYNC MODE in pose_fall_detection, process_frame blocks.
            # But invalidating the time simulation?
            # We add a small sleep to allow other system events and check stop signals
            time.sleep(1.0 / (fps * 2.0)) # Faster than realtime is OK
            
            frame_idx += 1
            
        cap.release()
        
        # Give threads time to finish buffer
        time.sleep(2.0)
        
        # Restore callback (good practice)
        detector.alert_callback = original_callback
        
        return detected_events
