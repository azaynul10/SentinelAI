import numpy as np
import cv2
from collections import deque
import threading
import time
import os

class CircularFrameBuffer:
    """
    Thread-safe circular buffer to store the last N seconds of video and audio.
    Used for 15-second backtracking when a fall is detected.
    """
    def __init__(self, duration_sec=15, fps=15, audio_sample_rate=16000):
        self.duration_sec = duration_sec
        self.fps = fps
        self.sample_rate = audio_sample_rate
        
        # Calculate buffer sizes
        self.max_frames = int(duration_sec * fps)
        # Audio comes in chunks, we'll store raw samples or chunks.
        # Storing raw samples is cleaner for reconstruction but requires more management.
        # We will store (audio_chunk, timestamp) tuples.
        self.max_audio_duration = duration_sec
        
        self.frame_buffer = deque(maxlen=self.max_frames)
        self.audio_buffer = deque() # Will manually manage size by time
        
        self.lock = threading.Lock()
        
    def push_frame(self, frame, timestamp):
        """Add a video frame with timestamp."""
        with self.lock:
            # Resize frame for storage efficiency if needed, but keeping full res for now
            self.frame_buffer.append((frame, timestamp))
            
    def push_audio(self, audio_data, timestamp, duration):
        """
        Add audio data (numpy array).
        timestamp: start time of this chunk
        duration: duration of this chunk in seconds
        """
        with self.lock:
            self.audio_buffer.append((audio_data, timestamp, duration))
            
            # Prune old audio
            while self.audio_buffer:
                first_chunk_end_time = self.audio_buffer[0][1] + self.audio_buffer[0][2]
                last_chunk_end_time = self.audio_buffer[-1][1] + self.audio_buffer[-1][2]
                
                if last_chunk_end_time - self.audio_buffer[0][1] > self.duration_sec + 2.0: # Keep a bit extra margin
                    self.audio_buffer.popleft()
                else:
                    break
                    
    def get_replay_data(self):
        """
        Return a snapshot of the current buffer.
        Returns:
            frames: list of (frame, timestamp)
            audio: list of (audio_chunk, timestamp, duration)
        """
        with self.lock:
            return list(self.frame_buffer), list(self.audio_buffer)

    def save_replay_clip(self, output_path, event_timestamp):
        """
        Saves the buffer content to a video file.
        Requires moviepy for audio-video muxing.
        """
        try:
            from moviepy.editor import ImageSequenceClip, AudioFileClip, AudioClip
            import soundfile as sf
        except ImportError:
            print("Warning: moviepy/soundfile not found. Saving video-only replay using OpenCV.")
            if not frames: return None
            
            # Fallback: Save Video Only using OpenCV
            try:
                # Sort frames
                frames.sort(key=lambda x: x[1])
                h, w, _ = frames[0][0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
                for f, _ in frames:
                    out.write(f)
                out.release()
                return output_path
            except Exception as e:
                print(f"OpenCV fallback failed: {e}")
                return None

        frames, audio_chunks = self.get_replay_data()
        
        if not frames:
            print("Buffer empty, cannot save replay.")
            return None
            
        # 1. Prepare Video
        # timestamps are assumed monotonic. Calculate roughly FPS or use fixed.
        # using fixed FPS from config is safer for alignment
        
        # Sort frames by timestamp just in case
        frames.sort(key=lambda x: x[1])
        
        video_frames = [cv2.cvtColor(f[0], cv2.COLOR_BGR2RGB) for f in frames]
        
        if not video_frames:
            return None
            
        clip = ImageSequenceClip(video_frames, fps=self.fps)
        
        # 2. Prepare Audio
        if audio_chunks:
            audio_chunks.sort(key=lambda x: x[1])
            
            # Flatten audio
            # Note: This assumes continuous audio. If there are gaps, we should pad.
            # For simplicity in this version, we concatenate.
            full_audio = np.concatenate([c[0] for c in audio_chunks])
            
            # Create a temporary audio file to load into moviepy (safest way to handle various formats)
            temp_audio_path = output_path + ".wav"
            sf.write(temp_audio_path, full_audio, self.sample_rate)
            
            # Trim audio to match video duration if necessary
            audio_clip = AudioFileClip(temp_audio_path)
            
            # Handle duration mismatch
            if audio_clip.duration > clip.duration:
                audio_clip = audio_clip.subclip(0, clip.duration)
            
            final_clip = clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
            
            # Cleanup
            audio_clip.close()
            try:
                os.remove(temp_audio_path)
            except:
                pass
        else:
            final_clip = clip
            final_clip.write_videofile(output_path, codec='libx264', logger=None)
            
        final_clip.close()
        return output_path
