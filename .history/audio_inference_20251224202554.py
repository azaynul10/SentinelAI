# audio_inference.py - Optimized for Production
import torch
import torch.nn as nn
import numpy as np
import io
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Lazy imports for faster startup
librosa = None
ASTForAudioClassification = None
ASTFeatureExtractor = None


def _lazy_import_librosa():
    """Lazy import librosa (heavy dependency)"""
    global librosa
    if librosa is None:
        import librosa as _librosa
        librosa = _librosa
    return librosa


def _lazy_import_transformers():
    """Lazy import transformers components"""
    global ASTForAudioClassification, ASTFeatureExtractor
    if ASTForAudioClassification is None:
        from transformers import ASTForAudioClassification as _AST
        from transformers import ASTFeatureExtractor as _FE
        ASTForAudioClassification = _AST
        ASTFeatureExtractor = _FE
    return ASTForAudioClassification, ASTFeatureExtractor


class FixedASTModel(nn.Module):
    """
    Audio Spectrogram Transformer model for fall detection.
    Architecture matches training script for weight compatibility.
    """
    
    def __init__(self, dropout_rate=0.1, num_labels=2):
        super().__init__()
        
        AST, _ = _lazy_import_transformers()
        
        # Load pre-trained AST
        self.ast = AST.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Enhanced classifier (matches training script)
        hidden_size = self.ast.config.hidden_size
        
        self.enhanced_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels)
        )
        
        # Replace original classifier
        self.ast.classifier = self.enhanced_classifier

    def forward(self, input_values):
        return self.ast(input_values=input_values)


class AudioFallDetector:
    """
    Production-ready Audio Fall Detector.
    
    Features:
    - Single model load at initialization
    - Memory-efficient inference (no disk I/O)
    - Thread-safe with torch.no_grad()
    - Flexible input formats (bytes, BytesIO, numpy)
    - Configurable detection threshold
    """
    
    def __init__(self, 
                 model_path='checkpoints/fold_1_best/model_state.pt',
                 sample_rate=16000,
                 clip_duration=4.0,
                 threshold=0.5,
                 device=None):
        """
        Initialize the audio fall detector.
        
        Args:
            model_path: Path to trained model weights
            sample_rate: Audio sample rate (default 16kHz)
            clip_duration: Audio clip length in seconds (default 4s)
            threshold: Detection threshold (default 0.5)
            device: torch device (auto-detected if None)
        """
        print("[Audio] Initializing Audio Fall Detector...")
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"[Audio] Device: {self.device}")
        
        # Audio parameters
        self.sr = sample_rate
        self.clip_duration = clip_duration
        self.target_length = int(clip_duration * sample_rate)
        self.threshold = threshold
        
        # Load components
        print("[Audio] Loading feature extractor...")
        _, FE = _lazy_import_transformers()
        self.feature_extractor = FE.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        print("[Audio] Loading model...")
        self.model = FixedASTModel()
        
        # Load custom weights
        self._load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Warm up model (optional, reduces first inference latency)
        self._warmup()
        
        print("[Audio] Audio Fall Detector ready!")

    def _load_weights(self, model_path):
        """Load model weights with multiple format support"""
        if not model_path or not os.path.exists(model_path):
            print(f"[Audio] Warning: Model not found at {model_path}")
            print("[Audio] Running with pre-trained weights only")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(
                model_path, 
                map_location=self.device,
                weights_only=False
            )
            
            # Extract state dict from various formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load with partial matching (handles minor architecture changes)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"[Audio] Note: {len(missing)} missing keys (may be normal)")
            if unexpected:
                print(f"[Audio] Note: {len(unexpected)} unexpected keys")
            
            print(f"[Audio] Loaded weights from: {model_path}")
            
        except Exception as e:
            print(f"[Audio] Warning: Could not load weights: {e}")
            print("[Audio] Running with pre-trained weights only")

    def _warmup(self):
        """Warm up model with dummy input to reduce first inference latency"""
        try:
            dummy_audio = np.zeros(self.target_length, dtype=np.float32)
            input_values = self._preprocess(dummy_audio)
            with torch.no_grad():
                _ = self.model(input_values)
        except Exception:
            pass  # Warmup is optional

    def _preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio to model input tensor.
        
        Args:
            audio: numpy array of audio samples
            
        Returns:
            torch.Tensor ready for model input
        """
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        # Flatten if needed
        audio = audio.flatten()
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Pad or crop to target length
        if len(audio) > self.target_length:
            # Center crop
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        elif len(audio) < self.target_length:
            # Zero pad
            pad_length = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt"
        )
        
        return inputs['input_values'].to(self.device)

    def _inference(self, input_values: torch.Tensor) -> tuple:
        """
        Run inference on preprocessed input.
        
        Returns:
            tuple: (is_fall, probability, label)
        """
        with torch.no_grad():
            outputs = self.model(input_values)
            probs = torch.softmax(outputs.logits, dim=1)
            
            # Index 1 = FALL, Index 0 = NORMAL
            fall_prob = probs[0][1].item()
            is_fall = fall_prob > self.threshold
            label = "FALL" if is_fall else "NORMAL"
            
            return is_fall, fall_prob, label

    def detect(self, audio_source) -> tuple:
        """
        Detect fall from audio source.
        
        Args:
            audio_source: One of:
                - bytes: Raw audio file bytes
                - io.BytesIO: Audio buffer  
                - file-like object with read() method
                - np.ndarray: Pre-loaded audio samples
                
        Returns:
            tuple: (is_fall: bool, confidence: float, label: str)
        """
        try:
            # Handle file-like objects (Flask uploads)
            if hasattr(audio_source, 'read'):
                audio_bytes = audio_source.read()
                if hasattr(audio_source, 'seek'):
                    audio_source.seek(0)
                return self._detect_from_bytes(audio_bytes)
            
            # Handle bytes directly
            elif isinstance(audio_source, (bytes, bytearray)):
                return self._detect_from_bytes(audio_source)
            
            # Handle BytesIO
            elif isinstance(audio_source, io.BytesIO):
                audio_source.seek(0)
                return self._detect_from_bytes(audio_source.read())
            
            # Handle numpy array
            elif isinstance(audio_source, np.ndarray):
                return self._detect_from_array(audio_source)
            
            else:
                return False, 0.0, f"Unsupported type: {type(audio_source)}"
                
        except Exception as e:
            print(f"[Audio] Detection error: {e}")
            return False, 0.0, "ERROR"

    def _detect_from_bytes(self, audio_bytes: bytes) -> tuple:
        """Detect from raw audio bytes"""
        lib = _lazy_import_librosa()
        
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            audio, _ = lib.load(audio_buffer, sr=self.sr)
            return self._detect_from_array(audio)
        except Exception as e:
            print(f"[Audio] Bytes decode error: {e}")
            return False, 0.0, "DECODE_ERROR"

    def _detect_from_array(self, audio: np.ndarray) -> tuple:
        """Detect from numpy array"""
        try:
            input_values = self._preprocess(audio)
            return self._inference(input_values)
        except Exception as e:
            print(f"[Audio] Inference error: {e}")
            return False, 0.0, "INFERENCE_ERROR"

    def detect_file(self, file_path: str) -> tuple:
        """
        Detect from audio file path.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            tuple: (is_fall: bool, confidence: float, label: str)
        """
        if not os.path.exists(file_path):
            return False, 0.0, "FILE_NOT_FOUND"
        
        try:
            lib = _lazy_import_librosa()
            audio, _ = lib.load(file_path, sr=self.sr)
            return self._detect_from_array(audio)
        except Exception as e:
            print(f"[Audio] File load error: {e}")
            return False, 0.0, "FILE_ERROR"

    def detect_memory(self, audio_data) -> tuple:
        """
        Alias for detect() - maintains backward compatibility.
        """
        return self.detect(audio_data)

    def set_threshold(self, threshold: float):
        """Update detection threshold (0.0 to 1.0)"""
        self.threshold = max(0.0, min(1.0, threshold))

    def get_info(self) -> dict:
        """Get detector configuration info"""
        return {
            'device': str(self.device),
            'sample_rate': self.sr,
            'clip_duration': self.clip_duration,
            'target_samples': self.target_length,
            'threshold': self.threshold,
            'model_loaded': True
        }


# Convenience function for quick testing
def quick_test(audio_path: str, model_path: str = None):
    """
    Quick test function for command-line usage.
    
    Usage:
        python audio_inference.py test.wav
    """
    detector = AudioFallDetector(model_path=model_path)
    
    if os.path.isfile(audio_path):
        is_fall, prob, label = detector.detect_file(audio_path)
        print(f"\nResult: {label}")
        print(f"Confidence: {prob:.2%}")
        print(f"Fall Detected: {is_fall}")
    else:
        print(f"File not found: {audio_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        model_file = sys.argv[2] if len(sys.argv) > 2 else None
        quick_test(audio_file, model_file)
    else:
        print("Usage: python audio_inference.py <audio_file> [model_path]")
        print("\nInitializing detector for testing...")
        detector = AudioFallDetector()
        print(f"\nDetector info: {detector.get_info()}")# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import time
import os

# Import the NEW advanced detector
from pose_fall_detection import AdvancedPoseFallDetector as PoseFallDetector
from audio_inference import AudioFallDetector

# Can serve React frontend build OR run as API-only
app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize Detectors
# Ensure 'xgb_final_model.json' is in the same directory as app.py
pose_detector = PoseFallDetector(model_path='xgb_final_model.json')

# Attempt to load the trained audio model
try:
    if os.path.exists('./checkpoints/fold_1_best/model_state.pt'):
        audio_detector = AudioFallDetector(model_path='./checkpoints/fold_1_best/model_state.pt')
    else:
        print("Audio model checkpoint not found.")
        audio_detector = None
except Exception as e:
    print(f"Warning: Audio detector failed to initialize: {e}")
    audio_detector = None

@app.route('/reset', methods=['POST'])
def reset_detector():
    pose_detector.reset()
    return jsonify({'status': 'reset'})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    pose_detector.toggle_pause()
    return jsonify({'paused': pose_detector.paused})

@app.route('/detect_fall', methods=['POST'])
def detect_fall():
    """Video-based detection endpoint"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data'}), 400

        # Decode base64 image
        try:
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': 'Image decode failed'}), 400
        
        # Process frame
        annotated_frame, fallen, state = pose_detector.process_frame(frame)
        
        if annotated_frame is not None:
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'fall_detected': fallen,
                'state': state,
                'annotated_frame': f'data:image/jpeg;base64,{annotated_frame_base64}',
                'paused': pose_detector.paused
            })
        
        return jsonify({'paused': pose_detector.paused})
        
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_audio', methods=['POST'])
def detect_audio():
    """New Audio-based detection endpoint"""
    try:
        if audio_detector is None:
             return jsonify({'error': 'Audio detector not initialized'}), 503

        # Check if audio file is present in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio part'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read file into memory
        audio_bytes = io.BytesIO(audio_file.read())
        
        # Run inference
        is_fall, probability, label = audio_detector.detect(audio_bytes)
        
        return jsonify({
            'audio_fall_detected': is_fall,
            'confidence': float(probability),
            'label': label
        })

    except Exception as e:
        print(f"Audio Server Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_replay', methods=['GET'])
def get_replay():
    # Retrieve the last 15 seconds of frames
    try:
        raw_frames = pose_detector.get_replay_clip()
        encoded_frames = []
        
        for frame in raw_frames:
            if frame is not None and isinstance(frame, np.ndarray):
                # Frame is already resized in the detector for performance
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                encoded_frames.append(f'data:image/jpeg;base64,{frame_base64}')
            
        return jsonify({'frames': encoded_frames})
    except Exception as e:
        print(f"Replay Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure checkpoint directory exists
    import os
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    print("Starting Flask server on http://127.0.0.1:5000")
    # Using threaded=True for better responsiveness
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)