# audio_inference.py (Refactored for efficiency - No disk I/O)
import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import io
import os


# --- 1. Define the Exact Model Architecture used in Training ---
class FixedASTModel(nn.Module):
    """
    Same architecture as research_grade_ast_FIXED.py
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        
        # Load pre-trained AST configuration
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Same classifier structure as your training script
        hidden_size = self.ast.config.hidden_size
        
        self.enhanced_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        
        # Replace original classifier
        self.ast.classifier = self.enhanced_classifier

    def forward(self, input_values):
        return self.ast(input_values=input_values)


# --- 2. Efficient Inference Wrapper Class ---
class AudioFallDetector:
    """
    Memory-efficient audio fall detector. 
    - Loads model ONCE at init
    - Accepts bytes/numpy directly (NO disk I/O)
    - Thread-safe inference with torch.no_grad()
    """
    
    def __init__(self, model_path='checkpoints/fold_1_best/model_state.pt'):
        print("🎧 Initializing Audio Fall Detector...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device}")
        
        self.sr = 16000
        self.target_length = int(4.0 * self.sr)  # 4 seconds context (64000 samples)
        
        # Load heavy components ONCE
        print("   Loading AST Feature Extractor...")
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        print("   Loading AST Model...")
        self.model = FixedASTModel()
        
        # Load custom weights if available
        self._load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Audio Fall Detector ready!")

    def _load_weights(self, model_path):
        """Safe weight loading with multiple format support"""
        if not model_path or not os.path.exists(model_path):
            print(f"⚠️ Model path not found: {model_path}")
            print("   Running with un-finetuned weights (for testing only)")
            return
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ Custom weights loaded: {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load custom weights: {e}")
            print("   Running with un-finetuned weights")

    def _preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess numpy audio array to model input tensor.
        Pads or clips to target_length (4s).
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Pad or clip to target length
        if len(audio) > self.target_length:
            # Take center crop for best context
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        elif len(audio) < self.target_length:
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
        """Run inference and return (is_fall, probability, label)"""
        with torch.no_grad():
            outputs = self.model(input_values)
            probs = torch.softmax(outputs.logits, dim=1)
            
            fall_prob = probs[0][1].item()
            is_fall = fall_prob > 0.5
            label = "FALL" if is_fall else "NORMAL"
            
            return is_fall, fall_prob, label

    def detect_memory(self, audio_data) -> tuple:
        """
        Memory-efficient detection. Accepts:
        - bytes: Raw audio file bytes
        - io.BytesIO: Audio buffer
        - np.ndarray: Pre-loaded audio samples
        
        Returns: (is_fall: bool, confidence: float, label: str)
        NO DISK I/O - everything in memory.
        """
        try:
            # 1. Convert input to numpy array
            if isinstance(audio_data, (bytes, bytearray)):
                audio, _ = librosa.load(io.BytesIO(audio_data), sr=self.sr)
            elif isinstance(audio_data, io.BytesIO):
                audio_data.seek(0)  # Reset buffer position
                audio, _ = librosa.load(audio_data, sr=self.sr)
            elif isinstance(audio_data, np.ndarray):
                audio = audio_data.astype(np.float32)
                # Resample if needed (assume input is at self.sr if ndarray)
            else:
                return False, 0.0, "Invalid Input Type"

            # 2. Preprocess
            input_values = self._preprocess(audio)
            
            # 3. Inference
            return self._inference(input_values)

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return False, 0.0, "ERROR"

    def detect(self, audio_source) -> tuple:
        """
        Backward-compatible detection method.
        Accepts file-like objects (BytesIO from Flask request.files).
        
        Returns: (is_fall: bool, confidence: float, label: str)
        """
        try:
            # Handle Flask file upload (BytesIO)
            if hasattr(audio_source, 'read'):
                audio_bytes = audio_source.read()
                if hasattr(audio_source, 'seek'):
                    audio_source.seek(0)  # Reset for potential re-reads
                return self.detect_memory(audio_bytes)
            else:
                return self.detect_memory(audio_source)
                
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return False, 0.0, "Error"

    def detect_file(self, file_path: str) -> tuple:
        """
        Convenience method for file-based detection (testing only).
        For production, prefer detect_memory() to avoid disk I/O.
        """
        try:
            audio, _ = librosa.load(file_path, sr=self.sr)
            return self.detect_memory(audio)
        except Exception as e:
            print(f"❌ File load error: {e}")
            return False, 0.0, "File Error"