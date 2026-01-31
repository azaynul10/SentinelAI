import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import io
import os

# --- 1. Define the Exact Model Architecture used in Training ---
# We must use the exact same class definition so state_dict loads correctly.
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

# --- 2. Inference Wrapper Class ---
class AudioFallDetector:
    def __init__(self, model_path='checkpoints/fold_1_best/model_state.pt'):
        print("🎧 Initializing Audio Fall Detector (CPU Mode)...")
        self.device = torch.device('cpu') # Force CPU as requested
        self.sr = 16000
        self.target_length = int(4.0 * self.sr) # 4 seconds context
        
        # Initialize Feature Extractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        # Initialize Model
        self.model = FixedASTModel()
        
        # Load Weights
        if os.path.exists(model_path):
            try:
                # Load state dict with map_location='cpu'
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different saving formats from your script
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load weights
                self.model.load_state_dict(state_dict)
                print(f"✅ Audio Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load audio weights: {e}")
                print("⚠️ Running with un-finetuned weights (for testing only)")
        else:
            print(f"⚠️ Model path not found: {model_path}")
            print("⚠️ Running with un-finetuned weights")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_audio(self, audio_data):
        """
        Convert raw bytes or numpy array to model input
        """
        try:
            # If input is bytes/buffer, load it
            if isinstance(audio_data, (bytes, bytearray, io.BytesIO)):
                # Load with librosa (this handles many formats)
                audio, _ = librosa.load(audio_data, sr=self.sr)
            else:
                audio = audio_data

            # Ensure length (pad or truncate to 4s)
            if len(audio) > self.target_length:
                # Take the center for inference
                start = (len(audio) - self.target_length) // 2
                audio = audio[start:start + self.target_length]
            else:
                pad_length = self.target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')

            # Extract features
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=self.sr, 
                return_tensors="pt"
            )
            return inputs['input_values'].to(self.device)
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def detect(self, audio_source):
        """
        Main inference method.
        Returns: (is_fall (bool), confidence (float), label (str))
        """
        try:
            input_values = self.preprocess_audio(audio_source)
            if input_values is None:
                return False, 0.0, "Error"

            with torch.no_grad():
                outputs = self.model(input_values)
                probs = torch.softmax(outputs.logits, dim=1)
                
                # Get fall probability (index 1 based on your training script)
                fall_prob = probs[0][1].item()
                prediction = torch.argmax(probs, dim=1).item()
                
                is_fall = prediction == 1
                label = "FALL" if is_fall else "Background"
                
                return is_fall, fall_prob, label

        except Exception as e:
            print(f"Inference error: {e}")
            return False, 0.0, "Error"