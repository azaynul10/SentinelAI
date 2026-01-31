# audio_inference.py (Refactored for efficiency)
import torch
import numpy as np
import librosa
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import io

class AudioFallDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 16000
        
        # Load heavy components ONCE
        print("Loading AST Model...")
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        if model_path:
            self._load_weights(model_path)
        
        self.model.eval()

    def _load_weights(self, path):
        try:
            # Safe loading logic
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ Custom weights loaded: {path}")
        except Exception as e:
            print(f"⚠️ Could not load custom weights: {e}")

    def detect_memory(self, audio_data):
        """
        Accepts numpy array or bytes directly. NO DISK I/O.
        """
        try:
            # 1. Handle Input Types
            if isinstance(audio_data, bytes):
                # Bytes -> Audio Stream -> Numpy
                audio, _ = librosa.load(io.BytesIO(audio_data), sr=self.sr)
            elif isinstance(audio_data, np.ndarray):
                audio = audio_data
            else:
                return False, 0.0, "Invalid Input"

            # 2. Preprocess (Pad/Clip to 10s for AST standard)
            target_len = 16000 * 10 
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
            else:
                audio = audio[:target_len]

            # 3. Inference
            inputs = self.feature_extractor(
                audio, sampling_rate=self.sr, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                
            fall_prob = probs[0][1].item()
            return fall_prob > 0.5, fall_prob, "FALL" if fall_prob > 0.5 else "NORMAL"

        except Exception as e:
            print(f"Inference Error: {e}")
            return False, 0.0, "ERROR"