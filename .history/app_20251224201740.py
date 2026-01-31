# app.py
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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize Detectors
# Ensure 'xgb_final_model.json' is in the root directory or update path
pose_detector = PoseFallDetector(model_path='xgb_final_model.json')

# Attempt to load the trained audio model (SKIP for faster startup)
# Set to True to enable audio detection
ENABLE_AUDIO = False

audio_detector = None
if ENABLE_AUDIO:
    try:
        if os.path.exists('./checkpoints/fold_1_best/model_state.pt'):
            audio_detector = AudioFallDetector(model_path='./checkpoints/fold_1_best/model_state.pt')
        else:
            print("Audio model checkpoint not found.")
            audio_detector = None
    except Exception as e:
        print(f"Warning: Audio detector failed to initialize: {e}")
        audio_detector = None
else:
    print("[Audio] Audio detector disabled for faster startup")

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
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)