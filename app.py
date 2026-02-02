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
# [NEW] Multimodal Imports
from multimodal_detector import MultimodalFallDetector
from video_folder_processor import VideoFolderProcessor

# Can serve React frontend build OR run as API-only
app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- Initialization ---
print("Initializing Fall Detection System...")

# 1. Legacy/Single Detectors (kept for backward compatibility or individual testing)
pose_detector = PoseFallDetector(model_path='xgb_final_model.json')

# 2. New Multimodal Detector for Folder Testing
# Define alert callback
def backend_alert_callback(data):
    print(f"ALERT: {data['message']} (Score: {data['score']:.2f})")
    # In a real app, emit to a websocket here

mm_detector = MultimodalFallDetector(alert_callback=backend_alert_callback, pose_detector_instance=pose_detector)

# Factory for creating NEW instances in worker threads
def detector_factory():
    # Create fresh Pose Detector inside the thread
    pd = PoseFallDetector(model_path='xgb_final_model.json')
    # Create fresh Multimodal Detector
    return MultimodalFallDetector(alert_callback=None, pose_detector_instance=pd)

folder_processor = VideoFolderProcessor(detector_factory=detector_factory)

# --- Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'pose_detector': True,
        'multimodal_ready': True
    })

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
    """Video-based detection endpoint (Legacy wrapper)"""
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
        annotated_frame, fallen, state, method, score = pose_detector.process_frame(frame)
        
        if annotated_frame is not None:
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'fall_detected': fallen,
                'state': state,
                'detection_method': method,
                'confidence_score': float(score),
                'annotated_frame': f'data:image/jpeg;base64,{annotated_frame_base64}',
                'paused': pose_detector.paused
            })
        
        return jsonify({'paused': pose_detector.paused})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Server Error in /detect_fall: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_folder', methods=['POST'])
def process_folder():
    """Process all videos in a separate process for high accuracy"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        
        if not folder_path or not os.path.exists(folder_path):
             return jsonify({'error': 'Invalid path'}), 400

        # Run as independent subprocess to avoid blocking/GIL issues
        # and to ensure clean memory usage for heavy batch jobs.
        cmd = [
            sys.executable, "multimodal_batch_processor.py",
            "--folder", folder_path,
            "--output", "./results"
        ]
        
        subprocess.Popen(cmd) # Detached process
        
        return jsonify({
            'status': 'started', 
            'message': 'Batch processing started in background. Check ./results folder for output.',
            'job_type': 'multimodal_batch_accuracy'
        })
             
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process a single video file asynchronously"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        
        if not video_path:
             return jsonify({'error': 'Invalid video path'}), 400
             
        job_id = folder_processor.process_video_async(video_path)
        return jsonify({'job_id': job_id, 'status': 'started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Check batch processing status"""
    status = folder_processor.progress.get(job_id)
    if not status:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(status)

@app.route('/test_audio', methods=['POST'])
def test_audio():
    """Trigger the ElevenLabs audio alert for testing"""
    try:
        # Access the protected method directly for testing
        if hasattr(pose_detector, '_play_alert'):
            pose_detector._play_alert()
            return jsonify({'status': 'played'})
        return jsonify({'error': 'Audio alert not configured'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500



from flask import send_from_directory

@app.route('/request_instant_replay', methods=['POST'])
def request_instant_replay():
    """Returns URL to the most recent replay clip"""
    try:
        import glob
        replay_dir = './results/replays'
        os.makedirs(replay_dir, exist_ok=True)
        
        # Find the most recently created .mp4 file
        files = glob.glob(os.path.join(replay_dir, '*.mp4'))
        if not files:
            return jsonify({'error': 'No replays available'}), 404
        
        latest_file = max(files, key=os.path.getctime)
        filename = os.path.basename(latest_file)
        
        return jsonify({'url': f'/replays/{filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replays/<path:filename>')
def serve_replays(filename):
    return send_from_directory('./results/replays', filename)

if __name__ == '__main__':
    # Ensure checkpoint directories exist
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results/replays', exist_ok=True)
    
    print("Starting Flask server on http://127.0.0.1:5000")
    # THREADED=FALSE IS CRITICAL FOR MEDIAPIPE STABILITY ON WINDOWS
    # This ensures the global detector is always accessed by the MainThread
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=False)
