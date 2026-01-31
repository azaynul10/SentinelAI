# app.py - Fall Detection API Server
# Can serve React frontend build OR run as API-only

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import time
import tempfile
import os
from pose_fall_detection import PoseFallDetector
from audio_inference import AudioFallDetector

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# IoT Configuration
IOT_CONFIG = {
    'enabled': os.environ.get('IOT_ENABLED', 'false').lower() == 'true',
    'webhook_url': os.environ.get('IOT_WEBHOOK_URL', None),
    'mqtt_broker': os.environ.get('IOT_MQTT_BROKER', None),
    'device_id': os.environ.get('IOT_DEVICE_ID', 'fall_detector_001')
}

# Initialize Detectors
pose_detector = PoseFallDetector(
    buffer_seconds=10,
    target_fps=15,
    iot_config=IOT_CONFIG
)

# Audio detector (optional)
try:
    audio_detector = AudioFallDetector(model_path='./checkpoints/fold_1_best/model_state.pt')
except Exception as e:
    print(f"Warning: Audio detector failed to initialize: {e}")
    audio_detector = None


# ==================== FRONTEND SERVING ====================

@app.route('/')
def serve_frontend():
    """Serve React frontend or show API info"""
    # Check if React build exists
    if os.path.exists(os.path.join(app.static_folder or '', 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    
    # Otherwise show API documentation
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Fall Detection API</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #1a1a2e; }
        .card { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .endpoint { background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; font-family: monospace; }
        .method { background: #2563eb; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }
        .method.get { background: #22c55e; }
        a { color: #2563eb; }
        .warning { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🎯 Fall Detection API Server</h1>
    
    <div class="warning">
        <strong>Note:</strong> This is the API server. The React frontend runs separately on port 3000.
        <br><br>
        <strong>To use:</strong>
        <ol>
            <li>Keep this server running (port 5000)</li>
            <li>In another terminal, run: <code>cd frontend && npm start</code></li>
            <li>Open <a href="http://localhost:3000">http://localhost:3000</a></li>
        </ol>
        <br>
        <strong>Or for production:</strong> Build React app and place in <code>./build</code> folder.
    </div>
    
    <div class="card">
        <h2>API Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> /health - Health check
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /detect_fall - Single frame detection (JSON: {frame: base64})
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /detect_video - Full video file detection (multipart: video file)
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /detect_audio - Audio-based detection (multipart: audio file)
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> /get_replay - Get replay video/frames
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /reset - Reset detector state
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> /iot/alerts - Get IoT alert history
        </div>
    </div>
    
    <div class="card">
        <h2>Quick Test</h2>
        <p>Test the API with curl:</p>
        <pre style="background: #1a1a2e; color: #fff; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl http://localhost:5000/health

curl -X POST http://localhost:5000/reset

curl -X POST -F "video=@test_video.mp4" http://localhost:5000/detect_video
        </pre>
    </div>
    
    <div class="card">
        <h2>Status</h2>
        <p>✅ Pose Detector: Ready</p>
        <p>''' + ('✅' if audio_detector else '❌') + ''' Audio Detector: ''' + ('Ready' if audio_detector else 'Not loaded') + '''</p>
        <p>''' + ('✅' if IOT_CONFIG['enabled'] else '⚪') + ''' IoT Alerts: ''' + ('Enabled' if IOT_CONFIG['enabled'] else 'Disabled') + '''</p>
    </div>
</body>
</html>
'''


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from React build"""
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return serve_frontend()


# ==================== HEALTH & STATUS ====================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'audio_detector': audio_detector is not None,
        'pose_detector': True,
        'iot_enabled': IOT_CONFIG['enabled'],
        'detector_stats': pose_detector.get_stats()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(pose_detector.get_stats())


# ==================== POSE DETECTION ====================

@app.route('/reset', methods=['POST'])
def reset_detector():
    pose_detector.reset()
    return jsonify({'status': 'reset', 'state': pose_detector.state})


@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    pose_detector.toggle_pause()
    return jsonify({'paused': pose_detector.paused})


@app.route('/detect_fall', methods=['POST'])
def detect_fall():
    """Single frame detection"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data'}), 400

        # Decode base64
        try:
            if ',' in frame_data:
                img_bytes = base64.b64decode(frame_data.split(',')[1])
            else:
                img_bytes = base64.b64decode(frame_data)
            
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Failed to decode image'}), 400
                
        except Exception as e:
            return jsonify({'error': f'Image decode failed: {str(e)}'}), 400
        
        # Process frame
        annotated_frame, fallen, state = pose_detector.process_frame(frame)
        
        if annotated_frame is not None:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'fall_detected': fallen,
                'state': state,
                'annotated_frame': f'data:image/jpeg;base64,{annotated_frame_base64}',
                'paused': pose_detector.paused,
                'detection_method': pose_detector.detection_method
            })
        
        return jsonify({
            'fall_detected': False,
            'state': state,
            'paused': pose_detector.paused
        })
        
    except Exception as e:
        print(f"Server Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/detect_video', methods=['POST'])
def detect_video():
    """Process entire video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            pose_detector.reset()
            
            fall_detected = False
            fall_frame = -1
            fall_timestamp = -1.0
            states_seen = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                _, fallen, state = pose_detector.process_frame(frame)
                states_seen.append(state)
                
                if fallen and not fall_detected:
                    fall_detected = True
                    fall_frame = frame_idx
                    fall_timestamp = frame_idx / fps
                
                frame_idx += 1
            
            cap.release()
            
            # Count states
            from collections import Counter
            state_counts = dict(Counter(states_seen))
            
            return jsonify({
                'fall_detected': fall_detected,
                'fall_frame': fall_frame,
                'fall_timestamp': fall_timestamp,
                'total_frames': total_frames,
                'fps': fps,
                'duration': total_frames / fps if fps > 0 else 0,
                'final_state': pose_detector.state,
                'states_distribution': state_counts
            })
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"Video Detection Error: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== REPLAY ====================

@app.route('/get_replay', methods=['GET'])
def get_replay():
    """Get replay as video, gif, or frames"""
    try:
        raw_frames = pose_detector.get_replay_clip()
        
        if not raw_frames:
            return jsonify({'type': 'empty', 'error': 'No frames in buffer', 'frame_count': 0})
        
        valid_frames = [f for f in raw_frames if f is not None and isinstance(f, np.ndarray) and f.size > 0]
        
        if not valid_frames:
            return jsonify({'type': 'empty', 'error': 'No valid frames', 'frame_count': 0})
        
        print(f"[Replay] Processing {len(valid_frames)} frames...")
        
        # Try video first
        video_result = _create_replay_video(valid_frames)
        if video_result:
            video_base64 = base64.b64encode(video_result).decode('utf-8')
            return jsonify({
                'type': 'video',
                'video': f'data:video/mp4;base64,{video_base64}',
                'frame_count': len(valid_frames),
                'duration': len(valid_frames) / 15.0
            })
        
        # Fallback to frames
        return _get_replay_frames_response(valid_frames, max_frames=30)
                
    except Exception as e:
        print(f"Replay Error: {e}")
        return jsonify({'type': 'error', 'error': str(e)}), 500


def _create_replay_video(frames, fps=15):
    """Create MP4 video from frames"""
    if not frames:
        return None
    
    h, w = frames[0].shape[:2]
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
        out = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
                if out.isOpened():
                    break
                out.release()
                out = None
            except:
                continue
        
        if out is None or not out.isOpened():
            return None
        
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
        
        out.release()
        
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, 'rb') as f:
                return f.read()
        return None
        
    except Exception as e:
        print(f"[Replay] Video error: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def _get_replay_frames_response(frames, max_frames=30):
    """Return frames as base64 images"""
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
    
    encoded_frames = []
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    
    for frame in frames:
        try:
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(f'data:image/jpeg;base64,{frame_base64}')
        except:
            continue
    
    return jsonify({
        'type': 'frames',
        'frames': encoded_frames,
        'frame_count': len(encoded_frames)
    })


@app.route('/get_replay_frames', methods=['GET'])
def get_replay_frames():
    """Get replay as frames only"""
    try:
        limit = request.args.get('limit', 30, type=int)
        limit = min(max(limit, 1), 60)
        
        raw_frames = pose_detector.get_replay_clip()
        valid_frames = [f for f in raw_frames if f is not None and isinstance(f, np.ndarray)]
        
        return _get_replay_frames_response(valid_frames, max_frames=limit)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== AUDIO DETECTION ====================

@app.route('/detect_audio', methods=['POST'])
def detect_audio():
    """Audio-based detection"""
    try:
        if audio_detector is None:
            return jsonify({'error': 'Audio detector not initialized'}), 503

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        audio_bytes = io.BytesIO(audio_file.read())
        is_fall, probability, label = audio_detector.detect(audio_bytes)
        
        return jsonify({
            'audio_fall_detected': is_fall,
            'confidence': float(probability),
            'label': label
        })

    except Exception as e:
        print(f"Audio Error: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== IoT ====================

@app.route('/iot/config', methods=['GET', 'POST'])
def iot_config():
    global IOT_CONFIG
    
    if request.method == 'POST':
        data = request.get_json()
        if data:
            IOT_CONFIG.update(data)
            pose_detector.iot_manager.config = IOT_CONFIG
            return jsonify({'status': 'updated', 'config': IOT_CONFIG})
    
    return jsonify(IOT_CONFIG)


@app.route('/iot/alerts', methods=['GET'])
def get_iot_alerts():
    return jsonify({
        'alerts': pose_detector.get_iot_alerts(),
        'total': len(pose_detector.iot_manager.alert_history)
    })


@app.route('/iot/test_alert', methods=['POST'])
def test_iot_alert():
    result = pose_detector.iot_manager.send_alert({
        'type': 'TEST',
        'message': 'Test alert from Fall Detection System'
    })
    return jsonify({'sent': result})


# ==================== STARTUP ====================

if __name__ == '__main__':
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    print("=" * 60)
    print("  Fall Detection Server - Hybrid Detection")
    print("=" * 60)
    print(f"  Pose Detector: Ready (with contour fallback)")
    print(f"  Audio Detector: {'Ready' if audio_detector else 'Not available'}")
    print(f"  IoT Alerts: {'Enabled' if IOT_CONFIG['enabled'] else 'Disabled'}")
    print("=" * 60)
    print()
    print("  API Server:    http://127.0.0.1:5000")
    print("  React Frontend: http://127.0.0.1:3000 (run 'npm start' separately)")
    print()
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)