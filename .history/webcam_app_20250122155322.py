import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# -----------------------------------------------------------------
# 1) Import or copy over the same Pose initialization & functions
#    from your original code. You can copy/paste them here
#    (process_frame, show_previous_5_seconds, etc.).
# -----------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# Exact copy of your "process_frame" from the other script:
def process_frame(frame, timestamp, fps):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    annotated_frame = frame.copy()
    
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    # Add timer text
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp % 1) * 100)
    timer_text = f'{minutes:02d}:{seconds:02d}.{milliseconds:02d}'
    cv2.putText(annotated_frame, timer_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated_frame


def show_previous_5_seconds(frame_buffer, fps):
    if not frame_buffer:
        return
    
    for frame, _ in list(frame_buffer):
        cv2.imshow('Previous 5 Seconds', frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Previous 5 Seconds')


# -----------------------------------------------------------------
# 2) The only difference here is that video_path = 0 (webcam)
# -----------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

fps = 30  # approximate webcam FPS if needed
from collections import deque
frame_buffer_size = 5 * fps  # 5 seconds
frame_buffer = deque(maxlen=frame_buffer_size)

paused = False
frame_count = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam.")
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        annotated_frame = process_frame(frame, current_time, fps)
        frame_buffer.append((annotated_frame, current_time))
        
        cv2.imshow('Webcam', annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        if paused:
            show_previous_5_seconds(frame_buffer, fps)

cap.release()
cv2.destroyAllWindows()
pose.close()