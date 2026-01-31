import os
import sys
import time
from multimodal_detector import MultimodalFallDetector

def test_callback(data):
    print(f"\n[TEST CALLBACK] Fall Detected! Score: {data['score']:.2f}")
    print(f"Timestamp: {data['timestamp']:.2f}s")
    print("-" * 20)

def main():
    print("--- Starting Multimodal System Verification ---")
    
    # 1. Initialize Detector
    detector = MultimodalFallDetector(alert_callback=test_callback)
    detector.start_threads()
    
    # 2. Check for Test Videos
    # Use relative path to ensure we are in the workspace
    test_folder = os.path.abspath("./test_videos")
    if not os.path.exists(test_folder):
        print(f"Creating test folder: {test_folder}")
        os.makedirs(test_folder, exist_ok=True)
        print("Please put some test videos in this folder and run again.")
        detector.stop_threads()
        return

    # 3. Import Processor (we use the class directly to control loop easier)
    from video_folder_processor import VideoFolderProcessor
    processor = VideoFolderProcessor(detector)
    
    # 4. Run Batch
    print(f"Processing videos in: {test_folder}")
    job_id = processor.process_folder_async(test_folder)
    
    # 5. Monitor
    while True:
        status = processor.progress.get(job_id)
        if status:
            print(f"Status: {status['status']} | {status.get('current',0)}/{status.get('total',0)} | {status.get('current_file','')}", end='\r')
            if status['status'] == 'completed':
                break
        time.sleep(1)
        
    print("\n--- Processing Complete ---")
    print(f"Check results in ./results/report_{job_id}.json")

if __name__ == "__main__":
    main()
