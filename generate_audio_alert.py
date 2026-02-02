import requests
import os
import subprocess

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not API_KEY:
    print("Error: ELEVENLABS_API_KEY not found in environment variables.")
    exit(1)
VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Rachel

def generate_audio():
    print("Requesting audio from ElevenLabs...")
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }

    data = {
        "text": "I have detected a fall. Don't worry, I am alerting your emergency contact now.",
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}", json=data, headers=headers)
    
    if response.status_code == 200:
        mp3_path = "fall_alert.mp3"
        wav_path = "fall_alert.wav"
        
        with open(mp3_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to {mp3_path}")
        
        # Convert to WAV for robust winsound playback
        print("Converting to WAV...")
        try:
            # Assumes ffmpeg is in path (verified by previous multimodal work)
            subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Success: {wav_path} created.")
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}")
            
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    generate_audio()
