import requests
import json
import sys

API_URL = "http://localhost:8000/api/v1/detect"
SAMPLE_AUDIO_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav"  # Public domain sample

def test_url_detection():
    print(f"\nTesting URL Detection with: {SAMPLE_AUDIO_URL}")
    
    payload = {
        "audio_url": SAMPLE_AUDIO_URL,
        "language_hint": "en"
    }
    
    headers = {
        "X-API-Key": "hackathon_secret_key"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("✅ URL Detection Success!")
            result = response.json()
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("=== Verifying Hackathon Tester Configuration ===")
    test_url_detection()
