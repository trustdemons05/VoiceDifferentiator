"""
Test script for AI Voice Detection API

Usage:
    python test_api.py <path_to_audio.mp3>
    python test_api.py --generate-sample  # Generate test sample
"""
import base64
import requests
import json
import sys
import os


API_URL = "http://localhost:8001"


def encode_audio(file_path: str) -> str:
    """Encode audio file to base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_detection(audio_path: str, language: str = "en"):
    """Test the detection endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing AI Voice Detection")
    print(f"{'='*60}")
    print(f"Audio file: {audio_path}")
    print(f"Language hint: {language}")
    
    # Encode audio
    print("\n[1/3] Encoding audio to Base64...")
    audio_base64 = encode_audio(audio_path)
    print(f"      Encoded size: {len(audio_base64)} characters")
    
    # Prepare request
    payload = {
        "audio_base64": audio_base64,
        "language_hint": language
    }
    
    # Send request
    print("\n[2/3] Sending request to API...")
    try:
        response = requests.post(
            f"{API_URL}/api/v1/detect",
            json=payload,
            timeout=120  # Detection can take time
        )
        
        process_time = response.headers.get('X-Process-Time', 'N/A')
        print(f"      Response time: {process_time}s")
        print(f"      Status code: {response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("   Make sure the server is running: python run.py --reload")
        return
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Parse response
    print("\n[3/3] Detection Result:")
    print(f"{'='*60}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Classification
        classification = result['classification']
        confidence = result['confidence']
        emoji = "🤖" if classification == "ai_generated" else "👤"
        
        print(f"\nClassification: {classification.upper()}")
        print(f"Confidence: {confidence:.1%}")
        
        # Explanation
        explanation = result['explanation']
        print("\nExplanation:")
        for indicator in explanation.get('key_indicators', []):
            print(f"   • {indicator}")
        
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.json())


def test_health():
    """Test the health endpoint"""
    print("\n[Health Check]")
    try:
        response = requests.get(f"{API_URL}/api/v1/health")
        print(f"Status: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def create_test_sample():
    """Create a simple test audio sample using basic sine waves"""
    import numpy as np
    from scipy.io import wavfile
    
    print("\n[Creating test audio sample]")
    
    # Generate 3 seconds of audio
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Simple speech-like signal (not real speech, just for testing)
    signal = np.sin(2 * np.pi * 200 * t)  # Fundamental
    signal += 0.5 * np.sin(2 * np.pi * 400 * t)  # Harmonic
    signal += 0.3 * np.sin(2 * np.pi * 600 * t)  # Harmonic
    signal += 0.1 * np.random.randn(len(t))  # Noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    signal = (signal * 32767).astype(np.int16)
    
    # Save as WAV (API will handle conversion)
    output_path = "test_sample.wav"
    wavfile.write(output_path, sample_rate, signal)
    print(f"Created: {output_path}")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <audio_file.mp3>")
        print("       python test_api.py --health")
        print("       python test_api.py --generate-sample")
        sys.exit(1)
    
    if sys.argv[1] == "--health":
        test_health()
    elif sys.argv[1] == "--generate-sample":
        sample_path = create_test_sample()
        test_detection(sample_path)
    else:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"Error: File not found: {audio_path}")
            sys.exit(1)
        
        language = sys.argv[2] if len(sys.argv) > 2 else "en"
        test_detection(audio_path, language)
