"""
AI Voice Detection System Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
CACHE_DIR = BASE_DIR / ".cache"

# Create directories
WEIGHTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Model configurations
class ModelConfig:
    # IndicWav2Vec - Best for Indian languages
    INDICWAV2VEC_MODEL = "ai4bharat/indicwav2vec-hindi"
    
    # Multilingual Wav2Vec2 as fallback
    XLSR_MODEL = "facebook/wav2vec2-large-xlsr-53"
    
    # Language detection
    LANG_ID_MODEL = "speechbrain/lang-id-voxlingua107"
    
    # Audio settings
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 60  # seconds
    MIN_AUDIO_LENGTH = 1   # seconds
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.7
    ENSEMBLE_WEIGHTS = {
        "wav2vec": 0.5,
        "spectrogram_cnn": 0.3,
        "acoustic_rules": 0.2
    }

# Supported languages
SUPPORTED_LANGUAGES = {
    "ta": "Tamil",
    "en": "English", 
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu"
}

# AI Tools signatures for detection
AI_TOOL_SIGNATURES = {
    "nvidia_personaplex": {
        "description": "NVIDIA PersonaPlex/Riva TTS",
        "markers": ["hifi_gan_vocoder", "low_latency_synthesis"]
    },
    "elevenlabs": {
        "description": "ElevenLabs Voice Synthesis",
        "markers": ["multilingual_v2", "voice_cloning"]
    },
    "azure_tts": {
        "description": "Microsoft Azure TTS",
        "markers": ["neural_voice", "ssml_prosody"]
    },
    "google_tts": {
        "description": "Google Cloud TTS",
        "markers": ["wavenet", "neural2"]
    }
}
