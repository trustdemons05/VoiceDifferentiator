"""
Models Package - AI Voice Detection Models
"""
from .wav2vec_detector import Wav2VecDetector, IndicWav2VecDetector
from .spectrogram_cnn import SpectrogramDetector, SpectrogramCNN
from .personaplex_detector import PersonaPlexDetector
from .ensemble_detector import EnsembleVoiceDetector, create_detector

__all__ = [
    "Wav2VecDetector",
    "IndicWav2VecDetector", 
    "SpectrogramDetector",
    "SpectrogramCNN",
    "PersonaPlexDetector",
    "EnsembleVoiceDetector",
    "create_detector"
]
