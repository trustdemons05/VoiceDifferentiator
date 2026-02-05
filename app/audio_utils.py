"""
Audio Utilities - Shared audio loading functions with MP3 support
"""
import os
from typing import Tuple
import numpy as np
import soundfile as sf
from scipy import signal
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file with MP3 support using soundfile.
    Returns: Tuple of (audio_array, sample_rate)
    """
    samples, sr = sf.read(audio_path, dtype='float32')
    
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    
    if sr != target_sr:
        samples = resample_audio(samples, sr, target_sr)
    
    return samples, target_sr


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy"""
    if orig_sr == target_sr:
        return samples
    
    duration = len(samples) / orig_sr
    new_length = int(duration * target_sr)
    
    resampled = signal.resample(samples, new_length)
    
    return resampled.astype(np.float32)


def load_audio_torch(audio_path: str, target_sr: int = 16000) -> "torch.Tensor":
    """Load audio and return as torch tensor"""
    samples, sr = load_audio(audio_path, target_sr)
    if TORCH_AVAILABLE:
        return torch.from_numpy(samples).float()
    else:
        raise ImportError("PyTorch is required for load_audio_torch")


def extract_advanced_features(audio_path: str, sample_rate: int = 16000) -> dict:
    """Extract advanced features using librosa (Flux, MFCC)"""
    import librosa
    try:
        # Load short segment for speed (max 10s)
        y, sr = librosa.load(audio_path, duration=10, sr=sample_rate)
        
        # Spectral Flux (Change in spectrum over time)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        flux = float(np.mean(onset_env))
        
        # MFCC Variance (Timbre complexity)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
        
        return {"spectral_flux": flux, "mfcc_variance": mfcc_var}
    except Exception as e:
        logger.error(f"Error extracting advanced features: {e}")
        return {"spectral_flux": 0.0, "mfcc_variance": 0.0}
