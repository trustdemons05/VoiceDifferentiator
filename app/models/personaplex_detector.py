"""
NVIDIA PersonaPlex and AI Tool Signature Detection

Detects specific AI voice synthesis tools by analyzing their unique
acoustic signatures and vocoder fingerprints.
"""
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.fft import fft, fftfreq

# Import audio utilities for MP3 support
from ..audio_utils import load_audio


class PersonaPlexDetector:
    """
    Specialized detector for NVIDIA PersonaPlex and other AI voice synthesis tools.
    
    NVIDIA PersonaPlex uses HiFi-GAN vocoder which has characteristic patterns:
    - Specific frequency artifacts in 6-8kHz range
    - Phase correlation patterns from real-time synthesis
    - Low-latency inference artifacts
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.resampler_cache = {}
        
        # Known AI tool signatures
        self.signatures = {
            "nvidia_personaplex": {
                "name": "NVIDIA PersonaPlex/Riva",
                "vocoder": "HiFi-GAN",
                "frequency_artifacts": (6000, 8000),  # Hz range
                "phase_coherence_threshold": 0.85,
                "spectral_tilt_range": (-2, 0),
            },
            "elevenlabs": {
                "name": "ElevenLabs",
                "vocoder": "Proprietary Neural",
                "frequency_artifacts": (7000, 10000),
                "phase_coherence_threshold": 0.80,
                "spectral_tilt_range": (-3, -1),
            },
            "azure_neural": {
                "name": "Microsoft Azure Neural TTS",
                "vocoder": "Neural Vocoder",
                "frequency_artifacts": (5000, 7000),
                "phase_coherence_threshold": 0.82,
                "spectral_tilt_range": (-2.5, -0.5),
            },
            "google_wavenet": {
                "name": "Google WaveNet/Neural2",
                "vocoder": "WaveNet",
                "frequency_artifacts": (6500, 9000),
                "phase_coherence_threshold": 0.78,
                "spectral_tilt_range": (-2, 0.5),
            }
        }
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file and return numpy array with MP3 support"""
        # Use audio_utils which supports MP3 via soundfile
        samples, sr = load_audio(audio_path, target_sr=self.sample_rate)
        return samples
    
    def _compute_stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Short-Time Fourier Transform"""
        nperseg = 1024
        noverlap = nperseg // 2
        
        frequencies, times, Zxx = signal.stft(
            audio, 
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        return frequencies, times, Zxx
    
    def _analyze_frequency_artifacts(self, frequencies: np.ndarray, Zxx: np.ndarray) -> Dict:
        """Analyze frequency-domain artifacts typical of neural vocoders"""
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        analysis = {}
        
        # Analyze different frequency bands
        for band_name, (low_hz, high_hz) in [
            ("low", (0, 2000)),
            ("mid", (2000, 5000)),
            ("high", (5000, 8000)),
            ("very_high", (8000, self.sample_rate // 2))
        ]:
            mask = (frequencies >= low_hz) & (frequencies < high_hz)
            if mask.any():
                band_mag = magnitude[mask, :]
                analysis[f"{band_name}_band_energy"] = float(np.mean(band_mag))
                analysis[f"{band_name}_band_std"] = float(np.std(band_mag))
        
        # Check for vocoder-specific artifacts in 6-8kHz range
        vocoder_mask = (frequencies >= 6000) & (frequencies <= 8000)
        if vocoder_mask.any():
            vocoder_band = magnitude[vocoder_mask, :]
            
            # Neural vocoders often show unnaturally consistent energy in this band
            analysis["vocoder_band_consistency"] = float(
                1.0 - (np.std(vocoder_band.mean(axis=0)) / (np.mean(vocoder_band) + 1e-8))
            )
            
            # Check for periodic patterns (grid artifacts)
            fft_of_band = np.abs(fft(vocoder_band.mean(axis=0)))
            analysis["vocoder_periodicity"] = float(
                np.max(fft_of_band[1:20]) / (np.mean(fft_of_band) + 1e-8)
            )
        else:
            analysis["vocoder_band_consistency"] = 0.0
            analysis["vocoder_periodicity"] = 0.0
        
        return analysis
    
    def _analyze_phase_coherence(self, Zxx: np.ndarray) -> Dict:
        """
        Analyze phase coherence patterns.
        AI-generated audio often shows higher phase coherence due to deterministic synthesis.
        """
        phase = np.angle(Zxx)
        
        # Compute phase difference between adjacent frames
        phase_diff = np.diff(phase, axis=1)
        
        # Unwrap phase differences
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        
        analysis = {
            "phase_coherence": float(1.0 - np.std(phase_diff)),
            "phase_consistency": float(np.mean(np.abs(phase_diff) < 0.5)),
            "phase_periodicity": float(
                np.corrcoef(phase_diff[:, :-1].flatten(), phase_diff[:, 1:].flatten())[0, 1]
                if phase_diff.shape[1] > 1 else 0
            )
        }
        
        return analysis
    
    def _compute_spectral_tilt(self, frequencies: np.ndarray, magnitude: np.ndarray) -> float:
        """
        Compute spectral tilt (slope of spectral envelope).
        AI voices often have different spectral tilt than natural voices.
        """
        # Use frequencies up to 8kHz
        mask = frequencies <= 8000
        freqs = frequencies[mask]
        mags = magnitude[mask, :].mean(axis=1)
        
        if len(freqs) < 2:
            return 0.0
        
        # Log-log regression for spectral tilt
        log_freqs = np.log(freqs + 1)
        log_mags = np.log(mags + 1e-8)
        
        # Simple linear regression
        slope = np.polyfit(log_freqs, log_mags, 1)[0]
        
        return float(slope)
    
    def _match_signatures(self, freq_analysis: Dict, phase_analysis: Dict, spectral_tilt: float) -> List[Dict]:
        """Match analysis results against known AI tool signatures"""
        matches = []
        
        for tool_id, sig in self.signatures.items():
            score = 0.0
            reasons = []
            
            # Check phase coherence
            if phase_analysis["phase_coherence"] >= sig["phase_coherence_threshold"]:
                score += 0.3
                reasons.append(f"Phase coherence matches {sig['vocoder']} pattern")
            
            # Check spectral tilt
            tilt_low, tilt_high = sig["spectral_tilt_range"]
            if tilt_low <= spectral_tilt <= tilt_high:
                score += 0.25
                reasons.append("Spectral tilt consistent with synthesis")
            
            # Check vocoder band artifacts
            if freq_analysis["vocoder_band_consistency"] > 0.7:
                score += 0.25
                reasons.append("Vocoder frequency artifacts detected")
            
            # Check periodicity (neural vocoder grid)
            if freq_analysis["vocoder_periodicity"] > 1.5:
                score += 0.2
                reasons.append("Periodic synthesis patterns found")
            
            if score > 0.4:  # Threshold for reporting
                matches.append({
                    "tool_id": tool_id,
                    "tool_name": sig["name"],
                    "confidence": score,
                    "reasons": reasons
                })
        
        # Sort by confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        
        return matches
    
    def detect(self, audio_path: str) -> Dict:
        """
        Detect AI voice synthesis tool signatures.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detection results including matched tools
        """
        audio = self._load_audio(audio_path)
        
        frequencies, times, Zxx = self._compute_stft(audio)
        magnitude = np.abs(Zxx)
        
        freq_analysis = self._analyze_frequency_artifacts(frequencies, Zxx)
        
        phase_analysis = self._analyze_phase_coherence(Zxx)
        
        spectral_tilt = self._compute_spectral_tilt(frequencies, magnitude)
        
        matched_tools = self._match_signatures(freq_analysis, phase_analysis, spectral_tilt)
        
        is_ai = len(matched_tools) > 0 and matched_tools[0]["confidence"] > 0.5
        
        ai_probability = max([m["confidence"] for m in matched_tools]) if matched_tools else 0.2
        
        result = {
            "classification": "ai_generated" if is_ai else "human",
            "confidence": ai_probability if is_ai else (1 - ai_probability),
            "model_scores": {
                "ai_probability": ai_probability,
                "human_probability": 1 - ai_probability
            },
            "detected_tools": matched_tools[:3] if matched_tools else [],
            "primary_tool": matched_tools[0]["tool_name"] if matched_tools else None,
            "frequency_analysis": freq_analysis,
            "phase_analysis": phase_analysis,
            "spectral_tilt": spectral_tilt,
            "indicators": self._generate_indicators(is_ai, matched_tools, freq_analysis, phase_analysis)
        }
        
        return result
    
    def _generate_indicators(
        self, 
        is_ai: bool, 
        matched_tools: List[Dict],
        freq_analysis: Dict,
        phase_analysis: Dict
    ) -> List[str]:
        """Generate human-readable indicators"""
        indicators = []
        
        if is_ai and matched_tools:
            top_match = matched_tools[0]
            indicators.append(f"Signature matches {top_match['tool_name']} ({top_match['confidence']:.0%} confidence)")
            indicators.extend(top_match["reasons"][:2])
        elif is_ai:
            if phase_analysis["phase_coherence"] > 0.7:
                indicators.append("Unusually high phase coherence suggesting deterministic synthesis")
            if freq_analysis["vocoder_band_consistency"] > 0.6:
                indicators.append("Vocoder artifacts detected in 6-8kHz band")
        else:
            indicators.append("No known AI synthesis tool signatures detected")
            if phase_analysis["phase_coherence"] < 0.6:
                indicators.append("Phase patterns consistent with natural recording")
        
        return indicators
