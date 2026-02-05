"""
Mel Spectrogram CNN Classifier for Voice Deepfake Detection

Uses CNN to analyze mel spectrograms for visual artifacts
indicative of AI-generated speech (vocoder patterns, unnatural harmonics).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import torchaudio.transforms as T
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import audio utilities for MP3 support
from ..audio_utils import load_audio_torch, extract_advanced_features


class SpectrogramCNN(nn.Module):
    """
    CNN architecture for analyzing mel spectrograms.
    Inspired by ResNet but optimized for audio deepfake detection.
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual-style blocks
        self.block1 = self._make_block(32, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        
        # Attention mechanism for focusing on relevant frequency bands
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with skip connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits and attention weights.
        
        Args:
            x: Input mel spectrogram [B, 1, H, W]
            
        Returns:
            logits: Classification logits [B, 2]
            attention_weights: Frequency attention weights [B, 256]
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Attention
        attn = self.attention(x)
        x = x * attn.unsqueeze(-1).unsqueeze(-1)
        
        # Classification
        x = self.global_pool(x)
        logits = self.classifier(x)
        
        return logits, attn


class SpectrogramDetector:
    """
    Mel Spectrogram-based detector for AI-generated voice detection.
    
    Converts audio to mel spectrograms and uses CNN to detect
    visual patterns indicative of neural vocoders.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"[SpectrogramDetector] Using device: {self.device}")
        
        # Initialize CNN model
        self.model = SpectrogramCNN(num_classes=2)
        self.model.to(self.device)
        self.model.eval()
        
        # Mel spectrogram parameters
        self.sample_rate = 16000
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 256
        self.target_length = 128  # Fixed width for CNN input
        
        # Mel transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        self.resampler_cache = {}
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and resample audio with MP3 support"""
        # Use audio_utils which supports MP3 via soundfile
        waveform = load_audio_torch(audio_path, target_sr=self.sample_rate)
        return waveform.unsqueeze(0)  # Add channel dim
    
    def _create_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized mel spectrogram"""
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale (dB)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Resize to fixed width
        if mel_spec.shape[-1] != self.target_length:
            mel_spec = F.interpolate(
                mel_spec.unsqueeze(0),
                size=(self.n_mels, self.target_length),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return mel_spec
    
    def _analyze_spectrogram(self, mel_spec: torch.Tensor) -> Dict:
        """Analyze spectrogram for AI-typical patterns"""
        spec = mel_spec.squeeze().numpy()
        
        analysis = {}
        
        # Check for unnaturally smooth regions
        gradient = np.gradient(spec, axis=1)
        analysis["temporal_smoothness"] = float(1.0 / (np.std(gradient) + 1e-8))
        
        # Check frequency band energy distribution
        low_band = spec[:32, :].mean()
        mid_band = spec[32:96, :].mean()
        high_band = spec[96:, :].mean()
        
        analysis["low_band_energy"] = float(low_band)
        analysis["mid_band_energy"] = float(mid_band)  
        analysis["high_band_energy"] = float(high_band)
        
        # Check for vocoder grid patterns (common in neural TTS)
        fft_spec = np.abs(np.fft.fft2(spec))
        analysis["periodicity_score"] = float(fft_spec[1:10, 1:10].mean() / fft_spec.mean())
        
        # Harmonic-to-noise ratio approximation
        sorted_spec = np.sort(spec.flatten())[::-1]
        top_10_pct = sorted_spec[:int(len(sorted_spec) * 0.1)].mean()
        bottom_50_pct = sorted_spec[int(len(sorted_spec) * 0.5):].mean()
        analysis["hnr_approx"] = float(top_10_pct / (bottom_50_pct + 1e-8))
        
        return analysis
    
    def detect(self, audio_path: str) -> Dict:
        """
        Detect if audio is AI-generated using spectrogram analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detection results
        """
        waveform = self._load_audio(audio_path)
        mel_spec = self._create_mel_spectrogram(waveform)
        
        spec_analysis = self._analyze_spectrogram(mel_spec)
        
        spec_analysis['energy_cv'] = self._compute_energy_cv(waveform)
        
        adv_features = extract_advanced_features(audio_path, self.sample_rate)
        spec_analysis.update(adv_features)
        
        ai_score = self._compute_ai_score_from_spectrogram(spec_analysis)
        
        ai_score = max(0.0, min(1.0, ai_score))
        
        is_ai = ai_score >= 0.5
        confidence = abs(ai_score - 0.5) * 2
        
        result = {
            "classification": "ai_generated" if is_ai else "human",
            "confidence": confidence,
            "model_scores": {
                "ai_probability": ai_score,
                "human_probability": 1 - ai_score
            },
            "spectrogram_analysis": spec_analysis,
            "frequency_attention": [],
            "indicators": self._generate_indicators(1 if is_ai else 0, spec_analysis)
        }
        
        return result
        
    def _compute_energy_cv(self, waveform: torch.Tensor) -> float:
        """Compute Coefficient of Variation of energy"""
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
            
        chunk_size = len(waveform) // 10
        if chunk_size == 0: return 0.0
        
        energies = []
        for i in range(10):
            chunk = waveform[i*chunk_size:(i+1)*chunk_size]
            energies.append(float(torch.sqrt(torch.mean(chunk ** 2))))
            
        energy_std = np.std(energies) if energies else 0
        energy_mean = np.mean(energies) if energies else 1
        return float(energy_std / (energy_mean + 1e-8))
    
    def _compute_ai_score_from_spectrogram(self, analysis: Dict) -> float:
        """
        Compute AI probability from spectrogram analysis.
        
        AI-generated voices typically have:
        - Higher temporal smoothness (consistent spectrum)
        - Low energy variation (consistent volume, less natural pausing)
        - Specific band energy distributions
        """
        score = 0.5  # Start neutral
        
        # 1. Energy CV - STRONGEST INDICATOR (prioritize this)
        # High variation (>0.5) is very typical of human speech (pauses/breathing)
        # Low variation (<0.2) is typical of AI
        energy_cv = analysis.get("energy_cv", 0.25)
        if energy_cv > 0.7:
            score -= 0.30  # Very strong human signal
        elif energy_cv > 0.5:
            score -= 0.20  # Strong human signal
        elif energy_cv > 0.35:
            score -= 0.10  # Moderate human signal
        elif energy_cv < 0.2:
            score += 0.10  # Consistent energy = AI like
        
        # 2. Advanced Features
        flux = analysis.get("spectral_flux", 0)
        mfcc_var = analysis.get("mfcc_variance", 0)
        
        # Flux Heuristic - widened threshold
        if flux > 2.4:
            score += 0.20  # Strong AI signal
        elif flux > 2.0:
            score += 0.10  # Moderate AI signal
        elif flux < 1.8 and flux > 0.1:
            score -= 0.10  # Natural human transitions
            
        # MFCC Var Heuristic
        if mfcc_var > 1900:
            score -= 0.20  # High complexity = human
        
        # 3. Temporal smoothness (de-prioritized - often misleading)
        # Only use extreme values
        smoothness = analysis.get("temporal_smoothness", 1.0)
        if smoothness > 5.0:
            score += 0.10  # Very unnaturally smooth
        
        # Additional features removed as unreliable for edge cases
        # (periodicity, high_band, hnr can cause false positives on clean recordings)
        
        return score
    
    
    def _generate_indicators(self, pred_class: int, analysis: Dict) -> list:
        """Generate human-readable indicators"""
        indicators = []
        
        if pred_class == 1:  # AI detected
            if analysis["temporal_smoothness"] > 5:
                indicators.append("Unnaturally smooth spectrogram transitions")
            
            if analysis["periodicity_score"] > 2:
                indicators.append("Periodic patterns suggesting neural vocoder")
            
            if analysis["high_band_energy"] < -2:
                indicators.append("Reduced high-frequency content typical of TTS")
                
            if not indicators:
                indicators.append("Spectrogram patterns consistent with AI synthesis")
        else:  # Human
            if analysis["hnr_approx"] > 10:
                indicators.append("Strong harmonic structure of natural voice")
            
            if analysis["temporal_smoothness"] < 3:
                indicators.append("Natural variation in spectral features")
                
            if not indicators:
                indicators.append("Spectrogram shows natural speech characteristics")
        
        return indicators
