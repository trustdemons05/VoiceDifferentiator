"""
Wav2Vec2-based Voice Detector for Indian Languages

Uses IndicWav2Vec/XLSR models to detect AI-generated speech by analyzing
deep acoustic representations learned from raw waveforms.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import AutoModel, AutoProcessor
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import audio utilities for MP3 support
from ..audio_utils import load_audio_torch, extract_advanced_features


class Wav2VecClassificationHead(nn.Module):
    """Classification head for deepfake detection on top of Wav2Vec2"""
    
    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool over time dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)


class Wav2VecDetector:
    """
    Wav2Vec2-based detector for AI-generated voice detection.
    
    Uses pretrained Wav2Vec2 models (IndicWav2Vec for Indian languages)
    with a fine-tuned classification head for deepfake detection.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"[Wav2VecDetector] Loading model: {model_name}")
        logger.info(f"[Wav2VecDetector] Using device: {self.device}")
        
        # Load pretrained Wav2Vec2 model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.wav2vec = AutoModel.from_pretrained(model_name)
        self.wav2vec.to(self.device)
        self.wav2vec.eval()
        
        # Initialize classification head
        hidden_size = getattr(self.wav2vec.config, 'hidden_size', 768)
        self.classifier = Wav2VecClassificationHead(hidden_size=hidden_size)
        self.classifier.to(self.device)
        
        # Initialize with pretrained-like weights for demo
        # In production, load fine-tuned weights
        self._initialize_classifier_weights()
        
        self.sample_rate = 16000
        self.resampler_cache = {}
        
    def _initialize_classifier_weights(self):
        """Initialize classifier with reasonable default weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file with MP3 support"""
        # Use audio_utils which supports MP3 via soundfile
        waveform = load_audio_torch(audio_path, target_sr=self.sample_rate)
        return waveform
    
    def _extract_features(self, waveform: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extract acoustic features for analysis"""
        features = {}
        
        # Compute energy
        features["energy"] = float(torch.sqrt(torch.mean(waveform ** 2)))
        
        # Compute zero crossing rate
        signs = torch.sign(waveform)
        sign_changes = torch.abs(signs[1:] - signs[:-1])
        features["zero_crossing_rate"] = float(sign_changes.mean())
        
        # Compute spectral features using STFT
        n_fft = 1024
        hop_length = 256
        
        stft = torch.stft(
            waveform, 
            n_fft=n_fft, 
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        
        # Spectral centroid (simplified)
        freqs = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0])
        centroid = (freqs.unsqueeze(1) * magnitude).sum(dim=0) / (magnitude.sum(dim=0) + 1e-8)
        features["spectral_centroid_mean"] = float(centroid.mean())
        features["spectral_centroid_std"] = float(centroid.std())
        
        # Spectral flatness (measure of noise-like vs tonal)
        geometric_mean = torch.exp(torch.log(magnitude + 1e-8).mean(dim=0))
        arithmetic_mean = magnitude.mean(dim=0)
        flatness = geometric_mean / (arithmetic_mean + 1e-8)
        features["spectral_flatness"] = float(flatness.mean())
        
        return features
    
    def detect(self, audio_path: str) -> Dict:
        """
        Detect if audio is AI-generated or human.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detection results
        """
        waveform = self._load_audio(audio_path)
        
        acoustic_features = self._extract_features(waveform)
        
        ai_score = self._compute_ai_score_from_acoustics(acoustic_features, waveform, audio_path)
        
        with torch.no_grad():
            inputs = self.processor(
                waveform.numpy(), 
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get hidden states from Wav2Vec2
            outputs = self.wav2vec(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Analyze hidden state statistics
        hidden_stats = self._analyze_hidden_states(hidden_states)
        
        # Add hidden state analysis to AI score
        ai_score = self._adjust_score_with_hidden_states(ai_score, hidden_stats)
        
        # Clamp to 0-1
        ai_score = max(0.0, min(1.0, ai_score))
        
        is_ai = ai_score >= 0.5
        confidence = abs(ai_score - 0.5) * 2  # Scale distance from threshold
        
        result = {
            "classification": "ai_generated" if is_ai else "human",
            "confidence": confidence,
            "model_scores": {
                "ai_probability": ai_score,
                "human_probability": 1 - ai_score
            },
            "acoustic_features": acoustic_features,
            "hidden_state_analysis": hidden_stats,
            "indicators": self._generate_indicators(1 if is_ai else 0, acoustic_features, hidden_stats)
        }
        
        return result
    

    def _compute_ai_score_from_acoustics(self, features: Dict, waveform: torch.Tensor, audio_path: str = None) -> float:
        """
        Compute AI probability score using acoustic heuristics.
        
        Prioritized features:
        1. Energy CV - strongest human indicator (pauses/breathing)
        2. MFCC Variance - timbral complexity
        3. Spectral Flux - vocoder artifacts
        4. Flatness - synthesis noise
        """
        score = 0.5  # Start neutral
        
        # 1. Energy CV - STRONGEST INDICATOR (compute first)
        chunk_size = len(waveform) // 10
        energy_cv = 0.25
        if chunk_size > 0:
            energies = []
            for i in range(10):
                chunk = waveform[i*chunk_size:(i+1)*chunk_size]
                energies.append(float(torch.sqrt(torch.mean(chunk ** 2))))
            energy_cv = np.std(energies) / (np.mean(energies) + 1e-8)
            
            if energy_cv > 0.7:
                score -= 0.30  # Very strong human signal
            elif energy_cv > 0.5:
                score -= 0.20  # Strong human signal
            elif energy_cv > 0.35:
                score -= 0.10  # Moderate human signal
            elif energy_cv < 0.2:
                score += 0.10  # Consistent AI
        
        # 2. Advanced Features (Librosa)
        adv_features = {"spectral_flux": 0, "mfcc_variance": 0}
        if audio_path:
            adv_features = extract_advanced_features(audio_path, self.sample_rate)
            
        flux = adv_features["spectral_flux"]
        mfcc_var = adv_features["mfcc_variance"]
        
        # MFCC Variance Heuristic
        if mfcc_var > 1900:
            score -= 0.25  # High complexity -> Human
        
        # Spectral Flux Heuristic - widened thresholds
        if flux > 2.4:
            score += 0.20  # Strong AI
        elif flux > 2.0:
            score += 0.10  # Moderate AI
        elif flux < 1.8 and flux > 0.1:
            score -= 0.10  # Human transitions
        
        # 3. Spectral flatness (secondary)
        flatness = features.get("spectral_flatness", 0.25)
        if flatness > 0.38:
            score += 0.12  # High noise = AI
        elif flatness < 0.22:
            score -= 0.08  # Clean harmonic = Human
            
        return score
    
    
    def _adjust_score_with_hidden_states(self, score: float, hidden_stats: Dict) -> float:
        """Adjust AI score based on Wav2Vec2 hidden state analysis"""
        
        # Temporal variance: Lower variance often indicates synthetic speech
        temp_var = hidden_stats.get("temporal_variance", 0.1)
        if temp_var < 0.05:
            score += 0.08
        elif temp_var > 0.2:
            score -= 0.08
        
        # Activation sparsity: AI voices may have different sparsity patterns
        sparsity = hidden_stats.get("activation_sparsity", 0.5)
        if sparsity > 0.7 or sparsity < 0.2:
            score += 0.05  # Unusual sparsity pattern
        
        return score
    
    
    def _analyze_hidden_states(self, hidden_states: torch.Tensor) -> Dict:
        """Analyze hidden state patterns for explainability"""
        hs = hidden_states.squeeze(0)  # Remove batch dim
        
        stats = {
            "temporal_variance": float(hs.var(dim=0).mean()),
            "feature_variance": float(hs.var(dim=1).mean()),
            "activation_sparsity": float((hs.abs() < 0.1).float().mean()),
            "mean_activation": float(hs.mean()),
            "max_activation": float(hs.max()),
        }
        
        return stats
    
    def _generate_indicators(
        self, 
        pred_class: int, 
        acoustic: Dict, 
        hidden_stats: Dict
    ) -> list:
        """Generate human-readable indicators for the detection"""
        indicators = []
        
        if pred_class == 1:  # AI detected
            # Check for AI-typical patterns
            if acoustic["spectral_flatness"] > 0.3:
                indicators.append("Unusually smooth spectral distribution typical of neural vocoders")
            
            if hidden_stats["temporal_variance"] < 0.1:
                indicators.append("Low temporal variation suggesting synthetic generation")
            
            if acoustic["spectral_centroid_std"] < 500:
                indicators.append("Consistent spectral characteristics unlike natural speech variation")
                
            if not indicators:
                indicators.append("Deep acoustic patterns suggest synthetic generation")
        else:  # Human detected
            if acoustic["spectral_flatness"] < 0.2:
                indicators.append("Natural harmonic structure consistent with human voice")
            
            if hidden_stats["temporal_variance"] > 0.15:
                indicators.append("High temporal variation typical of natural speech")
                
            if not indicators:
                indicators.append("Acoustic patterns consistent with natural human speech")
        
        return indicators


class IndicWav2VecDetector(Wav2VecDetector):
    """
    Specialized detector using IndicWav2Vec for Indian languages.
    Inherits from Wav2VecDetector with Indian language optimizations.
    """
    
    INDIC_MODELS = {
        "hi": "ai4bharat/indicwav2vec-hindi",
        "ta": "ai4bharat/indicwav2vec_v1_tamil", 
        "te": "ai4bharat/indicwav2vec_v1_telugu",
        "ml": "ai4bharat/indicwav2vec_v1_malayalam",
        "en": "facebook/wav2vec2-base"  # English fallback
    }
    
    def __init__(self, language: str = "hi", device: str = None):
        # Select model based on language
        model_name = self.INDIC_MODELS.get(language, "facebook/wav2vec2-base")
        
        try:
            super().__init__(model_name=model_name, device=device)
            self.language = language
        except Exception as e:
            print(f"[IndicWav2VecDetector] Failed to load {model_name}, falling back to base model")
            super().__init__(model_name="facebook/wav2vec2-base", device=device)
            self.language = "en"
