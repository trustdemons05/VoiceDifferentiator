"""
Ensemble Voice Detector

Combines multiple detection methods (Wav2Vec2, Spectrogram CNN, PersonaPlex)
with weighted fusion for robust AI voice detection.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import traceback
import logging

logger = logging.getLogger(__name__)

from .wav2vec_detector import Wav2VecDetector, IndicWav2VecDetector
from .spectrogram_cnn import SpectrogramDetector
from .personaplex_detector import PersonaPlexDetector


class EnsembleVoiceDetector:
    """
    Ensemble detector that combines multiple AI voice detection methods.
    
    Components:
    1. Wav2Vec2/IndicWav2Vec - Deep acoustic pattern analysis
    2. Spectrogram CNN - Visual pattern detection in mel spectrograms
    3. PersonaPlex Detector - AI tool signature matching
    
    Uses weighted fusion to combine predictions for robust detection.
    """
    
    DEFAULT_WEIGHTS = {
        "wav2vec": 0.45,      # Primary - best for language-specific patterns
        "spectrogram": 0.35,  # Secondary - catches vocoder artifacts
        "personaplex": 0.20   # Tertiary - identifies specific tools
    }
    
    def __init__(
        self,
        language: str = "en",
        device: str = None,
        weights: Dict[str, float] = None,
        enable_parallel: bool = True
    ):
        """
        Initialize ensemble detector.
        
        Args:
            language: Primary language code (en, hi, ta, te, ml)
            device: Torch device (cuda/cpu)
            weights: Custom weights for ensemble fusion
            enable_parallel: Enable parallel inference for speed
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.enable_parallel = enable_parallel
        
        logger.info(f"[EnsembleDetector] Initializing with language={language}, device={self.device}")
        
        self._init_detectors()
        
    def _init_detectors(self):
        """Initialize all component detectors"""
        logger.info("[EnsembleDetector] Loading Wav2Vec detector...")
        try:
            self.wav2vec_detector = IndicWav2VecDetector(
                language=self.language,
                device=self.device
            )
        except Exception as e:
            logger.warning(f"[EnsembleDetector] Warning: IndicWav2Vec failed, using base model: {e}")
            self.wav2vec_detector = Wav2VecDetector(
                model_name="facebook/wav2vec2-base",
                device=self.device
            )
        
        logger.info("[EnsembleDetector] Loading Spectrogram detector...")
        self.spectrogram_detector = SpectrogramDetector(device=self.device)
        
        logger.info("[EnsembleDetector] Loading PersonaPlex detector...")
        self.personaplex_detector = PersonaPlexDetector()
        
        logger.info("[EnsembleDetector] All detectors loaded successfully!")
    
    def _run_detector(self, detector_name: str, audio_path: str) -> Dict:
        """Run a single detector with error handling"""
        try:
            if detector_name == "wav2vec":
                return self.wav2vec_detector.detect(audio_path)
            elif detector_name == "spectrogram":
                return self.spectrogram_detector.detect(audio_path)
            elif detector_name == "personaplex":
                return self.personaplex_detector.detect(audio_path)
            else:
                raise ValueError(f"Unknown detector: {detector_name}")
        except Exception as e:
            logger.error(f"[EnsembleDetector] Error in {detector_name}: {e}")
            traceback.print_exc()
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "model_scores": {"ai_probability": 0.5, "human_probability": 0.5},
                "error": str(e)
            }
    
    def detect(self, audio_path: str) -> Dict:
        """
        Perform ensemble detection on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Comprehensive detection result with ensemble fusion
        """
        results = {}
        
        # Run detectors (parallel or sequential)
        if self.enable_parallel:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    name: executor.submit(self._run_detector, name, audio_path)
                    for name in ["wav2vec", "spectrogram", "personaplex"]
                }
                results = {name: future.result() for name, future in futures.items()}
        else:
            for name in ["wav2vec", "spectrogram", "personaplex"]:
                results[name] = self._run_detector(name, audio_path)
        
        ensemble_result = self._fuse_results(results)
        
        ensemble_result["component_results"] = {
            name: {
                "classification": r.get("classification"),
                "confidence": r.get("confidence"),
                "ai_probability": r.get("model_scores", {}).get("ai_probability")
            }
            for name, r in results.items()
        }
        
        ensemble_result["detailed_analysis"] = {
            "wav2vec_indicators": results["wav2vec"].get("indicators", []),
            "spectrogram_indicators": results["spectrogram"].get("indicators", []),
            "personaplex_indicators": results["personaplex"].get("indicators", []),
            "detected_tools": results["personaplex"].get("detected_tools", [])
        }
        
        return ensemble_result
    
    def _fuse_results(self, results: Dict[str, Dict]) -> Dict:
        """
        Fuse results from multiple detectors using weighted voting.
        
        Uses AI probability from each detector weighted by confidence.
        """
        weighted_ai_prob = 0.0
        total_weight = 0.0
        all_indicators = []
        
        for detector_name, result in results.items():
            if "error" in result:
                continue
                
            weight = self.weights.get(detector_name, 0.33)
            ai_prob = result.get("model_scores", {}).get("ai_probability", 0.5)
            confidence = result.get("confidence", 0.5)
            
            # Weight by both assigned weight and detector confidence
            effective_weight = weight * confidence
            weighted_ai_prob += ai_prob * effective_weight
            total_weight += effective_weight
            
            # Collect indicators
            all_indicators.extend(result.get("indicators", []))
        
        # Normalize
        if total_weight > 0:
            final_ai_prob = weighted_ai_prob / total_weight
        else:
            final_ai_prob = 0.5
        
        # Determine classification
        is_ai = final_ai_prob >= 0.5
        
        # Confidence is the probability of the winning class
        base_confidence = final_ai_prob if is_ai else (1.0 - final_ai_prob)
        
        # Boost confidence if detectors agree
        agreement = self._calculate_agreement(results)
        if agreement > 0.8:
            # Boost towards 1.0
            confidence = base_confidence + (1.0 - base_confidence) * 0.2
        else:
            confidence = base_confidence
        
        # Select top indicators
        unique_indicators = list(dict.fromkeys(all_indicators))[:5]
        
        # Determine detected AI tool
        detected_tools = results.get("personaplex", {}).get("detected_tools", [])
        primary_tool = detected_tools[0]["tool_name"] if detected_tools else None
        
        return {
            "classification": "ai_generated" if is_ai else "human",
            "confidence": round(confidence, 4),
            "ai_probability": round(final_ai_prob, 4),
            "human_probability": round(1 - final_ai_prob, 4),
            "ai_tool_detected": primary_tool,
            "detector_agreement": round(agreement, 4),
            "indicators": unique_indicators,
            "explanation": self._generate_explanation(
                is_ai, confidence, unique_indicators, primary_tool, results
            )
        }
    
    def _calculate_agreement(self, results: Dict[str, Dict]) -> float:
        """Calculate agreement between detectors (0-1)"""
        classifications = []
        
        for result in results.values():
            if "error" not in result:
                classifications.append(result.get("classification") == "ai_generated")
        
        if not classifications:
            return 0.0
        
        # Agreement is proportion of detectors that agree with majority
        ai_count = sum(classifications)
        human_count = len(classifications) - ai_count
        majority_count = max(ai_count, human_count)
        
        return majority_count / len(classifications)
    
    def _generate_explanation(
        self,
        is_ai: bool,
        confidence: float,
        indicators: List[str],
        tool_detected: Optional[str],
        results: Dict
    ) -> Dict:
        """Generate comprehensive explanation for the detection"""
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "high"
            summary_prefix = "Strong evidence"
        elif confidence >= 0.6:
            confidence_level = "medium"
            summary_prefix = "Moderate evidence"
        else:
            confidence_level = "low"
            summary_prefix = "Weak evidence"
        
        if is_ai:
            summary = f"{summary_prefix} of AI-generated voice detected"
            if tool_detected:
                summary += f" (likely {tool_detected})"
        else:
            summary = f"{summary_prefix} suggests this is authentic human speech"
        
        # Technical details
        technical_details = {
            "spectral_artifacts": [],
            "temporal_patterns": [],
            "synthesis_markers": []
        }
        
        # Categorize indicators
        for ind in indicators:
            ind_lower = ind.lower()
            if any(kw in ind_lower for kw in ["spectral", "frequency", "band", "harmonic"]):
                technical_details["spectral_artifacts"].append(ind)
            elif any(kw in ind_lower for kw in ["temporal", "time", "variation", "smooth"]):
                technical_details["temporal_patterns"].append(ind)
            elif any(kw in ind_lower for kw in ["vocoder", "synthesis", "signature", "neural"]):
                technical_details["synthesis_markers"].append(ind)
            else:
                # Default to synthesis markers for AI indicators
                if is_ai:
                    technical_details["synthesis_markers"].append(ind)
                else:
                    technical_details["temporal_patterns"].append(ind)
        
        return {
            "summary": summary,
            "confidence_level": confidence_level,
            "technical_details": technical_details,
            "key_indicators": indicators[:3],
            "model_contributions": {
                name: round(self.weights[name], 2)
                for name in self.weights
            }
        }


def create_detector(language: str = "en", device: str = None) -> EnsembleVoiceDetector:
    """
    Factory function to create an ensemble detector.
    
    Args:
        language: Primary language code (en, hi, ta, te, ml)
        device: Torch device (cuda/cpu/auto)
        
    Returns:
        Configured EnsembleVoiceDetector instance
    """
    return EnsembleVoiceDetector(language=language, device=device)
