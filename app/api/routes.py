"""
API Routes for Voice Detection

Defines all API endpoints for the voice detection service.
"""
import base64
import tempfile
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Security, Request
from fastapi.security import APIKeyHeader
import httpx
from pydantic import ValidationError

from .schemas import (
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ErrorResponse,
    LanguageCode
)
from ..models.ensemble_detector import EnsembleVoiceDetector, create_detector


_detector: Optional[EnsembleVoiceDetector] = None


def get_detector(language: str = "en") -> EnsembleVoiceDetector:
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = create_detector(language=language)
    return _detector


router = APIRouter(prefix="/api/v1", tags=["Voice Detection"])


@router.post(
    "/detect",
    response_model=DetectResponse,
    # ... (responses dict)
    summary="Detect AI-Generated Voice",
    description="..."
)
async def detect_voice(
    request: DetectRequest,
    api_key: str = Security(APIKeyHeader(name="X-API-Key", auto_error=False))
) -> DetectResponse:
    """
    Detect if the provided audio is AI-generated or human.
    """
    # Optional API Key Validation
    # We allow the key to be missing for public testing/hackathon judges.
    if api_key:
        # In a real app, validate against DB/Env
        pass
    
    temp_path = None
    
    try:
        audio_bytes = None
        
        if request.audio_url:
            # Download from URL
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get(request.audio_url, follow_redirects=True, timeout=30.0)
                    resp.raise_for_status()
                    audio_bytes = resp.content
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download audio from URL: {str(e)}"
                    )
        elif request.audio_base64:
            # Decode Base64
            try:
                audio_bytes = base64.b64decode(request.audio_base64)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Base64 encoding: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'audio_url' or 'audio_base64' must be provided."
            )
            
        if len(audio_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file too large. Maximum size is 10MB."
            )
        
        if len(audio_bytes) < 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file too small. Minimum duration is 1 second."
            )
        
        # Write audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        # Get language from request or default to English
        language = request.language_hint.value if request.language_hint else "en"
        
        detector = get_detector(language=language)

        
        result = detector.detect(temp_path)
        
        return DetectResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            explanation=result["explanation"]["key_indicators"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the voice detection service"
)
async def health_check() -> HealthResponse:
    """
    Check if the service is healthy and all models are loaded.
    """
    import torch
    
    detector = get_detector()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "wav2vec": hasattr(detector, 'wav2vec_detector'),
            "spectrogram_cnn": hasattr(detector, 'spectrogram_detector'),
            "personaplex": hasattr(detector, 'personaplex_detector')
        },
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


@router.get(
    "/languages",
    summary="Supported Languages",
    description="Get list of supported languages for voice detection"
)
async def get_languages():
    """Get supported languages"""
    return {
        "languages": [
            {"code": "ta", "name": "Tamil"},
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ml", "name": "Malayalam"},
            {"code": "te", "name": "Telugu"}
        ]
    }


@router.get(
    "/tools",
    summary="Detectable AI Tools",
    description="Get list of AI voice synthesis tools that can be detected"
)
async def get_detectable_tools():
    """Get list of detectable AI tools"""
    return {
        "tools": [
            {
                "id": "nvidia_personaplex",
                "name": "NVIDIA PersonaPlex/Riva",
                "description": "NVIDIA's conversational AI voice synthesis"
            },
            {
                "id": "elevenlabs",
                "name": "ElevenLabs",
                "description": "ElevenLabs voice cloning and synthesis"
            },
            {
                "id": "azure_neural",
                "name": "Microsoft Azure Neural TTS",
                "description": "Azure Cognitive Services Neural TTS"
            },
            {
                "id": "google_wavenet",
                "name": "Google WaveNet/Neural2",
                "description": "Google Cloud Text-to-Speech"
            }
        ]
    }
