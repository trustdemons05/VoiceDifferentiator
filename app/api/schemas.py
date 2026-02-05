"""
Pydantic Schemas for API Request/Response Validation
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class LanguageCode(str, Enum):
    """Supported language codes"""
    TAMIL = "ta"
    ENGLISH = "en"
    HINDI = "hi"
    MALAYALAM = "ml"
    TELUGU = "te"


class ClassificationType(str, Enum):
    """Voice classification types"""
    AI_GENERATED = "ai_generated"
    HUMAN = "human"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level descriptors"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============== Request Schemas ==============

class DetectRequest(BaseModel):
    """Request schema for voice detection endpoint"""
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded MP3 audio data (mutually exclusive with audio_url)",
        min_length=100,
        examples=["SGVsbG8gV29ybGQ..."],
        alias="audioBase64"
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to audio file (MP3, WAV, etc.) (mutually exclusive with audio_base64)",
        alias="audioUrl"
    )
    language_hint: Optional[LanguageCode] = Field(
        default=None,
        description="Optional hint for expected language (improves accuracy)"
    )
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "audio_url": "https://example.com/sample.mp3",
                "language_hint": "en"
            }
        }


class ExternalTesterRequest(BaseModel):
    """Request schema compatible with external endpoint tester"""
    Language: str = Field(
        ...,
        description="Language code (en, ta, hi, ml, te)",
        examples=["en"]
    )
    Audio_Format: str = Field(
        default="mp3",
        description="Audio format (mp3, wav, etc.)",
        alias="Audio Format"
    )
    Audio_Base64_Format: str = Field(
        ...,
        description="Base64-encoded audio data",
        alias="Audio Base64 Format"
    )
    
    class Config:
        populate_by_name = True



# ============== Response Schemas ==============

class TechnicalDetails(BaseModel):
    """Technical analysis details"""
    spectral_artifacts: List[str] = Field(default_factory=list)
    temporal_patterns: List[str] = Field(default_factory=list)
    synthesis_markers: List[str] = Field(default_factory=list)


class ExplanationResponse(BaseModel):
    """Detailed explanation of detection result"""
    summary: str = Field(
        ...,
        description="Human-readable summary of the detection"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Confidence level (high/medium/low)"
    )
    technical_details: TechnicalDetails = Field(
        ...,
        description="Technical analysis breakdown"
    )
    key_indicators: List[str] = Field(
        ...,
        description="Top indicators that led to the classification"
    )
    model_contributions: Dict[str, float] = Field(
        ...,
        description="Weight contribution of each detection model"
    )


class ComponentResult(BaseModel):
    """Result from an individual detection component"""
    classification: Optional[str] = None
    confidence: Optional[float] = None
    ai_probability: Optional[float] = None


class DetectedTool(BaseModel):
    """Information about detected AI voice tool"""
    tool_id: str
    tool_name: str
    confidence: float
    reasons: List[str]


class DetailedAnalysis(BaseModel):
    """Detailed analysis from all detection components"""
    wav2vec_indicators: List[str] = Field(default_factory=list)
    spectrogram_indicators: List[str] = Field(default_factory=list)
    personaplex_indicators: List[str] = Field(default_factory=list)
    detected_tools: List[DetectedTool] = Field(default_factory=list)


class DetectResponse(BaseModel):
    """Response schema for voice detection endpoint"""
    classification: ClassificationType = Field(
        ...,
        description="Classification result: ai_generated or human"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    explanation: List[str] = Field(
        ...,
        description="List of key indicators explaining the classification"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "classification": "ai_generated",
                "confidence": 0.85,
                "explanation": [
                    "Unusually smooth spectral distribution typical of neural vocoders",
                    "Low temporal variation suggesting synthetic generation",
                    "Periodic patterns suggesting neural vocoder"
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str
    models_loaded: Dict[str, bool]
    device: str


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    code: str
