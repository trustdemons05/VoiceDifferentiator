# AI Voice Detection API

Detect AI-generated voices across multiple languages with NVIDIA PersonaPlex detection.

## Features

- 🎯 **Multi-Model Detection**: Ensemble of Wav2Vec2, CNN, and signature analysis
- 🌐 **Multi-Language Support**: Tamil, English, Hindi, Malayalam, Telugu
- 🔍 **AI Tool Detection**: Identifies NVIDIA PersonaPlex, ElevenLabs, Azure TTS, Google WaveNet
- 📊 **Detailed Explanations**: Technical analysis with confidence scores

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Development mode with auto-reload
python run.py --reload

# Or directly with uvicorn
uvicorn app.main:app --reload --port 8000
```

### 3. Test the API

Open http://localhost:8000/docs for the interactive Swagger UI.

## API Usage

### Detect AI Voice

**Endpoint:** `POST /api/v1/detect`

**Request:**
```json
{
  "audio_base64": "<Base64-encoded MP3 audio>",
  "language_hint": "en"
}
```

**Response:**
```json
{
  "classification": "ai_generated",
  "confidence": 0.92,
  "ai_probability": 0.92,
  "human_probability": 0.08,
  "ai_tool_detected": "NVIDIA PersonaPlex/Riva",
  "explanation": {
    "summary": "Strong evidence of AI-generated voice detected",
    "confidence_level": "high",
    "technical_details": {
      "spectral_artifacts": ["Vocoder artifacts in 6-8kHz"],
      "temporal_patterns": ["Low temporal variation"],
      "synthesis_markers": ["HiFi-GAN fingerprint"]
    },
    "key_indicators": [
      "Signature matches NVIDIA PersonaPlex/Riva",
      "Phase coherence matches HiFi-GAN pattern"
    ]
  }
}
```

## Detection Methods

| Method | Weight | Description |
|--------|--------|-------------|
| IndicWav2Vec | 45% | Deep acoustic patterns using Indian language models |
| Spectrogram CNN | 35% | Visual artifact detection in mel spectrograms |
| PersonaPlex Detector | 20% | AI tool signature matching |

## Supported Languages

| Code | Language |
|------|----------|
| ta | Tamil |
| en | English |
| hi | Hindi |
| ml | Malayalam |
| te | Telugu |

## Project Structure

```
hackathon1/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── schemas.py       # Pydantic models
│   └── models/
│       ├── wav2vec_detector.py      # IndicWav2Vec detector
│       ├── spectrogram_cnn.py       # CNN classifier
│       ├── personaplex_detector.py  # AI tool detection
│       └── ensemble_detector.py     # Ensemble fusion
├── requirements.txt
├── run.py                   # Server entry point
└── README.md
```

## License

MIT License
