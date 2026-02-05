---
title: AI Voice Detection API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# AI Voice Detection API

Detect AI-generated voices using advanced acoustic analysis and neural network patterns.

## API Endpoints

- `POST /api/v1/detect` - Detect if audio is AI-generated
- `GET /api/v1/health` - Health check
- `GET /docs` - API documentation

## Usage

```bash
curl -X POST "https://YOUR-SPACE.hf.space/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{"audioUrl": "https://example.com/audio.mp3"}'
```
