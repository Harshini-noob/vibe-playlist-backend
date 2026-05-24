# vibe. — backend

FastAPI backend for the vibe-to-playlist app.

## Tech Stack
- FastAPI + Python
- Groq LLM (llama-3.3-70b) — mood extraction + song recommendations
- Groq Whisper — multilingual voice transcription (Tamil, Hindi, English)
- iTunes Search API — track metadata + 30-sec previews
- SQLite — feedback storage and re-ranking

## Features
- Multilingual voice input (Tamil / Hindi / English / Mixed)
- LLM-powered mood DNA extraction from scene descriptions
- Era-aware and language-aware song recommendations
- Community feedback loop — thumbs up/down re-ranks songs over time

## Setup
```bash