# VOXIS

VOXIS is a high-accuracy, real-time machine learning translation application. It leverages OpenAI's Whisper model for robust speech-to-text and Google Translate for seamless multilingual communication.

## Features
- **Machine Learning Core:** Powered by `faster-whisper` for ultra-accurate transcription of fast or long sentences.
- **Unified Interface:** A sleek, mobile-friendly Streamlit dashboard with "Hold to Translate" functionality.
- **Natural Voice Output:** Integrated `gTTS` for high-quality audio playback of translations.
- **Translation History:** Persistent SQLite database to track and visualize your activity.
- **Wide Language Support:** Translate between 15+ major global languages instantly.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/oisinodowd/voxisTranslate.git
   cd voxisTranslate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the unified Streamlit application:
```bash
python -m streamlit run app.py
```

## Technology Stack
- **ASR:** OpenAI Whisper (base)
- **NMT:** Google Translate
- **TTS:** Google Text-to-Speech
- **Frontend:** Streamlit
- **Persistence:** SQLite & JSON
