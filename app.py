import streamlit as st
import database
import settings
import os
import tempfile
import base64
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# --- Page Config ---
st.set_page_config(page_title="VOXIS Translator", layout="wide", initial_sidebar_state="collapsed")

# --- Global Styles ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');

    .main { background-color: #050810; color: #e8edf5; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; color: #00c8ff; text-align: center; }

    .mic-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    .history-card {
        background: #111827;
        border: 1px solid #1e2d4a;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Resource Initialization ---
@st.cache_resource
def load_whisper():
    """Loads Whisper ML model into memory once."""
    return WhisperModel("base", device="cpu", compute_type="int8")


def play_audio(file_path):
    """Renders an autoplay audio element and then removes the temp file."""
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# --- Main Application ---
def main():
    st.title("VOXIS")

    config = settings.load_settings()
    model = load_whisper()
    database.init_db()

    # ── Sidebar: Settings ──
    with st.sidebar:
        st.header("VOXIS Settings")
        LANG_MAP = {
            "Spanish":    "es", "French":     "fr", "German":  "de",
            "Italian":    "it", "Portuguese": "pt", "Russian": "ru",
            "Japanese":   "ja", "Chinese":    "zh-CN", "Hindi": "hi",
            "Arabic":     "ar", "English":    "en", "Korean": "ko",
            "Dutch":      "nl", "Turkish":    "tr",
        }
        current_code = config.get("target_lang", "es").lower()
        current_name = next(
            (n for n, c in LANG_MAP.items() if c == current_code), "Spanish"
        )
        target_name = st.selectbox(
            "Translate To",
            list(LANG_MAP.keys()),
            index=list(LANG_MAP.keys()).index(current_name),
        )
        target_code = LANG_MAP[target_name]

        if st.button("Save Settings"):
            config["target_lang"] = target_code.upper()
            settings.save_settings(config)
            st.success("Preferences Saved!")

    # ── UI Layout ──
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='mic-container'>", unsafe_allow_html=True)
        st.write("### Voice Capture")
        st.write("Tap to record your message for translation.")
        
        audio_record = mic_recorder(
            start_prompt="🎙️ Start Voxis",
            stop_prompt="🛑 Stop & Translate",
            key="voxis_recorder",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if audio_record:
            with st.spinner("Voxis: Analyzing..."):
                audio_bytes = audio_record["bytes"]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_bytes)
                    temp_path = f.name

                try:
                    segments, _ = model.transcribe(temp_path, beam_size=5)
                    text = "".join([s.text for s in segments]).strip()

                    if text:
                        translator = GoogleTranslator(source="auto", target=target_code)
                        translated_text = translator.translate(text)

                        st.markdown(
                            f"""
                            <div class='history-card'>
                                <p style='color:#5a6a8a; font-size:12px;'>DETECTED</p>
                                <p style='font-size:18px;'>{text}</p>
                                <hr style='border-color:#1e2d4a'>
                                <p style='color:#00c8ff; font-size:12px;'>VOXIS ({target_name})</p>
                                <p style='font-size:28px; color:white;'><b>{translated_text}</b></p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # TTS
                        try:
                            tts = gTTS(text=translated_text, lang=target_code)
                            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_tts:
                                tts.save(f_tts.name)
                                tts_path = f_tts.name
                            play_audio(tts_path)
                        except Exception as tts_err:
                            st.warning(f"Voice output failed: {tts_err}")

                        database.log_translation(
                            "AUTO", target_code.upper(), text, translated_text
                        )
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    with col2:
        st.markdown("### Activity Stream")
        history = database.get_recent_history(4)
        if history:
            for h in history:
                st.markdown(
                    f"""
                    <div class='history-card' style='padding:10px;'>
                        <small style='color:#5a6a8a;'>{h[1]}</small><br>
                        <b>{h[4]}</b> <span style='color:#00c8ff;'>→</span> <b>{h[5]}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<p style='color:#5a6a8a; text-align:center;'>No translations logged.</p>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#5a6a8a; font-size:10px;'>"
        "VOXIS v1.0 — HIGH ACCURACY ML TRANSLATOR</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
