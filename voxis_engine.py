import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import pygame
import database
import settings
import time
import threading
import queue
import tempfile
import collections
import numpy as np
from faster_whisper import WhisperModel
import io

class GemeniEngine:
    """
    ML-Powered Translation Engine for Gemeni.
    Uses OpenAI Whisper (via faster-whisper) for ultra-accurate ASR.
    """
    def __init__(self):
        self.config = settings.load_settings()
        database.init_db()
        
        # 1. Initialize ML Model (Task 2.2 - Accuracy Boost)
        print("Gemeni: Initializing Machine Learning Engine (Whisper)...")
        # 'base' is the sweet spot for speed/accuracy on CPU
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # 2. Audio & TTS Setup
        pygame.mixer.init()
        self.tts_queue = queue.Queue()
        self.is_tts_playing = False
        threading.Thread(target=self._tts_worker, daemon=True).start()
        
        # 3. Recognizer for Microphone Handling
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 500
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.5 
        
    def _tts_worker(self):
        """Sequential TTS playback with Blackout Logic."""
        while True:
            try:
                text = self.tts_queue.get()
                if text is None: break
                self.is_tts_playing = True
                
                target_code = self.config.get("target_lang", "es").lower()
                tts = gTTS(text=text, lang=target_code)
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    temp_path = f.name
                    tts.save(temp_path)
                
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy(): time.sleep(0.1)
                pygame.mixer.music.unload()
                os.remove(temp_path)
                
                self.is_tts_playing = False
                self.tts_queue.task_done()
            except Exception as e:
                self.is_tts_playing = False
                print(f"Gemeni TTS Error: {e}")

    def perform_ml_asr(self, audio_data):
        """Uses Local ML (Whisper) to transcribe audio with high accuracy."""
        # Convert SpeechRecognition audio to numpy array for Whisper
        wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe using ML (Task 2.2)
        segments, info = self.model.transcribe(audio_np, beam_size=5)
        
        # Join segments into a full sentence
        text = "".join([segment.text for segment in segments]).strip()
        return text

    def run(self):
        """The Main Continuous Communication Loop."""
        print("\n" + "="*50)
        print("      GEMENI: MACHINE LEARNING MODE (WHISPER)")
        print("="*50)
        
        try:
            with sr.Microphone() as source:
                print("Gemeni: Calibrating for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                print("\nGemeni: [Online] - Listening for natural speech.")
                
                while True:
                    if self.is_tts_playing:
                        time.sleep(0.2)
                        continue

                    try:
                        # 1. Capture Raw Audio
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=20)
                        if self.is_tts_playing: continue

                        # RELOAD SETTINGS (Instant updates from Dashboard)
                        self.config = settings.load_settings()

                        print("Gemeni: [ML Analyzing...]")
                        
                        # 2. High-Accuracy ML ASR
                        text = self.perform_ml_asr(audio)
                        
                        if not text or len(text) < 2: continue

                        # 3. Translation
                        target_code = self.config.get("target_lang", "es").lower()
                        translator = GoogleTranslator(source='auto', target=target_code)
                        translated_text = translator.translate(text)
                        
                        # 4. Output
                        print(f"\n[{time.strftime('%H:%M:%S')}] Detected: \"{text}\"")
                        print(f"[{time.strftime('%H:%M:%S')}] Gemeni: \"{translated_text}\"")
                        
                        self.tts_queue.put(translated_text)
                        database.log_translation("AUTO", target_code.upper(), text, translated_text)
                        
                    except sr.WaitTimeoutError: continue
                    except sr.UnknownValueError: continue
                    except Exception as e:
                        print(f"Gemeni: Engine Error: {e}")
                        
        except KeyboardInterrupt: print("\nGemeni: Shutdown.")

if __name__ == "__main__":
    gemeni = GemeniEngine()
    gemeni.run()
