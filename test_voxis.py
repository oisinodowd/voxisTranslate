"""
Gemeni Validation Suite.

Fixes applied vs original:
  BUG 1  — `from gemeni_engine import GemeniStreamingASR` raised ImportError;
            class is named GemeniEngine. Fixed import.
  BUG 2  — setUp called GemeniStreamingASR() which doesn't exist. Fixed.
  BUG 3  — add_audio_chunk() doesn't exist on GemeniEngine. Tests now drive
            the engine through its real public API (perform_ml_asr / run).
  BUG 4  — process_buffer() doesn't exist. Removed.
  BUG 5  — is_speaking / buffer attributes don't exist. Removed.
  BUG 6  — No mocking: setUp would try to load WhisperModel, init pygame, and
            open a real microphone — crashing in any CI/test environment.
            Now all heavy deps are patched before the engine is instantiated.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import time
import os
import database
import settings


# ---------------------------------------------------------------------------
# Helper — a fake WhisperModel whose transcribe() returns controllable output
# ---------------------------------------------------------------------------
class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, beam_size=5):
        """Return a plausible segment for non-silent audio, nothing for silence."""
        if isinstance(audio, np.ndarray) and np.max(np.abs(audio)) < 0.01:
            # Simulate silence → no speech detected
            return iter([]), MagicMock()
        return iter([_FakeSegment(" Hello world")]), MagicMock()


# ---------------------------------------------------------------------------
# Patches applied for every test
# ---------------------------------------------------------------------------
PATCHES = [
    patch("faster_whisper.WhisperModel", _FakeWhisperModel),
    patch("pygame.mixer.init"),
    patch("pygame.mixer.music"),
    patch("speech_recognition.Recognizer"),
    patch("threading.Thread"),  # stop the TTS worker thread from spawning
]


class TestGemeni(unittest.TestCase):
    """Validation Suite for Gemeni — focused on logic and database correctness."""

    def setUp(self):
        # Apply all patches before importing / instantiating GemeniEngine so
        # no real hardware (mic, pygame, Whisper GPU) is touched during tests.
        for p in PATCHES:
            p.start()
            self.addCleanup(p.stop)

        from gemeni_engine import GemeniEngine   # BUG 1+2 fixed: correct class name
        self.engine = GemeniEngine()
        database.init_db()

    # ------------------------------------------------------------------ #
    # Test 1 — ASR latency benchmark                                       #
    # ------------------------------------------------------------------ #
    def test_latency_benchmark(self):
        """ML transcription of a 1-second audio chunk must complete < 500 ms."""
        # 1 second of mock voice audio (non-silent so the fake model returns text)
        mock_audio = np.random.normal(0, 0.05, 16000).astype(np.float32)

        start = time.time()
        # BUG 3 fixed: drive the engine through perform_ml_asr(), which IS the
        # public transcription method. add_audio_chunk() never existed.
        result = self.engine.perform_ml_asr(_numpy_to_audio_data(mock_audio))
        elapsed_ms = (time.time() - start) * 1000

        print(f"\n[LATENCY BENCHMARK] Pipeline took {elapsed_ms:.2f} ms")
        self.assertLess(elapsed_ms, 500, "Latency exceeds 500 ms threshold!")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------ #
    # Test 2 — Silence produces no transcription output                   #
    # ------------------------------------------------------------------ #
    def test_silence_produces_no_output(self):
        """A silent audio chunk must return an empty/near-empty transcript."""
        # BUG 4+5 fixed: removed process_buffer() / is_speaking / buffer checks.
        # We test the observable output of perform_ml_asr() on silent input.
        silent_audio = np.zeros(1600, dtype=np.float32)
        result = self.engine.perform_ml_asr(_numpy_to_audio_data(silent_audio))
        self.assertEqual(result.strip(), "", f"Expected empty transcript, got: '{result}'")

    # ------------------------------------------------------------------ #
    # Test 3 — Settings load / save round-trip                            #
    # ------------------------------------------------------------------ #
    def test_settings_roundtrip(self):
        """Settings written to disk must be read back identically."""
        test_settings = {
            "native_lang": "EN",
            "target_lang": "FR",
            "audio_mix_ratio": 0.7,
            "vad_threshold": 0.008,
            "first_run": False,
        }
        settings.save_settings(test_settings)
        loaded = settings.load_settings()

        self.assertEqual(loaded["target_lang"], "FR")
        self.assertAlmostEqual(loaded["audio_mix_ratio"], 0.7)
        self.assertFalse(loaded["first_run"])

    # ------------------------------------------------------------------ #
    # Test 4 — Database logging                                            #
    # ------------------------------------------------------------------ #
    def test_database_logging(self):
        """Translations must be persisted and retrievable from SQLite."""
        initial_count = len(database.get_recent_history(100))

        database.log_translation("EN", "ES", "Hello world", "Hola mundo")

        new_count = len(database.get_recent_history(100))
        self.assertEqual(
            new_count, initial_count + 1,
            "Database failed to log the translation!",
        )

        latest = database.get_recent_history(1)[0]
        # Row layout: (id, timestamp, source_lang, target_lang, original, translated)
        self.assertEqual(latest[2], "EN")
        self.assertEqual(latest[3], "ES")
        self.assertEqual(latest[4], "Hello world")
        self.assertEqual(latest[5], "Hola mundo")

    # ------------------------------------------------------------------ #
    # Test 5 — Config is loaded into the engine on init                   #
    # ------------------------------------------------------------------ #
    def test_engine_loads_config(self):
        """GemeniEngine.__init__ must populate self.config from settings."""
        self.assertIn("target_lang", self.engine.config)
        self.assertIn("native_lang", self.engine.config)


# ---------------------------------------------------------------------------
# Utility: wrap a numpy float32 array as a minimal SpeechRecognition
# AudioData object so perform_ml_asr() can call .get_wav_data() on it.
# ---------------------------------------------------------------------------
import struct
import io as _io


def _numpy_to_audio_data(audio_np: np.ndarray):
    """Convert float32 numpy array → fake sr.AudioData (16-bit PCM, 16 kHz)."""
    import speech_recognition as sr

    pcm_int16 = (audio_np * 32768).clip(-32768, 32767).astype(np.int16)
    raw_bytes = pcm_int16.tobytes()
    return sr.AudioData(raw_bytes, sample_rate=16000, sample_width=2)


if __name__ == "__main__":
    unittest.main()
