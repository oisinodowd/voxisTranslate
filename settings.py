import json
import os

SETTINGS_FILE = 'voxis_settings.json'

def load_settings():
    """Loads user settings or returns defaults."""
    default_settings = {
        "native_lang": "EN",
        "target_lang": "ES",
        "audio_mix_ratio": 0.5,
        "vad_threshold": 0.005,
        "first_run": True
    }
    
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                return {**default_settings, **json.load(f)}
            except:
                return default_settings
    return default_settings

def save_settings(settings):
    """Saves user settings to a JSON file."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

if __name__ == "__main__":
    s = load_settings()
    print(f"Current Voxis Settings: {s}")
