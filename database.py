import sqlite3
from datetime import datetime

def init_db():
    """Initializes the Voxis translation history database."""
    conn = sqlite3.connect('voxis_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            source_lang TEXT,
            target_lang TEXT,
            original_text TEXT,
            translated_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_translation(source_lang, target_lang, original, translated):
    """Logs a successful translation to the database."""
    conn = sqlite3.connect('voxis_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO history 
                 (timestamp, source_lang, target_lang, original_text, translated_text) 
                 VALUES (?,?,?,?,?)''', 
              (timestamp, source_lang, target_lang, original, translated))
    conn.commit()
    conn.close()

def get_recent_history(limit=5):
    """Retrieves the most recent translations."""
    conn = sqlite3.connect('voxis_history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Voxis History Database Initialized.")
