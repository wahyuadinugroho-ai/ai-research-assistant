import json
import os

HISTORY_DIR = "chat_history"
ARCHIVE_DIR = "chat_archive"

def ensure_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)

def save_session(session_id, messages, title="Chat Baru"):
    ensure_dir()
    clean_messages = []
    for msg in messages:
        clean_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
        
    data = {"title": title, "messages": clean_messages}
    with open(os.path.join(HISTORY_DIR, f"{session_id}.json"), "w") as f:
        json.dump(data, f)

def load_session(session_id):
    ensure_dir()
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        # Try archive dir
        path = os.path.join(ARCHIVE_DIR, f"{session_id}.json")
        
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return "Chat Lama", data
            return data.get("title", "Chat Baru"), data.get("messages", [])
    return "Chat Baru", []

def list_sessions():
    ensure_dir()
    return _list_sessions_from_dir(HISTORY_DIR)

def list_archived_sessions():
    ensure_dir()
    return _list_sessions_from_dir(ARCHIVE_DIR)

def _list_sessions_from_dir(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    sessions = []
    for f in files:
        s_id = f.replace(".json", "")
        with open(os.path.join(directory, f), "r") as file:
            try:
                data = json.load(file)
                title = data.get("title", s_id) if isinstance(data, dict) else "Chat Lama"
                sessions.append((s_id, title))
            except Exception:
                pass
    return sessions

def delete_session(session_id):
    ensure_dir()
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
    arc_path = os.path.join(ARCHIVE_DIR, f"{session_id}.json")
    if os.path.exists(arc_path):
        os.remove(arc_path)

def archive_session(session_id):
    ensure_dir()
    src = os.path.join(HISTORY_DIR, f"{session_id}.json")
    dst = os.path.join(ARCHIVE_DIR, f"{session_id}.json")
    if os.path.exists(src):
        os.rename(src, dst)

def unarchive_session(session_id):
    ensure_dir()
    src = os.path.join(ARCHIVE_DIR, f"{session_id}.json")
    dst = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(src):
        os.rename(src, dst)
