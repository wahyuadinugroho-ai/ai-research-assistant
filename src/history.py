import json
import os
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HISTORY_DIR = str(_PROJECT_ROOT / "chat_history")
ARCHIVE_DIR = str(_PROJECT_ROOT / "chat_archive")


def ensure_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)


def save_session(session_id, messages, title="Chat Baru"):
    ensure_dir()
    clean_messages = []
    for msg in messages:
        clean_messages.append({"role": msg["role"], "content": msg["content"]})

    data = {"title": title, "messages": clean_messages}

    # If the session is currently in the archive directory, update it in archive
    # otherwise write to active history directory
    arc_path = os.path.join(ARCHIVE_DIR, f"{session_id}.json")
    hist_path = os.path.join(HISTORY_DIR, f"{session_id}.json")

    target_path = (
        arc_path
        if os.path.exists(arc_path) and not os.path.exists(hist_path)
        else hist_path
    )

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(session_id):
    ensure_dir()
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        path = os.path.join(ARCHIVE_DIR, f"{session_id}.json")

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return "Chat Lama", data
                return data.get("title", "Chat Baru"), data.get("messages", [])
        except Exception:
            return "Chat Baru", []
    return "Chat Baru", []


def list_sessions():
    ensure_dir()
    return _list_sessions_from_dir(HISTORY_DIR)


def list_archived_sessions():
    ensure_dir()
    return _list_sessions_from_dir(ARCHIVE_DIR)


def _list_sessions_from_dir(directory):
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    sessions = []
    for f in files:
        s_id = f.replace(".json", "")
        try:
            with open(os.path.join(directory, f), "r", encoding="utf-8") as file:
                data = json.load(file)
                title = (
                    data.get("title", s_id) if isinstance(data, dict) else "Chat Lama"
                )
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
        os.replace(src, dst)


def unarchive_session(session_id):
    ensure_dir()
    src = os.path.join(ARCHIVE_DIR, f"{session_id}.json")
    dst = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(src):
        os.replace(src, dst)


def export_session_to_markdown(title, messages):
    """Format session conversation into a downloadable Markdown string."""
    lines = [f"# 🔬 {title}", "", "---", ""]
    for msg in messages:
        role_label = (
            "👤 **User**"
            if msg.get("role") == "user"
            else "🤖 **AI Research Assistant**"
        )
        lines.append(role_label)
        lines.append("")
        lines.append(msg.get("content", ""))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
