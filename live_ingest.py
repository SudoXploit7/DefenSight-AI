# live_ingest.py (FIXED - Dynamic Collection Loading)
#
# - Can reindex all existing files in normalized/  (one-shot)
# - Can watch incoming_logs/ and index new logs in real-time
# - Uses the same embedding model as rag_engine.py:
#       "multi-qa-mpnet-base-dot-v1"  (768-dim)

import os
import sys
import json
import time
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from format_con import normalize_file  # converts raw -> normalized JSON

# ====== SETTINGS ======
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"
VECTOR_DB_PATH = "./DefenSight AI_db"
NORMALIZED_DIR = "./normalized"
INCOMING_DIR = "./incoming_logs"
BATCH_SIZE = 64  # for encoding

os.makedirs(NORMALIZED_DIR, exist_ok=True)
os.makedirs(INCOMING_DIR, exist_ok=True)

print(f"🔍 Using embedding model: {EMBED_MODEL}")

model = SentenceTransformer(EMBED_MODEL)
client = PersistentClient(path=VECTOR_DB_PATH)

# ====== DYNAMIC COLLECTION GETTER ======
def get_collection():
    """Get or create collection - always fresh reference to avoid stale collection errors"""
    return client.get_or_create_collection("defensight_ai")

# ====== HELPERS ======

def get_text(entry: dict) -> str:
    """Prefer description, fall back to raw, else JSON dump."""
    return entry.get("description") or entry.get("raw") or json.dumps(entry)


def clean_metadata(meta: dict) -> dict:
    """Ensure metadata contains only JSON-serializable primitives."""
    result = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
        elif v is None:
            result[k] = "null"
        else:
            result[k] = str(v)
    return result


def index_entries(entries, source_id: str):
    """
    Index a list of dict entries from a normalized file.
    source_id: typically the normalized filename.
    """
    if not entries:
        return

    # ✅ Get fresh collection reference to avoid stale collection after session clear
    collection = get_collection()

    # Normalize to list
    data = entries if isinstance(entries, list) else [entries]

    documents = []
    metadatas = []
    ids = []

    for idx, entry in enumerate(data):
        text = get_text(entry)
        if not text:
            continue

        meta = dict(entry)  # copy
        meta.setdefault("source_file", source_id)
        meta.setdefault("type", entry.get("type", "unknown"))
        meta.setdefault("timestamp", entry.get("timestamp", datetime.now().isoformat()))
        meta = clean_metadata(meta)

        doc_id = entry.get("id") or f"{source_id}-{idx}"
        ids.append(doc_id)
        documents.append(text)
        metadatas.append(meta)

    if not documents:
        return

    # Encode in batches
    for start in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[start:start + BATCH_SIZE]
        batch_ids = ids[start:start + BATCH_SIZE]
        batch_metas = metadatas[start:start + BATCH_SIZE]

        embeddings = model.encode(batch_docs, batch_size=BATCH_SIZE, show_progress_bar=False)

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings.tolist(),
            metadatas=batch_metas,
        )

    print(f"✅ Indexed {len(documents)} entries from {source_id}")


def index_normalized_file(normalized_path: str):
    """Load a normalized JSON file and index all entries."""
    filename = os.path.basename(normalized_path)

    if not os.path.isfile(normalized_path):
        print(f"⚠️ Not a file: {normalized_path}")
        return

    if not filename.lower().endswith(".json"):
        print(f"⚠️ Skipping non-JSON normalized file: {filename}")
        return

    try:
        with open(normalized_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load normalized file {filename}: {e}")
        return

    index_entries(data, source_id=filename)


def reindex_all_normalized():
    """
    One-shot reindex of everything in ./normalized.
    Run with:
        python live_ingest.py --reindex
    after you delete the old DB.
    """
    files = [
        f for f in os.listdir(NORMALIZED_DIR)
        if f.lower().endswith(".json")
    ]
    if not files:
        print("ℹ️ No normalized JSON files found to index.")
        return

    print(f"📦 Reindexing {len(files)} normalized files from {NORMALIZED_DIR}/ ...")
    for fname in files:
        path = os.path.join(NORMALIZED_DIR, fname)
        print(f"➡  {fname}")
        index_normalized_file(path)
    print("💾 Reindex complete.")


# ====== REAL-TIME INGEST ======

class LogHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print(f"📥 New raw log detected: {event.src_path}")

        # Normalize into ./normalized
        normalized_path = normalize_file(event.src_path)
        print(f"🧾 Normalized to: {normalized_path}")

        # Index normalized JSON into Chroma
        index_normalized_file(normalized_path)
        print(f"✅ Indexed: {event.src_path}")


def start_realtime_ingestion():
    event_handler = LogHandler()
    observer = Observer()
    observer.schedule(event_handler, INCOMING_DIR, recursive=False)
    observer.start()

    print("🔄 Real-time ingestion started.")
    print("   Watching folder:", os.path.abspath(INCOMING_DIR))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # Usage:
    #   python live_ingest.py          -> just start watcher
    #   python live_ingest.py --reindex -> reindex normalized/ once, then exit
    if len(sys.argv) > 1 and sys.argv[1] == "--reindex":
        reindex_all_normalized()
    else:
        start_realtime_ingestion()