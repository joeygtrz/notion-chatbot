import re
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Content loading ───────────────────────────────────────────────────────────

CONTENT_DIR = Path(__file__).parent / "content"

# Notion default/template pages to skip
SKIP_NAMES = {
    "get started", "example sub page", "home views",
    "my tasks", "untitled",
}

UUID_RE = re.compile(r'\s+[a-f0-9]{20,}$')


def clean_name(name: str) -> str:
    return UUID_RE.sub("", Path(name).stem).strip()


def load_documents() -> list[dict]:
    docs = []
    for md_file in CONTENT_DIR.rglob("*.md"):
        name = clean_name(md_file.name)
        if any(s in name.lower() for s in SKIP_NAMES):
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(text.strip()) < 150:
            continue
        parts = [clean_name(p) for p in md_file.relative_to(CONTENT_DIR).parts]
        source = " > ".join(p for p in parts if p)
        docs.append({"text": text, "source": source})
    return docs


def chunk_documents(docs: list[dict], size: int = 600, overlap: int = 80) -> list[dict]:
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        for i in range(0, len(words), size - overlap):
            window = words[i : i + size]
            if len(window) < 40:
                continue
            chunks.append({"text": " ".join(window), "source": doc["source"]})
    return chunks


# ── TF-IDF index ──────────────────────────────────────────────────────────────

class RAGIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        if not chunks:
            self.vectorizer = self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=15000, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not self.vectorizer:
            return []
        scores = cosine_similarity(
            self.vectorizer.transform([query]), self.matrix
        ).flatten()
        top = np.argsort(scores)[-k:][::-1]
        return [self.chunks[i] for i in top if scores[i] > 0.01]


# ── Startup ───────────────────────────────────────────────────────────────────

print("Loading Notion content…")
_docs = load_documents()
print(f"  {len(_docs)} documents loaded")
_chunks = chunk_documents(_docs)
print(f"  {len(_chunks)} chunks indexed")
_index = RAGIndex(_chunks)
print("  Ready.")

app = FastAPI()
# Resolves credentials automatically: ANTHROPIC_API_KEY if set, otherwise the
# profile saved by `ant auth login`.
_client = Anthropic()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text())


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_query = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )

    hits = _index.search(user_query)
    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in hits
    )

    system = (
        "You are a helpful assistant with access to the user's personal Notion notes. "
        "Answer questions using the excerpts provided. "
        "If the answer isn't in the notes, say so. Be concise.\n\n"
        f"EXCERPTS FROM NOTES:\n\n{context}"
        if context else
        "You are a helpful assistant. No relevant notes were found for this query."
    )

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    async def stream():
        try:
            with _client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system,
                messages=messages,
            ) as s:
                for text in s.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
