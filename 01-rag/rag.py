"""RAG over your own docs — grounded answers with citations, powered by the Claude API.

Pipeline:
    load → chunk → embed (local MiniLM) → Chroma (vector store) → search → answer (Claude)

The embedding + retrieval run locally (free). Only the final answer calls the Claude API,
and the model is instructed to answer *strictly* from the retrieved context and to cite
every source — or to say it does not know.

CLI:
    python rag.py build                      # index the rag-mini-wikipedia corpus
    python rag.py build --source local --folder ./my_docs
    python rag.py ask "When was ... ?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).parent
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "docs"

EMBED_MODEL = "all-MiniLM-L6-v2"          # tiny, fast, runs on a laptop CPU
# Claude model that writes the final grounded answer. Cheap + fast by default;
# override with CLAUDE_MODEL=claude-sonnet-5-5 for higher quality.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")

CHUNK_SIZE = 400                           # words per chunk (~300-500 tokens)
CHUNK_OVERLAP = 50
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a question-answering assistant that answers ONLY from the provided context "
    "passages. Every factual claim must cite its source inline as [source]. "
    "If the answer cannot be found in the context, reply exactly: "
    '"I don\'t know based on the provided documents." '
    "Never use outside knowledge and never invent sources."
)

_EMBEDDER = None


def get_embedder():
    """Load (once) and return the local sentence-transformers model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


# --------------------------------------------------------------------------- #
# 1. Loading                                                                   #
# --------------------------------------------------------------------------- #
def load_wikipedia(limit=None):
    """Yield (source, text) from the rag-mini-wikipedia passage corpus (Hugging Face)."""
    from datasets import load_dataset

    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield f"wiki-{row.get('id', i)}", row["passage"]


def load_local(folder):
    """Yield (source, text) from .txt / .md / .pdf files in a folder (recursively)."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel = str(path.relative_to(folder))
        if suffix in {".txt", ".md"}:
            yield rel, path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                print(f"  ! skipping {rel} (install pypdf to read PDFs)")
                continue
            reader = PdfReader(str(path))
            yield rel, "\n".join(page.extract_text() or "" for page in reader.pages)


# --------------------------------------------------------------------------- #
# 2. Chunking                                                                  #
# --------------------------------------------------------------------------- #
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------- #
# 3. Indexing                                                                  #
# --------------------------------------------------------------------------- #
def build_index(source="wikipedia", folder=None, limit=None, reset=True):
    """Embed documents and store them in a persistent Chroma collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    docs = load_wikipedia(limit) if source == "wikipedia" else load_local(folder)

    ids, texts, metas = [], [], []
    for source_name, text in docs:
        for j, piece in enumerate(chunk_text(text)):
            ids.append(f"{source_name}::chunk-{j}")
            texts.append(piece)
            metas.append({"source": source_name, "chunk": j})

    if not texts:
        raise SystemExit("No documents to index. Check your source/folder.")

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    embeddings = get_embedder().encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    batch = 2000  # Chroma caps the number of items per add()
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metas[i : i + batch],
        )
    print(f"Indexed {len(ids)} chunks into {CHROMA_DIR}")
    return len(ids)


# --------------------------------------------------------------------------- #
# 4. Retrieval                                                                 #
# --------------------------------------------------------------------------- #
def search(question, k=TOP_K):
    """Return the top-k most similar chunks for a question."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    q_emb = get_embedder().encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)

    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "chunk": meta.get("chunk"),
                "score": round(1 - dist, 4),  # cosine distance → similarity
            }
        )
    return hits


# --------------------------------------------------------------------------- #
# 5. Answering (Claude API)                                                    #
# --------------------------------------------------------------------------- #
def answer(question, hits=None, k=TOP_K):
    """Generate a grounded, cited answer. Returns (answer_text, hits)."""
    from anthropic import Anthropic

    if hits is None:
        hits = search(question, k)
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return message.content[0].text.strip(), hits


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(HERE / ".env")
    except ImportError:
        pass


def main():
    _load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the vector index")
    b.add_argument("--source", choices=["wikipedia", "local"], default="wikipedia")
    b.add_argument("--folder", help="folder of docs when --source local")
    b.add_argument("--limit", type=int, help="max documents (for a quick test)")

    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=TOP_K, help="number of chunks to retrieve")

    args = parser.parse_args()

    if args.command == "build":
        build_index(source=args.source, folder=args.folder, limit=args.limit)
    elif args.command == "ask":
        text, hits = answer(args.question, k=args.k)
        print("\n" + text + "\n")
        print("─" * 60)
        print("Retrieved from:")
        for h in hits:
            print(f"  • {h['source']} (chunk {h['chunk']}, score {h['score']})")


if __name__ == "__main__":
    main()
