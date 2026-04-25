"""
src/ingest.py — Document Ingestion Pipeline

Uses a pure-Python JSON vector store instead of ChromaDB to avoid
Windows Application Control blocking the Rust native bindings.

This is functionally identical to a real vector DB — we embed chunks,
store them with metadata, and save to a JSON file. The retrieve node
then loads this file and computes cosine similarity at query time.

Usage:
    python -m src.ingest
"""

import os
import sys
import json
import hashlib

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
VECTORSTORE_DIR = "./vectorstore"
STORE_FILE = os.path.join(VECTORSTORE_DIR, "store.json")


def ingest_documents():
    """Load docs → split → embed → store in JSON vector store."""
    print("=" * 60)
    print("  DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    # Check for documents
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        print(f"\n📁 Created docs/ folder at: {DOCS_DIR}")
        print("   Drop PDF/TXT files there, then re-run.")
        return

    files = [f for f in os.listdir(DOCS_DIR)
             if os.path.isfile(os.path.join(DOCS_DIR, f))
             and f.lower().endswith((".pdf", ".txt"))]

    if not files:
        print(f"\n❌ No documents found in {DOCS_DIR}/")
        print("   Add PDF/TXT files and run again.")
        return

    print(f"\n📄 Found {len(files)} document(s):")
    for f in files:
        print(f"   • {f}")

    # Load documents using LangChain loaders
    print("\n📖 Loading documents...")
    docs = []
    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = filename
            docs.extend(loaded)
            print(f"   ✅ Loaded: {filename}")
        except Exception as e:
            print(f"   ❌ Failed: {filename}: {e}")

    if not docs:
        print("\n❌ No documents loaded.")
        return

    print(f"   Total: {len(docs)} pages/sections")

    # Split into chunks
    print(f"\n✂️  Splitting into chunks (size=500, overlap=50)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"   ✅ Created {len(chunks)} chunks")

    # Generate embeddings
    print(f"\n🔢 Generating embeddings (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c.page_content for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
    print(f"   ✅ Embedded {len(embeddings)} chunks")

    # Build JSON vector store
    print(f"\n💾 Saving to JSON vector store...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    store = {"chunks": []}
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Content-based ID prevents duplicates if re-run
        chunk_id = hashlib.md5(
            f"{chunk.metadata.get('source', 'unknown')}:{i}:{chunk.page_content[:50]}".encode()
        ).hexdigest()

        store["chunks"].append({
            "id": chunk_id,
            "document": chunk.page_content,
            "embedding": embedding,
            "metadata": {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": i,
            }
        })

    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f)

    print(f"   ✅ Stored {len(chunks)} chunks")
    print(f"   📁 Path: {STORE_FILE}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ INGESTION COMPLETE")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    ingest_documents()