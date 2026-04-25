"""
src/ingest.py — Document Ingestion Pipeline

Uses ChromaDB for proper vector similarity search.
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_PATH = "./vectorstore"
COLLECTION_NAME = "rag_store"


def ingest_documents():
    """Load docs → split → embed → store in ChromaDB."""
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
    
    # Load documents
    print("\n📖 Loading documents...")
    docs = []
    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
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
    
    # Store in ChromaDB
    print(f"\n💾 Storing in ChromaDB...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Delete existing collection (fresh ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   🗑️  Cleared existing collection")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare data
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": c.metadata.get("source", "unknown"), 
         "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"   ✅ Stored {len(chunks)} chunks in ChromaDB")
    print(f"   📁 Collection: {COLLECTION_NAME}")
    print(f"   📁 Path: {CHROMA_PATH}")
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ INGESTION COMPLETE")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    ingest_documents()