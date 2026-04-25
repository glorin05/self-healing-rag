"""
scripts/health_check.py — Verify system is ready

Run this to check if the system is properly configured:
    python scripts/health_check.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check():
    """Check all prerequisites for the system to work."""
    errors = []
    warnings = []
    
    print("=" * 50)
    print("  SELF-HEALING RAG - HEALTH CHECK")
    print("=" * 50)
    
    # Check Groq API key
    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY not set in .env")
    else:
        print("[OK] GROQ_API_KEY found")
    
    # Check vectorstore exists
    if not os.path.exists("vectorstore"):
        errors.append("Vector store missing - run: python src/ingest.py")
    else:
        print("[OK] Vector store exists")
        
        # Check if store has data
        try:
            import json
            for fname in os.listdir("vectorstore"):
                if fname.endswith(".json"):
                    path = os.path.join("vectorstore", fname)
                    with open(path) as f:
                        data = json.load(f)
                    doc_count = len(data.get("documents", []))
                    print("   {} chunks indexed in {}".format(doc_count, fname))
        except Exception as e:
            warnings.append("Could not read vector store: {}".format(e))
    
    # Check docs directory
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        warnings.append("Created empty docs/ folder")
    
    doc_files = [f for f in os.listdir(docs_dir) if f.lower().endswith((".pdf", ".txt"))]
    if not doc_files:
        warnings.append("No documents in docs/ folder")
    else:
        print("[OK] {} document(s) in docs/".format(len(doc_files)))
        for f in doc_files:
            print("   - {}".format(f))
    
    # Check logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("[OK] Created logs/ folder")
    else:
        print("[OK] Logs folder exists")
    
    # Check src modules
    required_files = [
        "src/__init__.py",
        "src/state.py",
        "src/utils.py",
        "src/config.py",
        "src/ingest.py",
        "src/nodes.py",
        "src/graph.py",
    ]
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        errors.append("Missing files: {}".format(missing))
    else:
        print("[OK] All required source files present")
    
    # Report warnings
    for w in warnings:
        print("[WARN] {}".format(w))
    
    # Report errors
    if errors:
        print("\n--- ERRORS ---")
        for e in errors:
            print("[ERR] {}".format(e))
        return False
    
    print("\nSystem ready!")
    return True


if __name__ == "__main__":
    success = check()
    sys.exit(0 if success else 1)