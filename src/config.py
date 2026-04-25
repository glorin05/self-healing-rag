"""
src/config.py — Configuration settings

All configurable values in one place. Can be overridden via environment variables.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ChromaDB settings
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_store")

# Similarity settings
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# LLM settings
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))