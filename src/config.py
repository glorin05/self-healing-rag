"""
src/config.py — Configuration settings

All configurable values in one place. Can be overridden via environment variables.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Vector store settings
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
STORE_FILE = os.path.join(VECTORSTORE_DIR, "store.json")

# Similarity settings
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# LLM settings
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))