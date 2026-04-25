"""
src/utils.py — Utility functions for Self-Healing RAG

Contains:
- Cosine similarity calculation for scoring chunks
- Healing event logger that writes to logs/healing_log.json
"""

import math
import json
import os
from datetime import datetime, timezone


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns a value between -1 and 1:
      1  = identical direction (very similar)
      0  = orthogonal (unrelated)
     -1  = opposite direction
    
    This is used to score how relevant each retrieved chunk
    is to the user's question.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot / (mag_a * mag_b)


def log_healing_event(event: dict) -> None:
    """
    Append a healing event to logs/healing_log.json.
    
    This creates an audit trail of every self-healing attempt,
    useful for debugging and improving the system over time.
    
    Example event:
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "event": "grade_documents",
            "reason": "All chunks scored below 0.3",
            "retry_count": 1
        }
    """
    os.makedirs("logs", exist_ok=True)
    LOG_FILE = "logs/healing_log.json"
    
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Load existing log or start fresh
    existing = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []
    
    existing.append(event)
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)