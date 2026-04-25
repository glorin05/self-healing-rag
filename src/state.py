"""
src/state.py — GraphState TypedDict

Added previous_rewrites field for tracking rewrite attempts.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    question: str
    rephrased_question: str
    documents: List[str]
    relevance_scores: List[float]
    generation: str
    grade: str
    retry_count: int
    confidence: str
    healing_log: List[dict]
    source_metadata: List[dict]
    previous_rewrites: List[str]