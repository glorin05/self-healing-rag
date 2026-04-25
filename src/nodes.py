"""
src/nodes.py — All 7 Node Functions for Self-Healing RAG

Uses a pure-Python JSON vector store instead of ChromaDB.
The retrieve node loads store.json, embeds the query, and computes
cosine similarity manually — which is actually great for learning
how vector databases work under the hood!

THE SELF-HEALING LOOP:
  1. retrieve     → fetch chunks from JSON vector store
  2. grade_docs   → are the chunks actually relevant?
  3. generate     → produce an answer from relevant chunks
  4. grade_answer → does the LLM trust its own answer?
  5. rewrite      → rephrase the query and try again
  6. fallback     → give up gracefully after max retries
  7. finish       → package the final answer with metadata
"""

import os
import sys
import json
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

from src.state import GraphState
from src.utils import cosine_similarity, log_healing_event

# Force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Load environment
load_dotenv()

# Import configuration
from src.config import (
    RELEVANCE_THRESHOLD,
    MAX_RETRIES,
    GROQ_MODEL,
    TEMPERATURE,
)

# LLM initialization
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=TEMPERATURE,
    api_key=os.getenv("GROQ_API_KEY"),
)


# =====================================================================
# JSON VECTOR STORE — Pure Python, no native bindings needed
# =====================================================================
_embed_model = None
_vector_store = None


def _get_embed_model():
    """Lazy-load the sentence-transformers embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_vector_store() -> List[dict]:
    """
    Lazy-load the JSON vector store from vectorstore/store.json.

    Each entry has:
      - "document": the chunk text
      - "embedding": the vector (list of floats)
      - "metadata": {"source": ..., "chunk_index": ...}
    """
    global _vector_store
    if _vector_store is None:
        store_path = os.path.join("vectorstore", "store.json")
        if not os.path.exists(store_path):
            print("⚠  Vector store not found. Run: python -m src.ingest")
            return []
        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _vector_store = data.get("chunks", [])
        print(f"   📦 Loaded {len(_vector_store)} chunks from JSON store")
    return _vector_store


def _search_store(query_embedding: List[float], top_k: int = 5) -> List[Tuple[dict, float]]:
    """
    Pure-Python vector search: compute cosine similarity between the
    query embedding and every stored chunk, return the top-k results.

    This is exactly what ChromaDB/Pinecone/Weaviate do under the hood —
    we're just doing it in plain Python so you can see how it works!
    """
    store = _get_vector_store()
    if not store:
        return []

    scored = []
    for entry in store:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored.append((entry, round(score, 4)))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# =====================================================================
# NODE 1: RETRIEVE
# =====================================================================
def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    🔍 RETRIEVE — Search JSON vector store for relevant chunks.

    HOW IT WORKS:
      1. Takes the current working query (rephrased_question or original)
      2. Embeds it using sentence-transformers
      3. Searches the JSON store for the top-5 nearest chunks
      4. Scores each chunk using cosine similarity
      5. Stores chunks + scores + metadata in state

    WHY THIS IS THE STARTING POINT:
      Every RAG pipeline begins with retrieval. The quality of
      retrieved chunks determines everything downstream.
    """
    query = state.get("rephrased_question") or state["question"]
    print(f"\n🔍 [RETRIEVE] Searching: \"{query}\"")

    # Embed query
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode(query).tolist()

    # Search our pure-Python vector store
    results = _search_store(query_embedding, top_k=5)

    if not results:
        print("   ❌ No results. Run: python -m src.ingest")
        return {"documents": [], "relevance_scores": [], "source_metadata": []}

    # Unpack results
    documents = [entry["document"] for entry, _ in results]
    scores = [score for _, score in results]
    source_metadata = []
    for entry, score in results:
        meta = entry.get("metadata", {})
        source_metadata.append({
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "score": score,
        })

    print(f"   Found {len(documents)} chunks, scores: {scores}")

    return {
        "documents": documents,
        "relevance_scores": scores,
        "source_metadata": source_metadata,
    }


# =====================================================================
# NODE 2: GRADE DOCUMENTS
# =====================================================================
def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    📊 GRADE DOCUMENTS — Filter by relevance threshold.

    Keeps only documents with score >= RELEVANCE_THRESHOLD (default 0.3).
    This is the first quality gate — it prevents the LLM from
    generating answers based on irrelevant context.
    """
    documents = state.get("documents", [])
    scores = state.get("relevance_scores", [])

    relevant_docs = []
    relevant_scores = []

    for doc, score in zip(documents, scores):
        if score >= RELEVANCE_THRESHOLD:
            relevant_docs.append(doc)
            relevant_scores.append(score)

    print(f"📊 [GRADE] {len(relevant_docs)}/{len(documents)} passed (threshold={RELEVANCE_THRESHOLD})")

    if relevant_docs:
        return {"documents": relevant_docs, "relevance_scores": relevant_scores}

    # Log healing event — no relevant chunks found
    healing_entry = {
        "event": "grade_documents",
        "reason": f"All {len(documents)} chunks below {RELEVANCE_THRESHOLD}",
        "scores": scores,
        "retry_count": state.get("retry_count", 0),
    }
    log_healing_event(healing_entry)

    current_log = state.get("healing_log", [])
    current_log.append(healing_entry)

    return {"documents": [], "relevance_scores": [], "healing_log": current_log}


# =====================================================================
# NODE 3: REWRITE QUERY
# =====================================================================
def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """
    ✏️ REWRITE QUERY — Generate unique rewrites each attempt.

    Passes previous_rewrites to the LLM so it generates different
    rephrases each time instead of repeating the same one.

    This is the core of the self-healing loop — the system ADAPTS
    its query strategy instead of just failing.
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    previous_rewrites = state.get("previous_rewrites", [])

    print(f"✏️  [REWRITE] Attempt {retry_count} — Rephrasing...")

    # Build prompt with previous attempts context
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a query rewriter improving document retrieval.
            Generate a UNIQUE rephrase each attempt.

            Guidelines:
            - Attempt 1: Use synonyms, different phrasing
            - Attempt 2: Break into simpler sub-concepts
            - Attempt 3: Use more technical/formal terms

            CRITICAL: Each rewrite must be SIGNIFICANTLY different
            from all previous attempts below.

            FAILED PREVIOUS REWRITES (do NOT repeat these):
            {previous}

            Return ONLY the rephrased question, nothing else."""
        ),
        (
            "human",
            "Original question: {question}"
        ),
    ])

    # Format previous attempts
    previous_str = "\n".join(f"- {p}" for p in previous_rewrites) if previous_rewrites else "None"

    # Invoke LLM
    chain = rewrite_prompt | llm
    rephrased = chain.invoke({
        "question": question,
        "previous": previous_str
    }).content.strip()

    print(f"   Original:  \"{question}\"")
    print(f"   Rephrased: \"{rephrased}\"")

    # Update previous rewrites
    new_previous = previous_rewrites + [rephrased]

    # Log healing event
    healing_entry = {
        "event": "rewrite_query",
        "reason": "retrieval/answer grading failed",
        "retry_count": retry_count,
        "original": question,
        "rephrased": rephrased,
    }
    log_healing_event(healing_entry)

    current_log = state.get("healing_log", [])
    current_log.append(healing_entry)

    return {
        "rephrased_question": rephrased,
        "retry_count": retry_count,
        "previous_rewrites": new_previous,
        "healing_log": current_log,
    }


# =====================================================================
# NODE 4: GENERATE
# =====================================================================
def generate(state: GraphState) -> Dict[str, Any]:
    """
    🤖 GENERATE — Create answer strictly from context chunks.

    We explicitly tell the LLM to answer ONLY from the provided
    context. This reduces hallucination. If the context doesn't
    contain the answer, the LLM should say so — and the grade_answer
    node will catch it.
    """
    query = state.get("rephrased_question") or state["question"]
    documents = state.get("documents", [])

    print(f"🤖 [GENERATE] Building from {len(documents)} chunks...")

    # Build context
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {doc}" for i, doc in enumerate(documents)
    )

    # Generation prompt
    gen_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer ONLY from the "
            "provided context. If insufficient, say so clearly. "
            "Do not make up information."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    ])

    chain = gen_prompt | llm
    answer = chain.invoke({"context": context, "question": query}).content.strip()

    print(f"   Preview: \"{answer[:80]}...\"")

    return {"generation": answer}


# =====================================================================
# NODE 5: GRADE ANSWER
# =====================================================================
def grade_answer(state: GraphState) -> Dict[str, Any]:
    """
    ✅ GRADE ANSWER — LLM judges its own answer.

    This is the second self-healing quality gate. The LLM evaluates
    whether its generated answer is actually supported by the context.

    THIS IS THE "SELF" IN SELF-HEALING:
      The system doesn't need a human to say "bad answer, try again."
      It judges itself and autonomously decides to retry.
    """
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    print("✅ [GRADE_ANSWER] Evaluating response...")

    context = "\n\n".join(documents)

    # Grading prompt — binary YES/NO
    grade_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader. Answer YES if the answer is "
            "directly supported by the context. "
            "Answer NO if it contains information not in context. "
            "Reply with ONLY YES or NO."
        ),
        (
            "human",
            "Context:\n{context}\n\nAnswer: {answer}\n\nSupported?"
        ),
    ])

    chain = grade_prompt | llm
    grade = chain.invoke({"context": context, "answer": generation}).content.strip().upper()

    is_supported = "YES" in grade
    print(f"   Grade: {grade} → {'PASS ✓' if is_supported else 'FAIL ✗'}")

    if not is_supported:
        healing_entry = {
            "event": "grade_answer",
            "reason": f"unsupported answer (grade: {grade})",
            "retry_count": state.get("retry_count", 0),
        }
        log_healing_event(healing_entry)

        current_log = state.get("healing_log", [])
        current_log.append(healing_entry)
        return {"grade": grade, "healing_log": current_log}

    return {"grade": grade}


# =====================================================================
# NODE 6: FALLBACK
# =====================================================================
def fallback(state: GraphState) -> Dict[str, Any]:
    """
    ⚠️ FALLBACK — Graceful failure after exhausting retries.

    Instead of crashing, the user gets actionable feedback about
    what they can do to get better results.
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    print(f"⚠️  [FALLBACK] Failed after {retry_count} attempts.")

    fallback_msg = (
        f"I couldn't find a reliable answer to: '{question}'.\n\n"
        f"This could mean:\n"
        f"1. Relevant documents not in /docs folder\n"
        f"2. Question outside document scope\n\n"
        f"Suggestions:\n"
        f"- Add relevant PDFs/TXTs to docs/\n"
        f"- Run: python -m src.ingest\n"
        f"- Try rephrasing your question"
    )

    healing_entry = {"event": "fallback", "reason": f"max retries ({retry_count})", "question": question}
    log_healing_event(healing_entry)

    current_log = state.get("healing_log", [])
    current_log.append(healing_entry)

    return {"generation": fallback_msg, "confidence": "LOW", "healing_log": current_log}


# =====================================================================
# NODE 7: FINISH
# =====================================================================
def finish(state: GraphState) -> Dict[str, Any]:
    """
    🏁 FINISH — Return with confidence metadata.

    HIGH  = answered on first try (no healing needed)
    MEDIUM = answered after 1 rewrite
    LOW   = answered after 2+ rewrites
    """
    retry_count = state.get("retry_count", 0)

    if retry_count == 0:
        confidence = "HIGH"
    elif retry_count == 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    print(f"🏁 [FINISH] Confidence: {confidence}")

    return {"confidence": confidence}


# =====================================================================
# ROUTING FUNCTIONS (used as conditional edges in the graph)
# =====================================================================
def route_after_grading_docs(state: GraphState) -> str:
    """After grade_documents: go to generate or rewrite."""
    return "generate" if state.get("documents") else "rewrite_query"


def route_after_grading_answer(state: GraphState) -> str:
    """After grade_answer: finish or rewrite/fallback."""
    grade = state.get("grade", "NO")
    retry_count = state.get("retry_count", 0)

    if "YES" in grade:
        return "finish"
    return "fallback" if retry_count >= MAX_RETRIES else "rewrite_query"


def route_after_rewrite(state: GraphState) -> str:
    """After rewrite_query: try again or fallback."""
    return "fallback" if state.get("retry_count", 0) > MAX_RETRIES else "retrieve"