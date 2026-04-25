"""
src/nodes.py — All 7 Node Functions for Self-Healing RAG

Fixed:
- retrieve uses ChromaDB for proper similarity search
- rewrite_query tracks previous attempts for unique rewrites
- JSON fallback when ChromaDB unavailable
- Config imported from src.config
"""

import os
import sys
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from sentence_transformers import SentenceTransformer

from src.state import GraphState
from src.utils import log_healing_event

# Load environment
load_dotenv()

# Import configuration from src.config
from src.config import (
    RELEVANCE_THRESHOLD,
    MAX_RETRIES,
    CHROMA_PATH,
    COLLECTION_NAME,
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
# CHROMADB HELPER FUNCTIONS
# =====================================================================
def _get_collection():
    """Get ChromaDB collection with JSON fallback."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return _get_json_fallback()


def _get_json_fallback():
    """Fallback to JSON store when ChromaDB unavailable."""
    path = f"./vectorstore/{COLLECTION_NAME}.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        class FakeCollection:
            def count(self):
                return len(data["documents"])
            
            def query(self, query_embeddings=None, n_results=5, include=None, **kwargs):
                docs = data["documents"][:n_results]
                scores = data.get("scores", [0.5] * len(docs))[:n_results]
                metas = data.get("metadatas", [{}] * len(docs))[:n_results]
                return {
                    "documents": [docs],
                    "distances": [[1 - s for s in scores]],
                    "metadatas": [metas]
                }
        
        print("⚠️  Using JSON fallback (ChromaDB unavailable)")
        return FakeCollection()
    
    print("❌ No vector store found. Run: python src/ingest.py")
    return None


def _get_embed_model():
    """Load sentence-transformers model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================================
# NODE 1: RETRIEVE
# =====================================================================
def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    🔍 RETRIEVE — Search ChromaDB for relevant chunks.
    
    FIXED: Uses ChromaDB directly for proper similarity search.
    """
    query = state.get("rephrased_question") or state["question"]
    print(f"\n🔍 [RETRIEVE] Searching: \"{query}\"")
    
    # Get ChromaDB collection
    collection = _get_collection()
    if collection is None:
        print("   ❌ No vector store. Run: python src/ingest.py")
        return {"documents": [], "relevance_scores": [], "source_metadata": []}
    
    # Check if collection has data
    if collection.count() == 0:
        print("   ❌ Empty vector store. Run: python src/ingest.py")
        return {"documents": [], "relevance_scores": [], "source_metadata": []}
    
    # Embed query
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode([query]).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )
    
    # Extract results
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    
    # Convert distance to similarity (1 - distance for cosine)
    similarity_scores = [round(1 - d, 4) for d in distances]
    
    print(f"   Found {len(documents)} chunks, scores: {similarity_scores}")
    
    return {
        "documents": documents,
        "relevance_scores": similarity_scores,
        "source_metadata": metadatas
    }


# =====================================================================
# NODE 2: GRADE DOCUMENTS
# =====================================================================
def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    📊 GRADE DOCUMENTS — Filter by relevance threshold.
    
    Keeps only documents with score >= 0.3.
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
    
    # Log healing event
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
    
    FIXED: Passes previous_rewrites to LLM so it generates
    different rewrites each time instead of repeating.
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
    🤖 GENERATE — Create answer from context chunks.
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
    
    This is the second self-healing quality gate.
    """
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    
    print("✅ [GRADE_ANSWER] Evaluating response...")
    
    context = "\n\n".join(documents)
    
    # Grading prompt
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
        f"- Run: python src/ingest.py\n"
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
# ROUTING FUNCTIONS
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