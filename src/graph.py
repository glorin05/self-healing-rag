"""
src/graph.py — LangGraph StateGraph Definition

This module wires all 7 nodes together into an executable graph.

SELF-HEALING FLOW DIAGRAM:

User Query
    ↓
retrieve (ChromaDB search)
    ↓
grade_documents (are chunks relevant?)
    ↓ YES                    ↓ NO
generate              rewrite_query ←──────────────┐
    ↓                       ↓                       │
grade_answer          retrieve (retry)              │
    ↓ YES    ↓ NO           ↓                       │
finish   rewrite_query  grade_documents             │
             └──────────────────────────────────────┘
             (if retries > 2 → fallback)
"""

from langgraph.graph import StateGraph, END

from src.state import GraphState
from src.nodes import (
    retrieve,
    grade_documents,
    rewrite_query,
    generate,
    grade_answer,
    fallback,
    finish,
    route_after_grading_docs,
    route_after_grading_answer,
    route_after_rewrite,
)


def build_graph():
    """
    Build and compile the self-healing RAG state graph.
    
    Returns a compiled LangGraph that can be invoked with:
        result = graph.invoke({"question": "...", ...})
    
    ARCHITECTURE:
      - Each node is a pure function: (state) → partial_state
      - Conditional edges use routing functions to decide next node
      - The graph automatically handles state merging
      - END terminates execution
    """
    # Create the graph with our state schema
    workflow = StateGraph(GraphState)
    
    # Add all 7 nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_answer", grade_answer)
    workflow.add_node("fallback", fallback)
    workflow.add_node("finish", finish)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Linear edges (always go to next node)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("generate", "grade_answer")
    workflow.add_edge("finish", END)
    workflow.add_edge("fallback", END)
    
    # Conditional edge: grade_documents → generate OR rewrite_query
    # This is the first quality gate
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading_docs,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        }
    )
    
    # Conditional edge: grade_answer → finish OR rewrite_query OR fallback
    # This is the second quality gate
    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grading_answer,
        {
            "finish": "finish",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        }
    )
    
    # Conditional edge: rewrite_query → retrieve OR fallback
    # This creates the healing loop
    workflow.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        }
    )
    
    # Compile and return
    compiled = workflow.compile()
    print("✅ Self-Healing RAG graph compiled successfully!")
    
    return compiled