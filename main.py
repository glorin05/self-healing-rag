"""
main.py — CLI Entry Point for Self-Healing RAG

Supports both interactive and single question mode:
    python main.py                    (interactive mode)
    python main.py -q "What is RAG?"  (single question mode)
"""

import os
import sys
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from dotenv import load_dotenv

# Load environment
load_dotenv()


def check_prerequisites():
    """Verify setup before starting."""
    errors = []
    
    if not os.getenv("GROQ_API_KEY"):
        errors.append("❌ GROQ_API_KEY missing. Add to .env file.")
    
    if not os.path.exists("vectorstore"):
        errors.append("❌ Run: python src/ingest.py")
    
    if errors:
        print("\n" + "=" * 60)
        print("  SETUP ISSUES")
        print("=" * 60)
        for e in errors:
            print(f"\n{e}")
        print("\n" + "=" * 60 + "\n")
        return False
    
    return True


def run_query(graph, question: str) -> dict:
    """Run question through self-healing RAG pipeline."""
    initial_state = {
        "question": question,
        "rephrased_question": "",
        "documents": [],
        "relevance_scores": [],
        "generation": "",
        "grade": "",
        "retry_count": 0,
        "confidence": "",
        "healing_log": [],
        "source_metadata": [],
        "previous_rewrites": [],
    }
    
    print("\n" + "─" * 60)
    print("🚀 Running Self-Healing RAG...")
    print("─" * 60)
    
    return graph.invoke(initial_state)


def print_result(result: dict) -> None:
    """Display results with proper source info."""
    print(f"\n{'='*60}")
    print(f"  SELF-HEALING RAG — RESULT")
    print(f"{'='*60}\n")
    
    # Answer
    print(f"📝  Answer:\n{result['generation']}\n")
    
    # Confidence
    confidence = result.get("confidence", "LOW")
    emoji = "🟢" if confidence == "HIGH" else "🟡" if confidence == "MEDIUM" else "🔴"
    print(f"{emoji}  Confidence: {confidence}")
    
    # Healing attempts
    retries = result.get("retry_count", 0)
    if retries == 0:
        print(f"✅  Healing: Not needed (first try)")
    else:
        print(f"🔄  Healing attempts: {retries}")
    
    # Sources with proper info
    metadatas = result.get("source_metadata", [])
    scores = result.get("relevance_scores", [])
    if metadatas:
        print(f"\n📚  Sources ({len(metadatas)} chunks used):")
        for i, (meta, score) in enumerate(zip(metadatas, scores)):
            source = os.path.basename(meta.get("source", "unknown"))
            chunk = meta.get("chunk_index", i)
            print(f"   {i+1}. {source} — chunk #{chunk} (relevance: {score:.3f})")
    
    print(f"\n{'='*60}\n")


def interactive_mode(graph):
    """Interactive REPL loop."""
    print("\n" + "=" * 60)
    print("  🧬 SELF-HEALING RAG SYSTEM")
    print("  LangGraph + Groq + ChromaDB")
    print("=" * 60)
    print("\n  Commands:")
    print("    • Type a question")
    print("    • 'quit' to exit")
    print("    • 'ingest' to re-index")
    print()
    
    if not check_prerequisites():
        sys.exit(1)
    
    while True:
        try:
            question = input("💬 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break
        
        if question.lower() == "ingest":
            from src.ingest import ingest_documents
            ingest_documents()
            continue
        
        try:
            result = run_query(graph, question)
            print_result(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Check: API key, documents, run ingest.py\n")


def single_question_mode(graph, question: str):
    """Run a single question and exit."""
    if not check_prerequisites():
        sys.exit(1)
    
    try:
        result = run_query(graph, question)
        print_result(result)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check: API key, documents, run ingest.py\n")
        sys.exit(1)


def main():
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Self-Healing RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode
  python main.py -q "What is RAG?"  # Single question mode
  python main.py --question "Tell me about RAG"  # Long form
        """
    )
    
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Single question mode (implies --non-interactive)"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force interactive mode even if -q is provided"
    )
    
    args = parser.parse_args()
    
    # Build graph
    from src.graph import build_graph
    graph = build_graph()
    
    # Determine mode
    if args.question and not args.interactive:
        # Single question mode
        single_question_mode(graph, args.question)
    else:
        # Interactive mode
        interactive_mode(graph)


if __name__ == "__main__":
    main()