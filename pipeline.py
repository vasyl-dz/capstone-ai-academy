"""
Pipeline Orchestrator
Runs the complete RAG pipeline: parse → chunk → embed → chatbot → evaluate
"""

import argparse
from dotenv import load_dotenv
from steps import parse, chunk, embed, evaluate, chatbot

# Load environment variables from .env file
load_dotenv()


# Configuration
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
CHUNKS_FOLDER = "chunks"
DB_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"
WHISPER_MODEL = "base"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3.1:8b"  # Tool-calling capable Ollama model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_rag():
    """Execute steps 1-3: parse, chunk, and embed documents."""
    print("\n" + "🔨 " * 40)
    print("BUILDING RAG SYSTEM (Steps 1-3)")
    print("🔨 " * 40 + "\n")
    
    print("\n" + "=" * 80)
    print("STEP 1/3: PARSING")
    print("=" * 80 + "\n")
    
    pdf_files = parse.parse_pdfs(DATA_FOLDER, OUTPUT_FOLDER)
    audio_files = parse.parse_audio(DATA_FOLDER, OUTPUT_FOLDER, WHISPER_MODEL)
    
    total_parsed = len(pdf_files) + len(audio_files)
    
    if total_parsed == 0:
        print("\n✗ No files parsed. Please add PDFs or audio files to 'data/' folder.")
        return False
    
    print("\n" + "=" * 80)
    print("STEP 2/3: CHUNKING")
    print("=" * 80 + "\n")
    
    chunk_files = chunk.chunk_documents(OUTPUT_FOLDER, CHUNKS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP)
    
    if not chunk_files:
        print("\n✗ No chunks created.")
        return False
    
    print("\n" + "=" * 80)
    print("STEP 3/3: EMBEDDING")
    print("=" * 80 + "\n")
    
    chunks = embed.load_chunks(CHUNKS_FOLDER)
    embed.embed_and_store(chunks, DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL)
    
    print("\n✓ RAG system built successfully!\n")
    return True


def run_chatbot():
    """Execute step 4: launch interactive chatbot."""
    print("\n" + "💬 " * 40)
    print("RUNNING CHATBOT (Step 4)")
    print("💬 " * 40 + "\n")
    
    bot = chatbot.RAGChatbot(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL)
    bot.chat()


def run_evaluation():
    """Execute step 5: evaluate chatbot performance."""
    print("\n" + "📊 " * 40)
    print("RUNNING EVALUATION (Step 5)")
    print("📊 " * 40 + "\n")
    
    bot = chatbot.RAGChatbot(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL)
    evaluate.run_default_evaluation(bot.answer_question)


def run_full_pipeline():
    """Execute the complete RAG pipeline (steps 1-5)."""
    print("\n" + "🚀 " * 40)
    print("RAG PIPELINE WITH REASONING & TOOL CALLING")
    print("🚀 " * 40 + "\n")
    
    print("Pipeline steps:")
    print("  1. Parse documents (PDFs + audio/video)")
    print("  2. Chunk text into semantic segments")
    print("  3. Generate embeddings and store in database")
    print("  4. Launch interactive chatbot with reasoning & tool calling")
    
    # Steps 1-3: Build RAG
    if not build_rag():
        return
    
    # Step 4: Chatbot
    print("\n" + "=" * 80)
    print("STEP 4/4: CHATBOT WITH REASONING")
    print("=" * 80 + "\n")
    
    bot = chatbot.RAGChatbot(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL)
    bot.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Run different stages of the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py               # Run full pipeline (steps 1-4)
  python pipeline.py --build       # Build RAG system (steps 1-3)
  python pipeline.py --chatbot     # Run chatbot only (step 4)
  python pipeline.py --evaluate    # Run evaluation only (step 5)
  python pipeline.py --serve       # Start API server for Open WebUI
        """
    )
    
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build RAG system: parse, chunk, and embed documents (steps 1-3)"
    )
    
    parser.add_argument(
        "--chatbot",
        action="store_true",
        help="Run interactive chatbot (step 4)"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation (step 5)"
    )
    
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API server for Open WebUI integration"
    )
    
    args = parser.parse_args()
    
    # Determine which mode to run
    if args.build:
        build_rag()
    elif args.chatbot:
        run_chatbot()
    elif args.evaluate:
        run_evaluation()
    elif args.serve:
        # Start API server
        import uvicorn
        print("\n🚀 Starting API server for Open WebUI integration...")
        print("   API will be available at: http://localhost:8000")
        print("   OpenAI-compatible endpoint: http://localhost:8000/v1/chat/completions\n")
        uvicorn.run("api.api_server:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # No arguments provided - run full pipeline
        run_full_pipeline()
