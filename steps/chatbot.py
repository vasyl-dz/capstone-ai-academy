"""
RAG Chatbot with Reasoning and Tool Calling
Interactive chatbot with retrieval-augmented generation, reasoning, reflection, and tool calling
"""

from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from agents.reasoning_agent import ReasoningAgent
from agents.reflection_agent import ReflectionAgent

# Load environment variables from .env file
load_dotenv()


class RAGChatbot:
    def __init__(self, 
                 db_path: str = "chroma_db",
                 collection_name: str = "rag_documents",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "llama3.1:8b",
                 use_reflection: bool = True):
        """Initialize RAG chatbot with reasoning, reflection, and tool calling."""
        print("=" * 80)
        print("INITIALIZING RAG CHATBOT WITH REASONING")
        print("=" * 80)
        
        print(f"Embedding model: {embedding_model}")
        print(f"LLM model (Ollama): {llm_model}")
        print(f"Reflection enabled: {use_reflection}\n")
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✓ Embedding model loaded\n")
        
        print("Initializing reasoning agent...")
        self.reasoning_agent = ReasoningAgent(model=llm_model)
        print("✓ Reasoning agent ready\n")
        
        if use_reflection:
            print("Initializing reflection agent...")
            self.reflection_agent = ReflectionAgent(model=llm_model)
            print("✓ Reflection agent ready\n")
        else:
            self.reflection_agent = None
        
        self.use_reflection = use_reflection
        
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(name=collection_name)
        print(f"✓ Connected to collection: {collection_name}")
        print(f"  Total chunks in database: {self.collection.count()}\n")
        
        print("=" * 80)
        print("CHATBOT READY")
        print("=" * 80)
        print("Features: Reasoning | Tool Calling | Reflection | Conversation Memory")
        print("=" * 80)
    
    def retrieve(self, query: str, n_results: int = 5) -> Tuple[List[str], List[dict]]:
        """Retrieve relevant chunks from vector database."""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        chunks = results['documents'][0] if results['documents'][0] else []
        metadatas = results['metadatas'][0] if results['metadatas'][0] else []
        
        return chunks, metadatas
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using reasoning agent with retrieved context."""
        return self.reasoning_agent.reason(query, context)
    
    def answer_with_reflection(self, query: str, context: str) -> dict:
        """Generate answer with reflection for self-correction."""
        # Get initial answer
        initial_answer = self.generate_answer(query, context)
        
        if not self.use_reflection or not self.reflection_agent:
            return {
                "answer": initial_answer,
                "used_reflection": False
            }
        
        # Apply reflection if needed
        reflection_result = self.reflection_agent.chain_reflect(
            query=query,
            initial_answer=initial_answer,
            context=context,
            max_iterations=2
        )
        
        return {
            "answer": reflection_result["final_answer"],
            "initial_answer": reflection_result["initial_answer"],
            "used_reflection": reflection_result["iterations"] > 0,
            "iterations": reflection_result["iterations"]
        }
    
    def chat(self):
        """Interactive chat loop with reasoning and reflection."""
        print("\nRAG Chatbot (type 'quit', 'exit' to stop, 'clear' to reset memory, 'memory' to see history)\n")
        
        while True:
            query = input("Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'clear':
                self.reasoning_agent.clear_memory()
                print("✓ Conversation memory cleared\n")
                continue
            
            if query.lower() == 'memory':
                print("\nConversation History:")
                print(self.reasoning_agent.get_memory_summary())
                print()
                continue
            
            if not query:
                continue
            
            print()
            
            # Retrieve relevant context
            chunks, metadatas = self.retrieve(query)
            
            if chunks:
                sources = list(set([meta['source'] for meta in metadatas]))
                print(f"📚 Retrieved {len(chunks)} chunks from: {', '.join(sources)}")
                context = "\n\n".join(chunks)
            else:
                print("📚 No relevant context found in knowledge base")
                context = ""
            
            # Generate answer with optional reflection
            result = self.answer_with_reflection(query, context)
            
            print(f"\n💡 Answer: {result['answer']}")
            
            if result.get('used_reflection'):
                print(f"   (Refined through {result['iterations']} reflection iterations)")
            
            print("\n" + "-" * 80 + "\n")
    
    def answer_question(self, query: str) -> str:
        """Answer a single question (for evaluation)."""
        chunks, _ = self.retrieve(query)
        context = "\n\n".join(chunks) if chunks else ""
        result = self.answer_with_reflection(query, context)
        return result['answer']


if __name__ == "__main__":
    DB_PATH = "chroma_db"
    COLLECTION_NAME = "rag_documents"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "llama3.1:8b"
    
    chatbot = RAGChatbot(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL)
    chatbot.chat()
