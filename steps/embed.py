"""
Embedding and Storage
Generates embeddings and stores them in ChromaDB
"""

from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import chromadb


def load_chunks(chunks_folder: str = "chunks") -> List[Tuple[str, dict]]:
    """Load all chunks from files with metadata."""
    chunks_path = Path(chunks_folder)
    chunk_files = list(chunks_path.glob("*_chunks.txt"))
    
    if not chunk_files:
        print(f"No chunk files found in '{chunks_folder}/'")
        return []
    
    all_chunks = []
    
    for chunk_file in chunk_files:
        source = chunk_file.stem.replace("_chunks", "")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parts = content.split("=== CHUNK ")
        for part in parts[1:]:
            lines = part.split("\n", 1)
            if len(lines) == 2:
                chunk_id = lines[0].replace("===", "").strip()
                chunk_text = lines[1].strip()
                
                metadata = {
                    "source": source,
                    "chunk_id": chunk_id
                }
                all_chunks.append((chunk_text, metadata))
    
    return all_chunks


def embed_and_store(chunks: List[Tuple[str, dict]], 
                    db_path: str = "chroma_db",
                    collection_name: str = "rag_documents",
                    embedding_model: str = "BAAI/bge-base-en-v1.5") -> None:
    """Generate embeddings and store in ChromaDB."""
    if not chunks:
        print("No chunks to embed!")
        return
    
    print("=" * 80)
    print("EMBEDDING AND STORING")
    print("=" * 80)
    print(f"Embedding model: {embedding_model}")
    print(f"Total chunks: {len(chunks)}\n")
    
    print("Loading embedding model...")
    model = SentenceTransformer(embedding_model)
    print("✓ Model loaded\n")
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        client.delete_collection(name=collection_name)
        print("✓ Cleared existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ Created collection: {collection_name}\n")
    
    print("Generating embeddings...")
    texts = [chunk[0] for chunk in chunks]
    metadatas = [chunk[1] for chunk in chunks]
    
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print("\nStoring in database...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    sources = set([meta["source"] for meta in metadatas])
    
    print("\n" + "=" * 80)
    print("EMBEDDING COMPLETE")
    print("=" * 80)
    print(f"Embedded {len(chunks)} chunks from {len(sources)} documents")
    print(f"Database saved to: {db_path}/")
    print("=" * 80)


if __name__ == "__main__":
    CHUNKS_FOLDER = "chunks"
    DB_PATH = "chroma_db"
    COLLECTION_NAME = "rag_documents"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    
    chunks = load_chunks(CHUNKS_FOLDER)
    embed_and_store(chunks, DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL)
