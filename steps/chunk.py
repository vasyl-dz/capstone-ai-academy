"""
Text Chunking
Splits documents into semantic chunks using LangChain
"""

from pathlib import Path
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_content_type(text: str) -> str:
    """Detect if text is a presentation or transcription."""
    slide_indicators = text.count('•') + text.count('○') + text.count('■')
    avg_sentence_length = len(text.split()) / max(text.count('.'), 1)
    
    if slide_indicators > 5 or avg_sentence_length < 15:
        return "presentation"
    return "transcription"


def chunk_documents(input_folder: str = "output", 
                    output_folder: str = "chunks",
                    chunk_size: int = 1000,
                    chunk_overlap: int = 150) -> List[str]:
    """Chunk documents using content-aware splitting."""
    print("=" * 80)
    print("CHUNKING DOCUMENTS")
    print("=" * 80)
    
    Path(output_folder).mkdir(exist_ok=True)
    input_path = Path(input_folder)
    text_files = list(input_path.glob("*.txt"))
    
    if not text_files:
        print(f"No text files found in '{input_folder}/'")
        return []
    
    print(f"Found {len(text_files)} document(s)")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}\n")
    
    processed_files = []
    total_chunks = 0
    
    for text_file in text_files:
        print(f"Processing: {text_file.name}")
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            content_type = detect_content_type(text)
            
            if content_type == "presentation":
                separators = ["\n\n", "\n", ". ", " ", ""]
            else:
                separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len
            )
            
            chunks = splitter.split_text(text)
            
            output_path = Path(output_folder) / (text_file.stem + "_chunks.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"=== CHUNK {i} ===\n")
                    f.write(chunk)
                    f.write("\n\n")
            
            print(f"  ✓ Type: {content_type}")
            print(f"  ✓ Created {len(chunks)} chunks")
            print(f"  ✓ Saved: {output_path.name}")
            
            processed_files.append(str(output_path))
            total_chunks += len(chunks)
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)
    print(f"Processed {len(processed_files)} documents")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output saved to: {output_folder}/")
    print("=" * 80)
    
    return processed_files


if __name__ == "__main__":
    INPUT_FOLDER = "output"
    OUTPUT_FOLDER = "chunks"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    
    chunk_documents(INPUT_FOLDER, OUTPUT_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP)
