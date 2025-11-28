"""
Document Parsing
Extracts text from PDFs and transcribes audio/video files
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
import whisper


def parse_pdfs(data_folder: str = "data", output_folder: str = "output") -> List[str]:
    """Parse PDF files using LangChain's PyPDFLoader."""
    print("=" * 80)
    print("PARSING PDFs")
    print("=" * 80)
    
    Path(output_folder).mkdir(exist_ok=True)
    data_path = Path(data_folder)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{data_folder}/'")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s)\n")
    processed_files = []
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            full_text = "\n\n".join([page.page_content for page in pages])
            
            output_path = Path(output_folder) / (pdf_file.stem + ".txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"  ✓ Loaded {len(pages)} pages")
            print(f"  ✓ Saved: {output_path.name} ({len(full_text):,} characters)")
            processed_files.append(str(output_path))
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return processed_files


def parse_audio(data_folder: str = "data", 
                output_folder: str = "output",
                model_name: str = "base") -> List[str]:
    """Transcribe audio/video files using Whisper."""
    print("\n" + "=" * 80)
    print("PARSING AUDIO/VIDEO")
    print("=" * 80)
    
    Path(output_folder).mkdir(exist_ok=True)
    data_path = Path(data_folder)
    
    audio_extensions = ["*.mp4", "*.mp3", "*.wav", "*.m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_path.glob(ext))
    
    if not audio_files:
        print(f"No audio/video files found in '{data_folder}/'")
        return []
    
    print(f"Found {len(audio_files)} audio/video file(s)")
    print(f"Whisper model: {model_name}\n")
    
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)
    print("✓ Model loaded\n")
    
    processed_files = []
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        try:
            print("  Transcribing (this may take a few minutes)...")
            result = model.transcribe(str(audio_file), language="en")
            full_text = result["text"]
            
            output_path = Path(output_folder) / (audio_file.stem + ".txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            words = len(full_text.split())
            print(f"  ✓ Transcription complete")
            print(f"  ✓ Saved: {output_path.name} ({words:,} words)")
            processed_files.append(str(output_path))
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return processed_files


if __name__ == "__main__":
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"
    WHISPER_MODEL = "base"
    
    pdf_files = parse_pdfs(DATA_FOLDER, OUTPUT_FOLDER)
    audio_files = parse_audio(DATA_FOLDER, OUTPUT_FOLDER, WHISPER_MODEL)
    
    print("\n" + "=" * 80)
    print("PARSING COMPLETE")
    print("=" * 80)
    print(f"Processed {len(pdf_files) + len(audio_files)} files")
    print(f"  - PDFs: {len(pdf_files)}")
    print(f"  - Audio/Video: {len(audio_files)}")
    print(f"Output saved to: {OUTPUT_FOLDER}/")
    print("=" * 80)
