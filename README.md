# AI-Agentic RAG System

AI-Agentic system capable of reasoning, tool-calling, and self-reflection.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python pipeline.py

# Or run specific stages
python pipeline.py --build      # Build RAG (steps 1-3)
python pipeline.py --chatbot    # Interactive chat (step 4)
python pipeline.py --evaluate   # Run evaluation (step 5)
python pipeline.py --serve      # Start API server for Open WebUI
```

## Features

- **AI Agents**: Reasoning agent with tool-calling and reflection agent for self-correction
- **Document Processing**: PDFs (LangChain PyPDFLoader) and audio/video (Whisper)
- **Semantic Chunking**: Content-aware splitting with LangChain RecursiveCharacterTextSplitter
- **State-of-the-art Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions)
- **Powerful LLM**: Ollama with llama3.1:8b (tool-calling capable)
- **Tool Integration**: GitHub repository analysis
- **Vector Database**: ChromaDB with persistent storage
- **Conversation Memory**: Maintains context across interactions
- **OpenAI-Compatible API**: Use via Open WebUI or any OpenAI-compatible client

## Architecture

```
data/                    # Input: PDFs and audio/video files
├── lecture1.pdf
├── meeting1.mp4
└── ...

output/                  # Parsed text files
├── lecture1.txt
├── meeting1.txt
└── ...

chunks/                  # Chunked documents
├── lecture1_chunks.txt
├── meeting1_chunks.txt
└── ...

chroma_db/              # Vector database (persistent)

steps/                  # Pipeline steps
├── parse.py            # Step 1: Document parsing
├── chunk.py            # Step 2: Semantic chunking
├── embed.py            # Step 3: Generate embeddings
├── chatbot.py          # Step 4: Interactive Q&A
└── evaluate.py         # Step 5: Evaluation

agents/                 # AI Agents
├── reasoning_agent.py  # Reasoning with tool calling
└── reflection_agent.py # Self-reflection and correction

tools/                  # Tools for agents
└── github_tool.py      # GitHub API integration

api/                    # API Server
└── api_server.py       # OpenAI-compatible API for Open WebUI

pipeline.py             # Orchestrates all steps
requirements.txt        # Dependencies
```

## Pipeline Steps

### 1. Parse (steps/parse.py)

- Extracts text from PDFs using LangChain's PyPDFLoader
- Transcribes audio/video files using OpenAI Whisper
- Output: `.txt` files in `output/` folder

### 2. Chunk (steps/chunk.py)

- Detects content type (presentation vs transcription)
- Splits documents using RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
- Output: `_chunks.txt` files in `chunks/` folder

### 3. Embed (steps/embed.py)

- Generates embeddings with BAAI/bge-base-en-v1.5
- Stores in ChromaDB with metadata (source, chunk_id)
- Output: Persistent vector database in `chroma_db/`

### 4. Chatbot (steps/chatbot.py)

- Retrieves top 5 relevant chunks per query using vector similarity search
- Uses **Reasoning Agent** with Ollama (llama3.1:8b) for intelligent responses
- Applies **Reflection Agent** for self-correction and answer improvement
- Supports tool-calling (e.g., GitHub repository analysis)
- Maintains conversation memory for context-aware interactions
- Interactive Q&A loop with commands: 'quit', 'clear', 'memory'

### 5. Evaluate (steps/evaluate.py)

- LangSmith integration with OpenEvals LLM-as-a-Judge
- Measures accuracy, relevance, and clarity
- Requires LANGSMITH_API_KEY environment variable

## AI Agents

### Reasoning Agent (agents/reasoning_agent.py)

- **Tool-calling capable**: Can invoke external tools (GitHub API, etc.)
- **Conversation memory**: Maintains context across multi-turn dialogues
- **Structured reasoning**: Uses chain-of-thought for complex queries
- **Model**: Ollama llama3.1:8b (or other tool-calling capable models)

### Reflection Agent (agents/reflection_agent.py)

- **Self-correction**: Evaluates and improves initial answers
- **Quality checks**: Assesses accuracy, completeness, clarity, and relevance
- **Iterative refinement**: Performs up to 2 reflection passes via chain_reflect()
- **Smart activation**: Uses heuristics to determine when reflection is needed

## Tools

### GitHub Tool (tools/github_tool.py)

- Analyzes GitHub repositories via PyGithub API
- Fetches README, metadata, and repository information
- Integrated with reasoning agent for repository-related queries

## API Server

### FastAPI Server (api/api_server.py)

- **OpenAI-compatible API**: Works with Open WebUI and other clients
- **Endpoints**:
  - `GET /` - Root status
  - `GET /health` - Health check
  - `GET /v1/models` - List available models (OpenAI format)
  - `POST /v1/chat/completions` - Chat completions (OpenAI format)
  - `POST /query` - Simple query endpoint for testing
- **Usage**: Start with `python pipeline.py --serve`
- **Open WebUI**: Add `http://localhost:8000/v1` to OpenAI API connections

## Configuration

All settings are defined as constants in pipeline.py and individual modules:

**pipeline.py**

- `DATA_FOLDER = "data"`
- `OUTPUT_FOLDER = "output"`
- `CHUNKS_FOLDER = "chunks"`
- `DB_PATH = "chroma_db"`
- `WHISPER_MODEL = "base"`
- `EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"`
- `LLM_MODEL = "llama3.1:8b"` (Ollama)
- `CHUNK_SIZE = 1000`
- `CHUNK_OVERLAP = 150`

**Environment Variables (.env)**

- `GITHUB_TOKEN` - Optional, for GitHub API access (higher rate limits)
- `LANGSMITH_API_KEY` - Required for evaluation (step 5)
- `API_PORT` - Optional, default: 8000

## Testing

1. **Prepare data**: Place sample files in `data/` folder:

   - PDFs (lecture slides, presentations)
   - MP4/MP3/WAV files (meeting recordings, lectures)

2. **Build RAG system**:

   ```bash
   python pipeline.py --build
   ```

3. **Start chatbot**:

   ```bash
   python pipeline.py --chatbot
   ```

4. **Try commands**:

   - Ask questions: "What are the best practices for production RAG?"
   - Test tool-calling: "Analyze the github.com/owner/repo repository"
   - View memory: `memory`
   - Clear history: `clear`
   - Exit: `quit`

5. **Run evaluation**:

   ```bash
   python pipeline.py --evaluate
   ```

6. **Use via Open WebUI**:
   ```bash
   python pipeline.py --serve
   ```
   Then add `http://localhost:8000/v1` to Open WebUI's OpenAI API connections.

## Requirements

- Python 3.11+ (recommended)
- Ollama installed and running (for LLM)
- ~5GB disk space for models
- 8GB+ RAM recommended

## Dependencies

Core packages (see `requirements.txt`):

- `langchain-community` - PDF parsing
- `langchain-text-splitters` - Semantic chunking
- `openai-whisper` - Audio transcription
- `sentence-transformers` - Embeddings
- `chromadb` - Vector database
- `ollama` - LLM inference
- `PyGithub` - GitHub API
- `langsmith` - Evaluation framework
- `openevals` - LLM-as-a-Judge
- `fastapi` + `uvicorn` - API server
- `python-dotenv` - Environment variables

## Models Used

| Component     | Model                 | Size   | Purpose                             |
| ------------- | --------------------- | ------ | ----------------------------------- |
| Transcription | Whisper Base          | ~140MB | Audio → Text                        |
| Embeddings    | BAAI/bge-base-en-v1.5 | ~440MB | Text → Vectors                      |
| LLM           | Ollama llama3.1:8b    | ~4.7GB | Reasoning, Tool-calling, Generation |

## License

MIT
