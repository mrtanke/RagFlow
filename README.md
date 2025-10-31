# RagFlow

A modular **FastAPI-based backend** for Retrieval-Augmented Generation (RAG) applications.  
This project integrates **LangChain**, **ChromaDB**, and **FastAPI** to provide API endpoints and a simple app interface for LLM-powered Q&A systems.

## Installation

### 1. Clone the repository
```bash
git clone <url>
cd rag_fastapi
```

### 2. Create & activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Edit .env file
```bash
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the server
```bash
cd api
uvicorn main:app --reload
```

### 6. Start the Streamlit
```bash
cd frontend
streamlit run app.py
```

After running, the terminal will prompt:
- Local URL: http://localhost:8501

Click it (or copy it to your browser) and you’ll see the Streamlit interface.

