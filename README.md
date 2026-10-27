# Agentic HR & Employee Support Assistant

An agentic HR chatbot powered by LangGraph, FastAPI, Groq (LLM), Pinecone (Vector DB), HuggingFace embeddings, and Tavily (web search fallback).

## Features
- **Adaptive Routing**: Distinguishes general conversational queries from HR policy inquiries.
- **Private Knowledge Retrieval**: Searches company HR policy documents stored in Pinecone vector store.
- **Evidence Grading & Fallback**: Self-evaluates retrieved HR contexts and seamlessly falls back to Tavily web search if internal policy context is insufficient.
- **Query Rewriting**: Iteratively optimizes ambiguous employee queries for better retrieval.
- **Audit Logging**: Persists question, source used, and execution trace to SQLite (`data/audit.db`).
- **Interactive UI & REST API**: Includes a browser chat interface and FastAPI endpoints (`/api/chat`, `/api/health`, `/api/ingest`).

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Configure `.env` with your API keys:
   ```env
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   TAVILY_API_KEY=your_tavily_key
   ```

3. **Ingest HR Documents**:
   Place policy documents (`.pdf`, `.docx`, `.txt`, `.md`) inside `data/hr-data/` and run:
   ```bash
   python ingest_sample_kb.py
   ```

4. **Run the Application**:
   ```bash
   python run.py
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
