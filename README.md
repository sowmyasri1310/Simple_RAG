# Simple RAG Web Application

This project is a complete, beginner-friendly implementation of a **Retrieval-Augmented Generation (RAG)** web application using Python, Streamlit, ChromaDB, Sentence Transformers, and the Groq API.

## Project Overview

The application allows users to upload PDF documents, automatically extracts and processes the text into a vector database, and provides a chat interface to ask questions about the uploaded content. The AI answers are strictly based on the uploaded documents to prevent hallucinations.

## Architecture

1. **Frontend:** Streamlit (UI for file upload, configuration, and chat).
2. **Document Processing:** PyPDF2 (extracts text from PDFs).
3. **Chunking & Embedding:** `sentence-transformers` (`all-MiniLM-L6-v2`) converts chunks of text into vector representations.
4. **Vector Database:** ChromaDB (stores chunks, embeddings, and metadata persistently).
5. **LLM Generation:** Groq API (uses models like `llama-3.3-70b-versatile` for fast inference).
6. **Observability:** LangSmith (traces the execution pipeline).

## Educational Concepts

### How RAG Works
RAG bridges the gap between a Large Language Model's general knowledge and your specific, private data. Instead of fine-tuning a model (which is expensive and slow), RAG works by:
1. **Retrieval:** Searching your database for paragraphs relevant to the user's question.
2. **Augmentation:** Injecting those relevant paragraphs into the prompt alongside the user's question.
3. **Generation:** The LLM reads the injected paragraphs and generates a factual answer based *only* on that context.

### What is Similarity Search?
When you upload a document, we convert the text into **Embeddings** (arrays of numbers representing semantic meaning). When you ask a question, we convert the question into an embedding too. 
**Similarity Search** calculates the mathematical distance (e.g., Cosine Similarity) between the question's vector and all the document vectors. The closer the vectors, the more relevant the text.

### Metadata Filtering (Key-Value Search)
Sometimes you only want to search within a specific document. Since we store metadata (like `filename`, `page_number`) alongside the embeddings, we can apply filters before the similarity search. For example, filtering by `{"filename": "policy.pdf"}` ensures we only retrieve chunks from that specific file.

### What is LangSmith?
LangSmith is a platform for tracing and monitoring LLM applications. In a RAG pipeline, it's crucial to know *exactly* what context was retrieved and what exact prompt was sent to the LLM. LangSmith logs every step, helping developers debug bad answers or performance bottlenecks.

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys)
- A [LangSmith API Key](https://smith.langchain.com/)

### 2. Installation
Clone the repository, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and paste your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=SimpleRAGProject
LANGCHAIN_TRACING_V2=true
```

### 4. Running the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

## Screenshots

*(Placeholders for screenshots)*

- **App Interface:** `[Insert screenshot of the main chat UI here]`
- **Explainability Feature:** `[Insert screenshot of the retrieved chunks expander here]`
- **Sidebar Options:** `[Insert screenshot of the model selection and filtering sidebar here]`
