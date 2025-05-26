# Intelligent Document Analyzer & Q&A System

This project is a Generative AI-powered application that allows users to upload documents (TXT, PDF) and ask natural language questions about their content. It leverages a Retrieval-Augmented Generation (RAG) architecture.

## Features
- Document upload (TXT, PDF)
- Text extraction and chunking
- Embedding generation using Sentence Transformers (`all-MiniLM-L6-v2`)
- Vector storage and retrieval using ChromaDB
- Question answering using OpenAI's `gpt-3.5-turbo` via LangChain
- Streamlit web interface

## Tech Stack
- Python
- LangChain
- OpenAI API
- Sentence Transformers (Hugging Face)
- ChromaDB
- Streamlit
- PyPDF
- python-dotenv

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd intelligent-doc-analyzer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(We'll create requirements.txt next)*

4.  **Set up environment variables:**
    Create a `.env` file in the project root and add your OpenAI API key:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Running the Application

```bash
streamlit run app.py