# rag_core.py
import os
from dotenv import load_dotenv

# --- Core LangChain components (UPDATED IMPORTS) ---
from langchain_community.document_loaders import TextLoader   # UPDATED
from langchain_text_splitters import CharacterTextSplitter    # UPDATED
from langchain_huggingface import HuggingFaceEmbeddings # UPDATED (if using LangChain's wrapper)
# Alternatively, if you want to use sentence-transformers directly (no LangChain wrapper needed for this line then):
# from sentence_transformers import SentenceTransformer # And then model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# However, LangChain's HuggingFaceEmbeddings wrapper is convenient.
from langchain_community.vectorstores import Chroma           # UPDATED
from langchain_openai import ChatOpenAI                       # UPDATED
from langchain.chains import RetrievalQA                      # This one is often still in the base langchain.chains

# --- Setup ---
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY not found. Make sure it's set in your .env file.")
    exit()

# --- Constants ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DOCS_DIR = "sample_docs"
# PERSIST_DIR = "chroma_db" # We'll use this later

# --- Create Sample Documents (for testing) ---
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

with open(os.path.join(DOCS_DIR, "doc1.txt"), "w") as f:
    f.write("LangChain is a framework for developing applications powered by language models.\n")
    f.write("It provides building blocks for creating chains and agents.\n")
    f.write("RAG stands for Retrieval-Augmented Generation.")

with open(os.path.join(DOCS_DIR, "doc2.txt"), "w") as f:
    f.write("Chromadb is an open-source embedding database.\n")
    f.write("It makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.\n")
    f.write("Sentence Transformers is a Python framework for state-of-the-art sentence, text and image embeddings.")

print("--- Sample documents created ---")

# --- 1. Load Documents ---
print("--- Loading documents ---")
documents = []
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(DOCS_DIR, filename)
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if not documents:
    print("No documents loaded. Exiting.")
    exit()

# --- 2. Split Documents into Chunks ---
print("--- Splitting documents ---")
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=50,
    length_function=len
)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks")

# --- 3. Create Embeddings ---
print("--- Creating embeddings (this might take a moment) ---")
# This uses LangChain's wrapper for HuggingFace sentence-transformer models
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 4. Create Vector Store (ChromaDB) ---
print("--- Creating vector store ---")
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)
print("--- Vector store created ---")

# --- 5. Setup the RAG Chain ---
print("--- Setting up RAG chain ---")
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
print("--- RAG chain ready ---")

# --- 6. Ask a Question ---
query = "What is LangChain?"
print(f"\n--- Querying the RAG chain ---")
print(f"Question: {query}")

try:
    result = qa_chain.invoke({"query": query}) # Use .invoke for newer LangChain versions

    print("\n--- Answer ---")
    print(result["result"])

    # print("\n--- Source Documents ---")
    # for source_doc in result["source_documents"]:
    #     print(f"- {source_doc.page_content[:100]}... (Source: {source_doc.metadata.get('source', 'N/A')})")

except Exception as e:
    print(f"An error occurred during query: {e}")

print("\n--- Script finished ---")