# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import shutil # For removing directories

# --- Core LangChain components (UPDATED IMPORTS) ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader # PyPDFLoader for PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # UPDATED
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI                # UPDATED
from langchain.chains import RetrievalQA

# --- Constants ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db_streamlit" # Directory to store persisted ChromaDB
UPLOAD_DIR = "uploaded_files"       # Directory to temporarily store uploaded files

# --- Helper Functions ---

@st.cache_resource # Cache the embeddings model loading
def load_embeddings():
    """Loads the HuggingFace embeddings model."""
    st.write("Loading embeddings model (this may take a moment on first run)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.write("Embeddings model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings model: {e}")
        return None

@st.cache_resource # Cache the vector store loading/creation
def load_vector_store(_embeddings): # Pass embeddings as argument
    """Loads or creates the persisted Chroma vector store."""
    if _embeddings is None:
        st.error("Embeddings not loaded, cannot initialize vector store.")
        return None

    if not os.path.exists(PERSIST_DIR):
        st.write(f"Creating new vector store in '{PERSIST_DIR}'...")
        # Create dummy store if it doesn't exist, as Chroma requires some data on init
        # This ensures the directory and basic structure are there.
        # We'll add actual documents after they are processed.
        vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
        # You might add a single dummy document to ensure persistence works as expected on first creation
        # vector_store.add_texts(["Initialising store"], metadatas=[{"source":"system"}])
        vector_store.persist() # Persist immediately after creation
        st.write("New vector store directory created. Upload documents to populate it.")
    else:
        st.write(f"Loading existing vector store from '{PERSIST_DIR}'...")
        vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
        st.write("Existing vector store loaded.")
    return vector_store

def process_uploaded_files(uploaded_files, vector_store, embeddings):
    """Processes uploaded files and adds them to the vector store."""
    if not uploaded_files:
        st.info("No files selected for upload.")
        return "No files uploaded."

    if vector_store is None or embeddings is None:
        st.error("Vector store or embeddings not initialized. Cannot process files.")
        return "Error: System not ready."

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    processed_files_count = 0
    total_chunks_added = 0
    all_texts_to_add = [] # Accumulate all chunks before adding to DB

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if uploaded_file.name.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load_and_split() # PyPDFLoader can split by page
            else:
                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                os.remove(file_path) # Clean up unsupported file
                continue

            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)
            all_texts_to_add.extend(texts) # Add to our accumulator list

            st.success(f"Prepared {uploaded_file.name} ({len(texts)} chunks).")
            processed_files_count += 1

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path) # Clean up the temporary file

    if all_texts_to_add:
        try:
            st.write(f"Adding {len(all_texts_to_add)} chunks from {processed_files_count} file(s) to the vector store...")
            vector_store.add_documents(all_texts_to_add)
            vector_store.persist() # Save changes to disk
            total_chunks_added = len(all_texts_to_add)
            st.success(f"Successfully added {total_chunks_added} chunks to the vector store.")
             # Force a rerun to update UI elements that depend on vector_store state
            # st.experimental_rerun() # May not be needed if RAG chain re-creation is robust
        except Exception as e:
            st.error(f"Error adding documents to vector store: {e}")

    if os.path.exists(UPLOAD_DIR) and not os.listdir(UPLOAD_DIR): # Clean up upload dir if empty
        os.rmdir(UPLOAD_DIR)

    return f"Processed {processed_files_count} file(s). Added {total_chunks_added} chunks to the knowledge base."


def get_rag_chain(vector_store):
    """Creates the RetrievalQA chain."""
    if vector_store is None:
      st.error("Vector store not initialized.")
      return None

    # A more robust check to see if the vector store has any actual content
    try:
        # Attempt a dummy search. If this fails, the store might be empty or unusable.
        # The exact behavior of an empty Chroma store search might vary, adjust if needed.
        if not vector_store.similarity_search("test_query_to_check_store", k=1):
            # This branch might be hit if search executes but returns empty list for an empty store
            # However, Chroma usually requires data to be initialized. The `load_vector_store`
            # handles initial creation, but it might be empty of user docs.
            st.warning("Vector store appears to be empty. Please upload and process documents.")
            return None
    except Exception as e:
        # This exception could occur if the store is in a bad state or truly empty in a way that
        # `similarity_search` fails.
        st.warning(f"Vector store not ready or empty. Upload documents. (Details: {e})")
        return None


    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.2
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" is good for a few retrieved docs
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 chunks
        return_source_documents=True
    )
    return qa_chain

def clear_vector_store():
    """Deletes the persisted vector store directory."""
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            st.success(f"Vector store directory '{PERSIST_DIR}' cleared.")
            # Clear Streamlit caches related to the vector store to force recreation
            st.cache_resource.clear() # This is important!
            # Need to manually trigger a rerun for the changes to reflect immediately in the UI state
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error clearing vector store: {e}")
    else:
        st.info("Vector store directory not found (already cleared or never created).")


# --- Streamlit App Interface ---

st.set_page_config(page_title="Intelligent Document Analyzer", layout="wide")
st.title("📄 Intelligent Document Analyzer & Q&A")
st.markdown("Upload your documents (TXT, PDF), and ask questions about their content.")

# --- Check for API Key ---
if not OPENAI_API_KEY:
    st.error("🚨 OpenAI API Key not found! Please set it in your .env file or environment variables.")
    st.stop() # Halt execution if no API key

# --- Load Embeddings and Vector Store (cached) ---
embeddings = load_embeddings()
vector_store = load_vector_store(embeddings) # Pass loaded embeddings

# --- Sidebar for Uploading and Clearing ---
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key="file_uploader_key" # Unique key for the uploader widget
    )

    if st.button("Process Uploaded Files", key="process_button"):
        if uploaded_files: # Check if files are actually uploaded
            with st.spinner("Processing files... This may take a moment."):
                result_message = process_uploaded_files(uploaded_files, vector_store, embeddings)
                st.success(result_message)
                # Optionally, clear the file uploader after processing
                # This requires using session_state trickery or a more complex component.
                # For simplicity, we'll leave files listed. User can remove or upload new.
                # To clear: st.session_state.file_uploader_key = [] and then rerun, but keys need care.
        else:
            st.info("No files selected to process.")


    st.subheader("Manage Vector Store")
    if st.button("⚠️ Clear All Documents (Vector Store)", key="clear_button"):
         clear_vector_store()
         # The clear_vector_store function now handles rerun and cache clearing

# --- Main Area for Q&A ---
st.header("❓ Ask Questions")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the RAG chain (might return None if store is empty or not ready)
    qa_chain = get_rag_chain(vector_store)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if qa_chain:
            with st.spinner("Thinking..."):
                try:
                    # Use .invoke for new LangChain
                    result = qa_chain.invoke({"query": prompt})
                    full_response = result["result"]

                    with st.expander("View Source Chunks", expanded=False):
                        if result.get("source_documents"):
                            for i, source_doc in enumerate(result["source_documents"]):
                                st.write(f"**Chunk {i+1} (Source: {source_doc.metadata.get('source', 'N/A')})**")
                                st.markdown(f"```\n{source_doc.page_content}\n```") # Use markdown for better formatting
                            if not result["source_documents"]:
                                st.write("No relevant source chunks found for this answer.")
                        else:
                            st.write("Source documents not available in the result.")

                except Exception as e:
                    full_response = f"An error occurred: {e}"
                    st.error(full_response) # Show error in chat
        else:
            full_response = "Vector store is not ready or is empty. Please upload and process documents first."
            st.warning(full_response) # Show warning in chat

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})