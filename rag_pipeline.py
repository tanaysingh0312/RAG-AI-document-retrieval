import os
import json
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings # For OpenAI Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # For local models like sentence-transformers
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI # For OpenAI Chat Models
from langchain_community.llms import Ollama # For local models like Llama2 via Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv # Import to load .env file

# Load environment variables from .env file (for API keys like OPENAI_API_KEY)
load_dotenv()

# --- Configuration for file paths ---
DOCS_PROCESSED_FOLDER = "docs_processed"
CHUNKS_FOLDER = "chunks" # This folder is for optional saving of chunks, not strictly necessary for pipeline run
VECTOR_DB_PATH = "faiss_index"

# --- Initialize Embeddings Model ---
# Choose ONE option below by uncommenting the relevant line and commenting out others.
# Option 1: OpenAI Embeddings (Requires OPENAI_API_KEY set in .env)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Option 2: Local HuggingFace Embeddings (e.g., from sentence-transformers).
# This model ('all-MiniLM-L6-v2') will be downloaded automatically the first time it's used.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # <--- THIS LINE IS NOW CORRECTLY UNINDENTED AND UNCOMMENTED


# --- Initialize Large Language Model (LLM) ---
# Choose ONE option below by uncommenting the relevant line and commenting out others.
# Option 1: OpenAI Chat Model (Requires OPENAI_API_KEY set in .env)
# llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Option 2: Local LLM via Ollama (Requires Ollama server running and model pulled, e.g., 'ollama pull llama2' or 'ollama pull mistral')
# IMPORTANT: Ensure the model name here matches the model you pulled with Ollama.
llm = Ollama(model="mistral", temperature=0.1) # <--- THIS LINE IS UNCOMMENTED AND CORRECT FOR MISTRAL

# --- Document Loading and Chunking Functions ---
def load_documents_from_folder(folder_path):
    """
    Loads text or structured JSON documents from a specified folder.
    Assumes JSON files from OCR pipeline have a list of objects with a 'text' key.
    """
    documents = []
    if not os.path.exists(folder_path):
        print(f"Warning: Document processed folder not found: {folder_path}")
        return []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                documents.extend(loader.load())
                print(f"Loaded text from {filename}")
            except Exception as e:
                print(f"Error loading text from {filepath}: {e}")
        elif filename.endswith(".json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Assumes JSON structure like: [{"type": "Text", "text": "...", "box": [...]}]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            doc_text = item['text']
                            # Add metadata like source file path and block type
                            metadata = {"source": filepath, "type": item.get('type', 'unknown')}
                            documents.append(Document(page_content=doc_text, metadata=metadata))
                else:
                    print(f"Warning: JSON file {filepath} is not a list of objects. Skipping.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filepath}: {e}")
            except Exception as e:
                print(f"Error loading structured JSON from {filepath}: {e}")
            print(f"Loaded structured JSON from {filename}")
        else:
            print(f"Skipping unsupported document type: {filename}")
    return documents

def chunk_documents(documents):
    """Splits documents into smaller, overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum characters in a chunk
        chunk_overlap=200,    # Overlap between chunks to maintain context
        length_function=len,  # Use standard Python len() for chunk size calculation
        add_start_index=True, # Adds metadata about the starting character index of the chunk
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# --- Vector Store Functions ---
def create_vector_store(chunks):
    """Creates a FAISS vector store from document chunks and saves it locally."""
    print("Creating vector store...")
    if not chunks:
        print("No chunks provided to create vector store. Returning None.")
        return None
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
        print(f"Vector store created and saved to {VECTOR_DB_PATH}")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def load_vector_store():
    """Loads an existing FAISS vector store from disk."""
    if os.path.exists(VECTOR_DB_PATH):
        print(f"Loading existing vector store from {VECTOR_DB_PATH}...")
        try:
            # allow_dangerous_deserialization=True is necessary when loading FAISS indexes
            # that might contain custom object types (like Document objects with metadata).
            vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded.")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store from {VECTOR_DB_PATH}: {e}")
            print("You might need to rebuild the vector store.")
            return None
    else:
        print(f"No vector store found at {VECTOR_DB_PATH}. It needs to be created first.")
        return None

# --- RAG Chain Setup Function ---
def setup_rag_chain(retriever, llm_model):
    """
    Sets up the Retrieval Augmented Generation (RAG) chain.
    This chain takes a user query, retrieves relevant documents, and passes them
    to the LLM for answer generation.
    """
    print("Setting up RAG chain...")

    # Define the prompt template for the LLM
    # The 'context' will be the retrieved document chunks, 'input' is the user's question.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based only on the provided context. If you don't know the answer, explicitly state that you don't know."),
        ("user", "Context: {context}\nQuestion: {input}")
    ])

    # Create a chain that combines the retrieved documents into the prompt
    # 'stuff' means it will "stuff" all retrieved documents into the single prompt context.
    document_chain = create_stuff_documents_chain(llm_model, prompt)

    # Create the full retrieval chain:
    # 1. Takes the user's question.
    # 2. Uses the retriever to get relevant documents.
    # 3. Passes the question and documents to the document_chain (which then uses the LLM).
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain setup complete.")
    return rag_chain

# This file (rag_pipeline.py) does not have an 'if __name__ == "__main__":' block.
# It serves as a module of functions and pre-initialized components that other scripts (like rag_app.py) will import and use.