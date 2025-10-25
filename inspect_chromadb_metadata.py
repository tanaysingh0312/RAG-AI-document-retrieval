<<<<<<< HEAD
# inspect_chromadb_metadata.py

import logging
import json
import os
from chromadb import PersistentClient

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Must match rag_ingest.py and your RAG system) ---
# IMPORTANT CHANGE: Use the absolute path for ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/home/jahnavi/chroma_storage")
CHROMA_COLLECTION_NAME = "pyrotech_docs"

def inspect_chromadb_collection():
    """
    Connects to ChromaDB and prints metadata for a sample of documents.
    """
    logger.info(f"Connecting to ChromaDB at '{CHROMA_DB_PATH}', collection: '{CHROMA_COLLECTION_NAME}'")
    try:
        client = PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME) # Use get_collection, not get_or_create_collection

        count = collection.count()
        logger.info(f"Total documents in collection '{CHROMA_COLLECTION_NAME}': {count}")

        if count == 0:
            logger.warning("Collection is empty. No documents to inspect.")
            return

        # Fetch a sample of documents (e.g., first 10)
        results = collection.peek(limit=10) # 'peek' gets the first N items

        if not results or not results.get('metadatas'):
            logger.warning("No metadatas found in the sample documents.")
            return

        logger.info("\n--- Sample Document Metadata ---")
        for i, metadata in enumerate(results['metadatas']):
            logger.info(f"\nDocument {i+1} ID: {results['ids'][i]}")
            logger.info(f"Content Snippet: '{results['documents'][i][:150]}...'") # First 150 chars of content
            logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
            
            # Highlight key metadata fields
            logger.info(f"  -> product_name: {metadata.get('product_name', 'N/A')}")
            logger.info(f"  -> section_type: {metadata.get('section_type', 'N/A')}")
            logger.info(f"  -> document_type: {metadata.get('document_type', 'N/A')}")
            logger.info(f"  -> user: {metadata.get('user', 'N/A')}")
            logger.info(f"  -> source_file: {metadata.get('source_file', 'N/A')}")
            logger.info(f"  -> page_number: {metadata.get('page_number', 'N/A')}")

        logger.info("\n--- Inspection Complete ---")

    except Exception as e:
        logger.error(f"Error inspecting ChromaDB collection: {e}. Make sure ChromaDB is accessible and the collection name is correct.", exc_info=True)

if __name__ == "__main__":
    inspect_chromadb_collection()

=======
# inspect_chromadb_metadata.py

import logging
import json
import os
from chromadb import PersistentClient

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Must match rag_ingest.py and your RAG system) ---
# IMPORTANT CHANGE: Use the absolute path for ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/home/jahnavi/chroma_storage")
CHROMA_COLLECTION_NAME = "pyrotech_docs"

def inspect_chromadb_collection():
    """
    Connects to ChromaDB and prints metadata for a sample of documents.
    """
    logger.info(f"Connecting to ChromaDB at '{CHROMA_DB_PATH}', collection: '{CHROMA_COLLECTION_NAME}'")
    try:
        client = PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME) # Use get_collection, not get_or_create_collection

        count = collection.count()
        logger.info(f"Total documents in collection '{CHROMA_COLLECTION_NAME}': {count}")

        if count == 0:
            logger.warning("Collection is empty. No documents to inspect.")
            return

        # Fetch a sample of documents (e.g., first 10)
        results = collection.peek(limit=10) # 'peek' gets the first N items

        if not results or not results.get('metadatas'):
            logger.warning("No metadatas found in the sample documents.")
            return

        logger.info("\n--- Sample Document Metadata ---")
        for i, metadata in enumerate(results['metadatas']):
            logger.info(f"\nDocument {i+1} ID: {results['ids'][i]}")
            logger.info(f"Content Snippet: '{results['documents'][i][:150]}...'") # First 150 chars of content
            logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
            
            # Highlight key metadata fields
            logger.info(f"  -> product_name: {metadata.get('product_name', 'N/A')}")
            logger.info(f"  -> section_type: {metadata.get('section_type', 'N/A')}")
            logger.info(f"  -> document_type: {metadata.get('document_type', 'N/A')}")
            logger.info(f"  -> user: {metadata.get('user', 'N/A')}")
            logger.info(f"  -> source_file: {metadata.get('source_file', 'N/A')}")
            logger.info(f"  -> page_number: {metadata.get('page_number', 'N/A')}")

        logger.info("\n--- Inspection Complete ---")

    except Exception as e:
        logger.error(f"Error inspecting ChromaDB collection: {e}. Make sure ChromaDB is accessible and the collection name is correct.", exc_info=True)

if __name__ == "__main__":
    inspect_chromadb_collection()

>>>>>>> dd6123601b3b1df1e426d6eac04f58ebd5c6ba7f
