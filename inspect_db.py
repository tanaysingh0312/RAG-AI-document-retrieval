<<<<<<< HEAD
import chromadb
import os
import json
from typing import Optional

import logging
import numpy as np # Added for robust type checking of embeddings
import argparse # For CLI support
from collections import Counter # For most common section_type
import traceback # For detailed error logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (MUST match your rag.py settings) ---
# Assuming BASE_DIR is the directory where your rag.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CHROMA_DB_PATH = "/mnt/c/Users/Asus/pyrotech/db/chroma_db_final"
COLLECTION_NAME = "pyrotech_docs"
# --------------------------------------------------------

def inspect_chromadb(
    limit: Optional[int] = None, # Changed default to None to signify "all"
    offset: int = 0,
    source_file_name: Optional[str] = None,
    save_json_path: Optional[str] = None,
    no_embeddings: bool = False
):
    """
    Connects to the ChromaDB, retrieves the specified collection,
    and prints information about its contents.
    
    Args:
        limit (int, optional): The maximum number of documents to retrieve for general inspection.
                               If None, retrieves all documents.
        offset (int): The offset from which to start retrieving documents.
        source_file_name (str, optional): If provided, filters documents by this source file name.
        save_json_path (str, optional): If provided, saves the inspected documents to this JSON file.
        no_embeddings (bool): If True, skips including embedding vectors in the output for faster inspection.
    """
    logger.info(f"Attempting to connect to ChromaDB at: {CHROMA_DB_PATH}")

    # Handle edge case: CHROMA_DB_PATH doesn’t exist
    if not os.path.exists(CHROMA_DB_PATH):
        logger.error(f"Error: ChromaDB path does not exist: {CHROMA_DB_PATH}")
        logger.error("Please ensure the RAG ingestion process has run successfully to create the database.")
        return # Exit cleanly

    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info("ChromaDB client initialized successfully.")

        # Get the collection
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"Accessed collection: '{COLLECTION_NAME}'")

        # 1. Check the total number of documents
        total_documents_count = collection.count()
        logger.info(f"\n--- Collection Summary ---")
        logger.info(f"Total documents in collection '{COLLECTION_NAME}': {total_documents_count}")

        if total_documents_count == 0:
            logger.warning("No documents found in the collection. Ingestion might not have completed successfully.")
            return

        # Prepare include list for ChromaDB query
        include_list = ['documents', 'metadatas']
        if not no_embeddings:
            include_list.append('embeddings')

        # 2. Query documents based on CLI arguments
        query_where_clause = {}
        if source_file_name:
            query_where_clause["source_file_name"] = source_file_name
            logger.info(f"\n--- Inspecting documents filtered by source_file_name: '{source_file_name}' ---")
            # When filtering by source_file_name, retrieve all matching documents
            results = collection.get(
                where=query_where_clause,
                include=include_list
            )
            logger.info(f"Found {len(results.get('ids', []))} documents matching source_file_name: '{source_file_name}'. Displaying all.")
        else:
            # Retrieve all documents if limit is None, otherwise apply limit/offset
            logger.info(f"\n--- Inspecting documents (Limit: {'All' if limit is None else limit}, Offset: {offset}) ---")
            
            # Fetch all documents if limit is None, otherwise use the provided limit and offset
            if limit is None:
                results = collection.get(
                    limit=total_documents_count, # Fetch all documents up to the total count
                    offset=offset,
                    include=include_list
                )
            else:
                results = collection.get(
                    limit=limit,
                    offset=offset,
                    include=include_list
                )
            
            if not results.get('ids', []):
                logger.info(f"No documents found for the specified limit/offset range (Limit: {'All' if limit is None else limit}, Offset: {offset}).")
                return


        ids_to_display = results.get('ids', [])
        documents_to_display = results.get('documents', [])
        metadatas_to_display = results.get('metadatas', [])
        embeddings_to_display = results.get('embeddings', []) if not no_embeddings else [None] * len(ids_to_display)

        inspected_chunks_for_json = [] # To store data for optional JSON export
        missing_embeddings_count = 0
        all_section_types = []
        unique_source_files = set()

        if not ids_to_display: # Check if there are documents to display after query
            if source_file_name:
                logger.info(f"No documents found with source_file_name: '{source_file_name}'")
            else:
                logger.info(f"No documents found for the specified limit/offset range.")
            return

        # Iterate over the documents selected for display
        for i in range(len(ids_to_display)):
            doc_id = ids_to_display[i]
            doc_content = documents_to_display[i]
            doc_metadata = metadatas_to_display[i]
            doc_embedding = embeddings_to_display[i]

            # Robustly check embedding presence and length
            embedding_present = False
            embedding_length = 0
            if doc_embedding is not None:
                safe_doc_embedding = []
                if isinstance(doc_embedding, (list, tuple)):
                    safe_doc_embedding = doc_embedding
                elif isinstance(doc_embedding, np.ndarray):
                    safe_doc_embedding = doc_embedding.tolist()
                
                if safe_doc_embedding:
                    embedding_present = True
                    embedding_length = len(safe_doc_embedding)
                else:
                    missing_embeddings_count += 1
            else:
                missing_embeddings_count += 1 # Count as missing if None

            logger.info(f"\n--- Document {i+1} ---") # Added separator for clarity
            logger.info(f"Document ID: {doc_id}")
            logger.info(f"  Content (Full):") # Changed to indicate full content
            # Print content with indentation for readability
            for line in doc_content.splitlines():
                logger.info(f"    {line}")
            logger.info(f"  Metadata: {json.dumps(doc_metadata, indent=2)}")
            logger.info(f"  Embedding present: {embedding_present} (Length: {embedding_length})")
            if embedding_present:
                logger.debug(f"  First 5 embedding values: {safe_doc_embedding[:5]}...")

            # Collect data for summary and JSON export
            if 'section_type' in doc_metadata:
                all_section_types.append(doc_metadata['section_type'])
            if 'source_file_name' in doc_metadata:
                unique_source_files.add(doc_metadata['source_file_name'])
            
            chunk_data = {
                "id": doc_id,
                "content": doc_content, # Store full content for JSON export
                "metadata": doc_metadata,
                "embedding_present": embedding_present,
                "embedding_length": embedding_length
            }
            if embedding_present and not no_embeddings:
                chunk_data["embedding"] = safe_doc_embedding
            inspected_chunks_for_json.append(chunk_data)


        # 3. Add Output Summary
        logger.info(f"\n--- Inspection Summary ---")
        logger.info(f"Total documents inspected (displayed): {len(ids_to_display)}") # Updated count
        logger.info(f"Documents with missing embeddings: {missing_embeddings_count}")
        logger.info(f"Unique source files: {len(unique_source_files)}")
        for source_file in sorted(list(unique_source_files)):
            logger.info(f"  - {source_file}")

        if all_section_types:
            most_common_section_type = Counter(all_section_types).most_common(1)
            if most_common_section_type:
                logger.info(f"Most common section type: '{most_common_section_type[0][0]}' (Count: {most_common_section_type[0][1]})")
            else:
                logger.info("No common section types found (or all are unique).")
        else:
            logger.info("No section types found in inspected documents.")

        # 4. Optional JSON Export
        if save_json_path:
            try:
                with open(save_json_path, 'w', encoding='utf-8') as f:
                    json.dump(inspected_chunks_for_json, f, indent=2, ensure_ascii=False)
                logger.info(f"\nInspection results saved to JSON: {save_json_path}")
            except Exception as e:
                logger.error(f"Error saving inspection results to JSON file '{save_json_path}': {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"An error occurred during ChromaDB inspection: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a ChromaDB collection for document content, metadata, and embeddings."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None, # Changed default to None to signify "all"
        help="Maximum number of documents to display for general inspection (default: All). Use -1 for all."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset from which to start displaying documents (default: 0)."
    )
    parser.add_argument(
        "--source_file_name",
        type=str,
        default=None,
        help="Optional: Filter documents by a specific source file name. If provided, overrides limit/offset for this query."
    )
    parser.add_argument(
        "--save_json_path",
        type=str,
        default=None,
        help="Optional: Path to a JSON file where all inspected chunks will be saved."
    )
    parser.add_argument(
        "--no_embeddings",
        action="store_true",
        help="Optional: Skip including embedding vectors in the output for faster inspection."
    )
    parser.add_argument(
        "--show_users",
        action="store_true",
        help="Print all unique user values in the collection."
    )

    args = parser.parse_args()

    # If limit is explicitly set to -1, interpret it as None for "all"
    effective_limit = args.limit if args.limit != -1 else None

    if args.show_users:
        # Print all unique user values in the collection
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            meta = collection.get(include=["metadatas"])
            user_set = {str(m.get("user", "company_data")).strip().lower() for m in meta["metadatas"] if isinstance(m, dict) and "user" in m}
            print(f"\nUnique user values in collection ({len(user_set)}):")
            for user in sorted(user_set):
                print(f"  - {user}")
        except Exception as e:
            print(f"Error inspecting user values: {e}")
        exit(0)

    # Call the inspection function with parsed arguments
    inspect_chromadb(
        limit=effective_limit,
        offset=args.offset,
        source_file_name=args.source_file_name,
        save_json_path=args.save_json_path,
        no_embeddings=args.no_embeddings
    )
=======
import chromadb
import os
import json
from typing import Optional

import logging
import numpy as np # Added for robust type checking of embeddings
import argparse # For CLI support
from collections import Counter # For most common section_type
import traceback # For detailed error logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (MUST match your rag.py settings) ---
# Assuming BASE_DIR is the directory where your rag.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CHROMA_DB_PATH = "/mnt/c/Users/Asus/pyrotech/db/chroma_db_final"
COLLECTION_NAME = "pyrotech_docs"
# --------------------------------------------------------

def inspect_chromadb(
    limit: Optional[int] = None, # Changed default to None to signify "all"
    offset: int = 0,
    source_file_name: Optional[str] = None,
    save_json_path: Optional[str] = None,
    no_embeddings: bool = False
):
    """
    Connects to the ChromaDB, retrieves the specified collection,
    and prints information about its contents.
    
    Args:
        limit (int, optional): The maximum number of documents to retrieve for general inspection.
                               If None, retrieves all documents.
        offset (int): The offset from which to start retrieving documents.
        source_file_name (str, optional): If provided, filters documents by this source file name.
        save_json_path (str, optional): If provided, saves the inspected documents to this JSON file.
        no_embeddings (bool): If True, skips including embedding vectors in the output for faster inspection.
    """
    logger.info(f"Attempting to connect to ChromaDB at: {CHROMA_DB_PATH}")

    # Handle edge case: CHROMA_DB_PATH doesn’t exist
    if not os.path.exists(CHROMA_DB_PATH):
        logger.error(f"Error: ChromaDB path does not exist: {CHROMA_DB_PATH}")
        logger.error("Please ensure the RAG ingestion process has run successfully to create the database.")
        return # Exit cleanly

    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info("ChromaDB client initialized successfully.")

        # Get the collection
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"Accessed collection: '{COLLECTION_NAME}'")

        # 1. Check the total number of documents
        total_documents_count = collection.count()
        logger.info(f"\n--- Collection Summary ---")
        logger.info(f"Total documents in collection '{COLLECTION_NAME}': {total_documents_count}")

        if total_documents_count == 0:
            logger.warning("No documents found in the collection. Ingestion might not have completed successfully.")
            return

        # Prepare include list for ChromaDB query
        include_list = ['documents', 'metadatas']
        if not no_embeddings:
            include_list.append('embeddings')

        # 2. Query documents based on CLI arguments
        query_where_clause = {}
        if source_file_name:
            query_where_clause["source_file_name"] = source_file_name
            logger.info(f"\n--- Inspecting documents filtered by source_file_name: '{source_file_name}' ---")
            # When filtering by source_file_name, retrieve all matching documents
            results = collection.get(
                where=query_where_clause,
                include=include_list
            )
            logger.info(f"Found {len(results.get('ids', []))} documents matching source_file_name: '{source_file_name}'. Displaying all.")
        else:
            # Retrieve all documents if limit is None, otherwise apply limit/offset
            logger.info(f"\n--- Inspecting documents (Limit: {'All' if limit is None else limit}, Offset: {offset}) ---")
            
            # Fetch all documents if limit is None, otherwise use the provided limit and offset
            if limit is None:
                results = collection.get(
                    limit=total_documents_count, # Fetch all documents up to the total count
                    offset=offset,
                    include=include_list
                )
            else:
                results = collection.get(
                    limit=limit,
                    offset=offset,
                    include=include_list
                )
            
            if not results.get('ids', []):
                logger.info(f"No documents found for the specified limit/offset range (Limit: {'All' if limit is None else limit}, Offset: {offset}).")
                return


        ids_to_display = results.get('ids', [])
        documents_to_display = results.get('documents', [])
        metadatas_to_display = results.get('metadatas', [])
        embeddings_to_display = results.get('embeddings', []) if not no_embeddings else [None] * len(ids_to_display)

        inspected_chunks_for_json = [] # To store data for optional JSON export
        missing_embeddings_count = 0
        all_section_types = []
        unique_source_files = set()

        if not ids_to_display: # Check if there are documents to display after query
            if source_file_name:
                logger.info(f"No documents found with source_file_name: '{source_file_name}'")
            else:
                logger.info(f"No documents found for the specified limit/offset range.")
            return

        # Iterate over the documents selected for display
        for i in range(len(ids_to_display)):
            doc_id = ids_to_display[i]
            doc_content = documents_to_display[i]
            doc_metadata = metadatas_to_display[i]
            doc_embedding = embeddings_to_display[i]

            # Robustly check embedding presence and length
            embedding_present = False
            embedding_length = 0
            if doc_embedding is not None:
                safe_doc_embedding = []
                if isinstance(doc_embedding, (list, tuple)):
                    safe_doc_embedding = doc_embedding
                elif isinstance(doc_embedding, np.ndarray):
                    safe_doc_embedding = doc_embedding.tolist()
                
                if safe_doc_embedding:
                    embedding_present = True
                    embedding_length = len(safe_doc_embedding)
                else:
                    missing_embeddings_count += 1
            else:
                missing_embeddings_count += 1 # Count as missing if None

            logger.info(f"\n--- Document {i+1} ---") # Added separator for clarity
            logger.info(f"Document ID: {doc_id}")
            logger.info(f"  Content (Full):") # Changed to indicate full content
            # Print content with indentation for readability
            for line in doc_content.splitlines():
                logger.info(f"    {line}")
            logger.info(f"  Metadata: {json.dumps(doc_metadata, indent=2)}")
            logger.info(f"  Embedding present: {embedding_present} (Length: {embedding_length})")
            if embedding_present:
                logger.debug(f"  First 5 embedding values: {safe_doc_embedding[:5]}...")

            # Collect data for summary and JSON export
            if 'section_type' in doc_metadata:
                all_section_types.append(doc_metadata['section_type'])
            if 'source_file_name' in doc_metadata:
                unique_source_files.add(doc_metadata['source_file_name'])
            
            chunk_data = {
                "id": doc_id,
                "content": doc_content, # Store full content for JSON export
                "metadata": doc_metadata,
                "embedding_present": embedding_present,
                "embedding_length": embedding_length
            }
            if embedding_present and not no_embeddings:
                chunk_data["embedding"] = safe_doc_embedding
            inspected_chunks_for_json.append(chunk_data)


        # 3. Add Output Summary
        logger.info(f"\n--- Inspection Summary ---")
        logger.info(f"Total documents inspected (displayed): {len(ids_to_display)}") # Updated count
        logger.info(f"Documents with missing embeddings: {missing_embeddings_count}")
        logger.info(f"Unique source files: {len(unique_source_files)}")
        for source_file in sorted(list(unique_source_files)):
            logger.info(f"  - {source_file}")

        if all_section_types:
            most_common_section_type = Counter(all_section_types).most_common(1)
            if most_common_section_type:
                logger.info(f"Most common section type: '{most_common_section_type[0][0]}' (Count: {most_common_section_type[0][1]})")
            else:
                logger.info("No common section types found (or all are unique).")
        else:
            logger.info("No section types found in inspected documents.")

        # 4. Optional JSON Export
        if save_json_path:
            try:
                with open(save_json_path, 'w', encoding='utf-8') as f:
                    json.dump(inspected_chunks_for_json, f, indent=2, ensure_ascii=False)
                logger.info(f"\nInspection results saved to JSON: {save_json_path}")
            except Exception as e:
                logger.error(f"Error saving inspection results to JSON file '{save_json_path}': {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"An error occurred during ChromaDB inspection: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a ChromaDB collection for document content, metadata, and embeddings."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None, # Changed default to None to signify "all"
        help="Maximum number of documents to display for general inspection (default: All). Use -1 for all."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset from which to start displaying documents (default: 0)."
    )
    parser.add_argument(
        "--source_file_name",
        type=str,
        default=None,
        help="Optional: Filter documents by a specific source file name. If provided, overrides limit/offset for this query."
    )
    parser.add_argument(
        "--save_json_path",
        type=str,
        default=None,
        help="Optional: Path to a JSON file where all inspected chunks will be saved."
    )
    parser.add_argument(
        "--no_embeddings",
        action="store_true",
        help="Optional: Skip including embedding vectors in the output for faster inspection."
    )
    parser.add_argument(
        "--show_users",
        action="store_true",
        help="Print all unique user values in the collection."
    )

    args = parser.parse_args()

    # If limit is explicitly set to -1, interpret it as None for "all"
    effective_limit = args.limit if args.limit != -1 else None

    if args.show_users:
        # Print all unique user values in the collection
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            meta = collection.get(include=["metadatas"])
            user_set = {str(m.get("user", "company_data")).strip().lower() for m in meta["metadatas"] if isinstance(m, dict) and "user" in m}
            print(f"\nUnique user values in collection ({len(user_set)}):")
            for user in sorted(user_set):
                print(f"  - {user}")
        except Exception as e:
            print(f"Error inspecting user values: {e}")
        exit(0)

    # Call the inspection function with parsed arguments
    inspect_chromadb(
        limit=effective_limit,
        offset=args.offset,
        source_file_name=args.source_file_name,
        save_json_path=args.save_json_path,
        no_embeddings=args.no_embeddings
    )
>>>>>>> dd6123601b3b1df1e426d6eac04f58ebd5c6ba7f
