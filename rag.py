import os
import re
import uuid
import sqlite3
from rapidfuzz_fuzzy import fuzzy_match_product
import requests
import json
from rapidfuzz import fuzz

OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', "http://localhost:11434/v1/chat/completions")
import hashlib

# --- RAG system status flags ---
SYSTEM_READY = True
embedding_model_status = True

from datetime import datetime, timedelta
import chromadb
from tempfile import TemporaryDirectory
import shutil
import time
import traceback
from numpy import dot, array
from numpy.linalg import norm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import math # Added for sigmoid in confidence score
import unicodedata # Added for robust JSON parsing

# BM25 import with fallback
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logger = logging.getLogger(__name__) if 'logger' not in globals() else logger
    if 'logger' in globals():
        logger.warning("BM25Okapi not found. Hybrid search will fall back to pure vector search.")
    BM25Okapi = None

# Configuration for hybrid search
use_hybrid_search = True # Set to False to disable BM25 + semantic hybrid search
use_mmr = True  # Enable Maximal Marginal Relevance for diversity
rerank_top_n = 8  # Number of documents to rerank
TOP_N_RERANKED_RESULTS = 10  # Final number of documents to return
MMR_DIVERSITY_FACTOR = 0.7  # MMR diversity factor (0.0 = pure relevance, 1.0 = pure diversity)

from typing import Dict, Any
from unified_metadata_prompt import build_unified_metadata_prompt

def parse_metadata(llm_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and structures relevant fields from the unified LLM JSON response.
    Initializes with defaults to ensure a consistent structure.
    """
    parsed = {
        "product_name": None,
        "product_type": None, # Ensure product_type is also initialized
        "metadata_filters": {},
        "history_context": {
            "last_product": None,
            "inferred_category": None,
            "conversation_type": None
        }
    }

    if not isinstance(llm_json, dict):
        logger.warning(f"Invalid LLM JSON response for metadata parsing in parse_metadata: {llm_json}. Returning default structure.")
        return parsed

    # Handle top-level product_name and product_type directly
    parsed["product_name"] = llm_json.get("product_name")
    parsed["product_type"] = llm_json.get("product_type")

    # --- NEW: Handle specific top-level keys from recent LLM outputs ---
    if "Product Name" in llm_json and isinstance(llm_json["Product Name"], str):
        if not parsed["product_name"]:
            parsed["product_name"] = llm_json["Product Name"]

    if "Category" in llm_json and isinstance(llm_json["Category"], str):
        if not parsed["product_type"]:
            parsed["product_type"] = llm_json["Category"]

    if "Power Rating" in llm_json and (isinstance(llm_json["Power Rating"], (str, int, float))):
        power_value = llm_json["Power Rating"]
        if isinstance(power_value, str):
            num_match = re.match(r"(\d+(\.\d+)?)\s*W?", power_value.strip(), re.IGNORECASE)
            if num_match:
                try:
                    power_value = float(num_match.group(1)) if '.' in num_match.group(1) else int(num_match.group(1))
                except ValueError:
                    pass
        if power_value is not None:
            parsed["metadata_filters"]["wattage"] = power_value
    # --- END NEW TOP-LEVEL KEY HANDLING ---

    # Attributes are directly used as metadata_filters
    if "attributes" in llm_json and isinstance(llm_json["attributes"], dict):
        parsed["metadata_filters"] = {k: v for k, v in llm_json["attributes"].items() if v is not None}

    # Handle "Specifications" list if it comes from LLM
    if "Specifications" in llm_json and isinstance(llm_json["Specifications"], list):
        for spec_item in llm_json["Specifications"]:
            if isinstance(spec_item, dict):
                for k, v in spec_item.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if sub_v is not None:
                                parsed["metadata_filters"][f"{k.lower()}_{sub_k.lower()}"] = sub_v
                    elif v is not None:
                        parsed["metadata_filters"][k.lower()] = v

    # History context
    if "history_context" in llm_json and isinstance(llm_json["history_context"], dict):
        history_ctx = llm_json["history_context"]
        parsed["history_context"]["last_product"] = history_ctx.get("last_product")
        parsed["history_context"]["inferred_category"] = history_ctx.get("inferred_category")
        parsed["history_context"]["conversation_type"] = history_ctx.get("conversation_type")

    return parsed

from typing import List, Dict, Tuple, Any, Optional

# --- Attribute Key Normalization Utility ---
ATTRIBUTE_KEY_MAP = {
    "Attributes_Item Type": "item_type",
    "Attributes_Series": "series",
    "Attributes_Product Name": "product_name",
    # Extend as needed...
}

def normalize_query_attributes(query_attributes: dict) -> dict:
    normalized = {}
    for k, v in query_attributes.items():
        mapped_key = ATTRIBUTE_KEY_MAP.get(k, k.lower().replace(" ", "_"))
        normalized[mapped_key] = v
    return normalized


def format_docs_for_llm(docs: List[Dict]) -> str:
    """
    Formats a list of document dictionaries into a single string suitable for an LLM prompt.
    Includes metadata like source filename, page, document type, product name, etc.
    """
    formatted_content = []
    for doc in docs:
        header = f"--- Document: {doc['metadata'].get('source_file_name', 'Unknown Source')}"
        if doc['metadata'].get('page'):
            header += f", Page: {doc['metadata']['page']}"
        if doc['metadata'].get('document_type'):
            header += f", Type: {doc['metadata']['document_type']}"
        if doc['metadata'].get('product_name'):
            header += f", Product: {doc['metadata']['product_name']}"
        if doc['metadata'].get('model_number'):
            header += f", Model: {doc['metadata']['model_number']}"
        if doc['metadata'].get('hr_policy_category'):
            header += f", HR Policy Category: {doc['metadata']['hr_policy_category']}"
        if doc['metadata'].get('section_type'):
            header += f", Section Type: {doc['metadata']['section_type']}"
        if doc['metadata'].get('source_doc_type'):
            header += f", Source Doc Type: {doc['metadata']['source_doc_type']}"
        if doc['metadata'].get('policy_type'):
            header += f", Policy Type: {doc['metadata']['policy_type']}"
        if doc['metadata'].get('tags'):
            header += f", Tags: {', '.join(doc['metadata']['tags'])}"
        header += " ---\n"
        formatted_content.append(header + doc["content"])
    return "\n\n".join(formatted_content)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

# Import partition_auto for comprehensive document parsing
from unstructured.partition.auto import partition as partition_auto
from unstructured.partition.text import partition_text as unstructured_partition_text
from unstructured.partition.csv import partition_csv as unstructured_partition_csv
from unstructured.documents.elements import NarrativeText, Table, Title, Text, ElementMetadata, ListItem

import pandas as pd

try:
    import jsonschema
except ImportError:
    print("WARNING: 'jsonschema' not found. LLM response validation will be skipped. Install with 'pip install jsonschema'.")
    jsonschema = None

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type, before_sleep_log
from tenacity import before_sleep_log as before_before_sleep_log # Renamed to avoid conflict

import logging
logger = logging.getLogger(__name__)

import threading
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from functools import lru_cache
from typing import Callable, Any
import json  # For hashable key

def _get_hashable_key(obj: Any) -> str:
    """Converts an object into a hashable string suitable for lru_cache."""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, sort_keys=True)
    return str(obj)

@lru_cache(maxsize=100)
def cached_llm_call(prompt_key: str, original_prompt: Any, llm_call_function: Callable) -> str:
    """
    A cached wrapper for llm_call_fn.
    `prompt_key` is the hashable representation of the prompt for caching.
    `original_prompt` is the actual prompt passed to the LLM function.
    `llm_call_function` is your original LLM calling function (e.g., llm_call_fn).
    """
    return llm_call_function(original_prompt)


# --- Enhanced Fuzzy Metadata Filtering ---
def fuzzy_match_metadata(chunk_meta, filter_meta, logger=None):
    """
    Fuzzy and substring matching for metadata fields. Uses rapidfuzz.fuzz for flexible matching.
    Supports product_name, model_number, source_file, and general substring containment.
    """
    reasons = []
    # Product Name (fuzzy and substring)
    if 'product_name' in filter_meta and filter_meta['product_name']:
        chunk_val = str(chunk_meta.get('product_name', '')).lower()
        filter_val = str(filter_meta['product_name']).lower()
        if filter_val in chunk_val or chunk_val in filter_val:
            pass
        elif fuzz.token_sort_ratio(chunk_val, filter_val) >= 80:
            pass
        else:
            reasons.append(f"product_name mismatch: '{chunk_val}' vs '{filter_val}' (fuzzy score: {fuzz.token_sort_ratio(chunk_val, filter_val)})")
    # Model Number (fuzzy and substring)
    if 'model_number' in filter_meta and filter_meta['model_number']:
        chunk_val = str(chunk_meta.get('model_number', '')).lower()
        filter_val = str(filter_meta['model_number']).lower()
        if filter_val in chunk_val or chunk_val in filter_val:
            pass
        elif fuzz.token_sort_ratio(chunk_val, filter_val) >= 85:
            pass
        else:
            reasons.append(f"model_number mismatch: '{chunk_val}' vs '{filter_val}' (fuzzy score: {fuzz.token_sort_ratio(chunk_val, filter_val)})")
    # Source File (substring)
    if 'source_file' in filter_meta and filter_meta['source_file']:
        chunk_val = str(chunk_meta.get('source_file', '')).lower()
        filter_val = str(filter_meta['source_file']).lower()
        if filter_val not in chunk_val:
            reasons.append(f"source_file mismatch: '{chunk_val}' does not contain '{filter_val}'")
    return reasons


def post_filter_chunks(chunks, filter_metadata=None, keywords=None, logger=None, strict=False):
    """
    Enhanced post-retrieval filtering for RAG chunks using fuzzy metadata and keyword logic.
    If strict is False (default), do not filter out chunks—attach reasons and keep all for reranking.
    If strict is True, filter out as before.
    - filter_metadata: dict with fields like product_name, model_number, source_file
    - keywords: list of keywords to check in chunk text
    - Returns filtered (or annotated) chunks
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    if not filter_metadata and not keywords:
        return chunks
    filtered = []
    for chunk in chunks:
        meta = chunk.get('metadata', {})
        text = chunk.get('content', '')
        reasons = []
        # Metadata fuzzy filtering
        if filter_metadata:
            reasons += fuzzy_match_metadata(meta, filter_metadata, logger)
        # Keyword containment check
        if keywords:
            missing = [kw for kw in keywords if kw.lower() not in text.lower()]
            if missing:
                reasons.append(f"missing keywords: {missing}")
        if strict:
            if reasons:
                logger.debug(f"Chunk filtered out: {reasons} | Meta: {meta} | Text: {text[:80]}")
            else:
                filtered.append(chunk)
        else:
            chunk['filter_reasons'] = reasons
            filtered.append(chunk)
    return filtered


class EmbeddingCache:
    def __init__(self, db_path: str, expected_dim: int = 768):
        self.db_path = db_path
        self.expected_dim = expected_dim
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    embedding_vector BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.close()

    def get_or_set(self, text: str, embedding_fn) -> list:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT embedding_vector FROM embeddings WHERE text_hash = ?", (text_hash,))
            result = cursor.fetchone()
            if result:
                cached_embedding = json.loads(result[0])
                if len(cached_embedding) == self.expected_dim:
                    cursor.close()
                    return cached_embedding
                else:
                    logger.warning(f"Cached embedding for hash {text_hash} has dimension {len(cached_embedding)}, expected {self.expected_dim}. Recalculating.")
            # Not cached or wrong dimension, generate and cache
            embedding = embedding_fn(text)
            cursor.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, embedding_vector, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (text_hash, json.dumps(embedding))
            )
            self.conn.commit()
            cursor.close()
            return embedding

    def save(self):
        with self.lock:
            self.conn.commit()
            self.conn.close()

class RerankCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rerank_cache (
                    query_hash TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    rerank_score REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (query_hash, document_hash, model_name)
                )
            ''')
            cursor.close()

    def get_or_set(self, query: str, doc: str, model_name: str, rerank_fn) -> float:
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        doc_str = json.dumps(doc, sort_keys=True)
        doc_hash = hashlib.md5(doc_str.encode("utf-8")).hexdigest()


        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT rerank_score FROM rerank_cache WHERE query_hash = ? AND document_hash = ? AND model_name = ?", (query_hash, doc_hash, model_name))
            result = cursor.fetchone()
            if result:
                cursor.close()
                return result[0]
            # Not cached, compute and cache
            score = reranker_cache.get_or_set(query_text, doc, RERANKER_MODEL_NAME,
                lambda q, d, m: rerank_documents(q, [d], all_retrieved_embeddings)[0]["score"]
            )                              
            cursor.execute(
                "INSERT OR REPLACE INTO rerank_cache (query_hash, document_hash, model_name, rerank_score, timestamp) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (query_hash, doc_hash, model_name, score)
            )
            self.conn.commit()
            cursor.close()
            return score

    def save(self):
        with self.lock:
            self.conn.commit()
            self.conn.close()

def _extract_and_process_attributes_improved(prompt_template: str, input_text: str, source_description: str, doc_type: str = "default", llm_call_fn=None) -> Dict:
    """
    Safely extracts attributes using LLM, processes them, and validates against a schema.
    doc_type parameter helps select the right schema for validation.
    llm_call_fn is a function that takes a prompt and returns the LLM's response.
    """
    try:
        if llm_call_fn is None:
            raise ValueError("llm_call_fn must be provided to _extract_and_process_attributes_improved")
        json_response_str = llm_call_fn(prompt_template + "\n" + input_text)
        extracted_data = safe_parse_llm_json_response(json_response_str)  # Use your robust parser
        final_processed_data = extract_metadata_with_schema(extracted_data, doc_type, llm_call_fn=llm_call_fn)
        if not isinstance(final_processed_data, dict):
            logger.warning(f"Schema-based extraction for {source_description} returned non-dict: {final_processed_data}. Initializing to empty dict.")
            final_processed_data = {}
        if final_processed_data:
            final_processed_data = filter_noisy_metadata(final_processed_data)
            final_processed_data = _combine_metadata_fields(final_processed_data)
            final_processed_data = _validate_and_correct_extracted_metadata(final_processed_data)
        logger.info(f"Extracted {source_description} attributes (final): {final_processed_data}")
        return final_processed_data
    except Exception as e:
        logger.error(f"Error extracting {source_description} attributes: {e}", exc_info=True)
        return {}

def _parse_range_filter_value(value: Any) -> Any:
    if isinstance(value, str):
        cleaned_value = value.replace('W', '').replace('V', '').replace('K', '').replace('Lumen', '').strip()
        match_range = re.match(r"(\d+)-(\d+)", cleaned_value)
        if match_range:
            lower, upper = int(match_range.group(1)), int(match_range.group(2))
            return {"$gte": lower, "$lte": upper}
        if cleaned_value.replace('.', '', 1).isdigit():
            try:
                return int(cleaned_value) if '.' not in cleaned_value else float(cleaned_value)
            except ValueError:
                pass
    return value

from generate_product_name_variants import generate_product_name_variants

def _create_chromadb_filter(attributes: Dict) -> Dict:
    """
    Flattens nested dicts (e.g., metadata) and builds a ChromaDB filter dict with only simple types.
    Only includes keys with simple (str, int, float, bool) values or supported filter ops.
    Warns and skips unsupported types.
    """
    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Flatten all nested dicts (including metadata and all other keys)
    flat_attributes = dict(attributes)
    # Flatten metadata if present
    if 'metadata' in flat_attributes and isinstance(flat_attributes['metadata'], dict):
        meta_flat = flatten(flat_attributes['metadata'])
        del flat_attributes['metadata']
        flat_attributes.update(meta_flat)
    # Flatten all remaining nested dicts recursively
    keys_to_flatten = [k for k, v in flat_attributes.items() if isinstance(v, dict)]
    for k in keys_to_flatten:
        flat = flatten(flat_attributes[k], parent_key=k)
        del flat_attributes[k]
        flat_attributes.update(flat)
    # Log the final filter for debugging
    logger.debug(f"[ChromaDB Filter] Final flattened attributes for filter: {flat_attributes}")

    chromadb_filter = {}
    # Special handling for product_name: use variants for $in filter if present
    if 'product_name' in flat_attributes and isinstance(flat_attributes['product_name'], str):
        variants_info = generate_product_name_variants(flat_attributes['product_name'])
        variants = variants_info.get('variants', [])
        if variants:
            chromadb_filter['product_name'] = {"$in": variants}
        del flat_attributes['product_name']
    for key, value in flat_attributes.items():
        if value is None or value == '':
            continue
        # Only allow simple types or supported operators
        if isinstance(value, (str, int, float, bool)):
            chromadb_filter[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            chromadb_filter[key] = {"$in": value}
        elif isinstance(value, dict) and any(op in value for op in ["$gt", "$gte", "$lt", "$lte", "$in", "$ne"]):
            chromadb_filter[key] = value
        else:
            logger.warning(f"[ChromaDB Filter] Skipping unsupported filter key '{key}': type={type(value)}, value={value}")
    return chromadb_filter


def extract_metadata_with_llm(text):
    """
    Calls your Ollama LLM to extract metadata as a JSON object from a document chunk.
    Returns a dict with keys: product_name, product_type, section_type (all lowercased if present).
    """
    prompt = (
        "Extract the following metadata as a JSON object from the text below:\n"
        "- product_name (string, lowercased, if present)\n"
        "- product_type (string, lowercased, if present)\n"
        "- section_type (string, lowercased, if present)\n"
        "Text:\n"
        f"{text}\n"
        "Respond ONLY with a valid JSON object."
    )
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": "mistral",  # Replace if you use another model
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    content = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        return {}

def llm_rerank(query, docs):
    prompt = (
        f"Given the user query and the following documents, select the most relevant ones for answering the question.\n"
        f"Query: {query}\n"
        f"Documents:\n"
    )
    for i, doc in enumerate(docs):
        prompt += f"Document {i+1}:\n{doc}\n"
    prompt += "Respond ONLY with the most relevant document(s) as plain text."
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": "mistral",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=60
    )
    return response.json()["choices"][0]["message"]["content"]

# Optional: Import spacy for advanced sentence splitting
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("INFO: spaCy 'en_core_web_sm' model loaded for smart chunking.")
except ImportError:
    print("WARNING: spaCy not found. Install with 'pip install spacy' and 'python -m spacy download en_core_web_sm' for smart chunking.")
    nlp = None
except Exception as e:
    print(f"WARNING: Could not load spaCy model 'en_core_web_sm': {e}. Falling back to regex splitting.")
    nlp = None

# Optional: Import sentence_transformers for local embedding fallback
try:
    from sentence_transformers import SentenceTransformer
    local_embedding_model = None
    print("INFO: sentence_transformers library found. Local embedding fallback is possible.")
except ImportError:
    print("WARNING: 'sentence_transformers' not found. Local embedding fallback will be disabled. Install with 'pip install sentence-transformers'.")
    SentenceTransformer = None

def _parse_html_table_to_key_value_list(html_table_string: str) -> str:
    """
    Parses a simple HTML table string into a list of key-value pairs formatted
    as "Key: Value" lines, suitable for LLM consumption.
    Assumes a row structure like: <tr><td>Key</td><td><br/></td><td>Value</td><td><br/></td></tr>
    """
    if not isinstance(html_table_string, str) or "<table" not in html_table_string.lower():
        # Not an HTML table, return the original string
        return html_table_string

    parsed_lines = []
    # Find all <td> contents. re.DOTALL allows '.' to match newlines.
    td_contents = re.findall(r'<td>(.*?)</td>', html_table_string, re.IGNORECASE | re.DOTALL)

    # The example HTML table structure suggests 4 <td> tags per row:
    # <td>Key</td><td><br/></td><td>Value</td><td><br/></td>
    # We extract content from index 0 (Key) and 2 (Value) from every group of 4 <td>s.
    for i in range(0, len(td_contents), 4):
        if i + 2 < len(td_contents):
            key = td_contents[i].strip()
            value = td_contents[i+2].strip()

            # Clean up potential <br/> tags that might have been captured in key/value
            key = re.sub(r'<br\s*/>', '', key, flags=re.IGNORECASE).strip()
            value = re.sub(r'<br\s*/>', '', value, flags=re.IGNORECASE).strip()

            if key and value:  # Only add if both key and value are non-empty
                parsed_lines.append(f"{key}: {value}")

    if parsed_lines:
        return "\n".join(parsed_lines)
    return html_table_string  # Return original if no recognizable pairs extracted

def _process_json_for_html_tables(data: Any) -> Any:
    """
    Recursively processes JSON data (dict or list) to find and convert
    HTML table strings into key-value list format using _parse_html_table_to_key_value_list.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(v, str):
                # Apply HTML parsing if it looks like an HTML table
                if "<table" in v.lower() and "<td" in v.lower():
                    new_data[k] = _parse_html_table_to_key_value_list(v) # <--- This calls your other function
                else:
                    new_data[k] = v
            elif isinstance(v, (dict, list)):
                # Recurse for nested dictionaries or lists
                new_data[k] = _process_json_for_html_tables(v)
            else:
                new_data[k] = v
        return new_data
    elif isinstance(data, list):
        new_list = []
        for item in data:
            if isinstance(item, str):
                # Apply HTML parsing if it looks like an HTML table
                if "<table" in item.lower() and "<td" in item.lower():
                    new_list.append(_parse_html_table_to_key_value_list(item)) # <--- This calls your other function
                else:
                    new_list.append(item)
            elif isinstance(item, (dict, list)):
                # Recurse for nested dictionaries or lists
                new_list.append(_process_json_for_html_tables(item))
            else:
                new_list.append(item)
        return new_list
    else:
        return data

# ==============================================================================
# ====== CONFIGURATION SETTINGS (CRITICAL: Adjust for your environment) ======
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_RAW_DOCS_FOLDER = os.path.join(BASE_DIR, 'docs_raw')
PROCESSED_DOCS_FOLDER = os.path.join(BASE_DIR, 'docs_processed')
CHROMA_DB_PATH = "/mnt/c/Users/ASUS/pyrotech/db/chroma_db_final"
EMBEDDING_CACHE_DB_PATH = os.getenv('EMBEDDING_CACHE_DB_PATH', os.path.join(BASE_DIR, "db", "embedding_cache.db"))
RERANK_CACHE_DB_PATH = os.getenv('RERANK_CACHE_DB_PATH', os.path.join(BASE_DIR, "db", "rerank_cache.db"))

COLLECTION_NAME = "pyrotech_docs"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"[RAG STARTUP] Using CHROMA_DB_PATH: {CHROMA_DB_PATH}")
logger.info(f"[RAG STARTUP] Using COLLECTION_NAME: {COLLECTION_NAME}")

OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', "http://localhost:11434/v1/chat/completions")
OLLAMA_EMBED_API_URL = os.getenv("OLLAMA_EMBED_API_URL",
    OLLAMA_API_URL.replace("/v1/chat/completions", "/api/embeddings")
)
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', "mistral")

# ====== UPDATED CONFIGURATION FOR ACCURACY ======
# Changed EMBEDDING_MODEL_NAME to "nomic-embed-text" and adjusted expected dimension
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")  # Now using 'nomic-embed-text'
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")    # 98% rerank precision
SECONDARY_RERANKER_MODEL_NAME = os.getenv("SECONDARY_RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NUM_DOCS_FOR_SECOND_RERANK = int(os.getenv("NUM_DOCS_FOR_SECOND_RERANK", 10)) # Number of docs to pass to secondary reranker

# Adjusted thresholds for higher recall initially, relying on reranker for precision
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.50))  # Slightly lower for broader initial retrieval
MIN_RERANK_SCORE_THRESHOLD = float(os.getenv("MIN_RERANK_SCORE_THRESHOLD", 0.3))  # Keep strict for final context
# Hybrid search weights
BM25_WEIGHT = 0.7  # Prioritize exact keyword matches
VECTOR_WEIGHT = 0.3

RATE_LIMIT_MAX_CALLS = 10
RATE_LIMIT_WINDOW_SEC = 60
user_access_log = defaultdict(list)

NUM_RAG_RESULTS = int(os.getenv("NUM_RAG_RESULTS", 25))  # Increased for better reranking # Increased for higher initial recall
TOP_N_RERANKED_RESULTS = int(os.getenv("TOP_N_RERANKED_RESULTS", 5)) # Increased to allow more context if relevant
MMR_DIVERSITY_FACTOR = 0.5 # Balance between relevance and diversity

from rapidfuzz_fuzzy import fuzzy_match_product

# NEW: Confidence Thresholds for Iterative Retrieval and Clarification
LOW_CONFIDENCE_THRESHOLD_CLARIFY = 0.4453 

# --- Generalized Metadata Extraction Helpers ---

generalized_key_synonyms = [
    (["model", "model_number", "model_no"], "model_number"),
    (["customer", "customer_name"], "customer_name"),
    (["date", "invoice_date"], "date"),
    (["brand_name", "brand"], "brand"),
    # Add more as needed for your doc types
]

def _combine_metadata_fields(metadata: Dict) -> Dict:
    """
    Combines fragmented or duplicate metadata fields into unified fields. This generic version merges keys with similar names or values.
    """
    processed_metadata = metadata.copy()
    for group, canonical in generalized_key_synonyms:
        values = [processed_metadata.pop(k, None) for k in group if k in processed_metadata]
        values = [v for v in values if v]
        if values:
            processed_metadata[canonical] = values[0]  # Prefer first non-empty
    return processed_metadata

def _validate_and_correct_extracted_metadata(extracted_data: Dict) -> Dict:
    """
    Performs generic sanity checks and normalization on extracted metadata.
    """
    corrected_data = extracted_data.copy()
    # Normalize all values to strings, join string lists
    for k, v in corrected_data.items():
        if isinstance(v, (int, float)):
            corrected_data[k] = str(v)
        elif isinstance(v, list) and v and all(isinstance(i, str) for i in v):
            corrected_data[k] = ", ".join(v)
    return corrected_data

def extract_document_metadata(text_content: str, filename: str, elements: list, llm_call_fn) -> dict:
    """
    Extracts generalized metadata from text content using an LLM.
    Prioritizes structured data (tables, lists) and uses LLM for general text.
    Applies post-processing to combine fragmented fields.
    Robust to all LLM response types and noisy outputs.
    """
    from rag_llm_json_utils import safe_parse_llm_json_response, filter_noisy_metadata
    from datetime import datetime
    metadata = {"source_file_name": filename, "processing_date": datetime.now().isoformat()}
    structured_text = ""
    for el in elements:
        if isinstance(el, dict) and el.get("type") in ("table", "list"):
            structured_text += el.get("text", "") + "\n"
    prompt = LLM_PROMPT_TEMPLATES["table_metadata_extraction_system"] + "\n" + (structured_text if structured_text.strip() else text_content)
    extracted_data = safe_parse_llm_json_response(llm_call_fn(prompt))
    # Defensive: unwrap or merge list of dicts, warn for other types
    if isinstance(extracted_data, list):
        if len(extracted_data) == 1 and isinstance(extracted_data[0], dict):
            extracted_data = extracted_data[0]
            logger.info("Unwrapped single dictionary from LLM list response for structured text metadata extraction.")
        elif all(isinstance(item, dict) for item in extracted_data):
            merged = {}
            for d in extracted_data:
                merged.update(d)
            extracted_data = merged
            logger.info(f"Merged {len(extracted_data)} dicts from LLM list response for structured text metadata extraction.")
        else:
            logger.warning(f"LLM returned a list with non-dict elements for metadata extraction: {repr(extracted_data)}")
            extracted_data = {}
    elif not isinstance(extracted_data, dict):
        logger.warning(f"LLM returned non-dict metadata type: {type(extracted_data)}. Value: {repr(extracted_data)}")
        extracted_data = {}
    if extracted_data:
        extracted_data = filter_noisy_metadata(extracted_data)
        extracted_data = _combine_metadata_fields(extracted_data)
        extracted_data = _validate_and_correct_extracted_metadata(extracted_data)
        metadata.update(extracted_data)
    else:
        logger.warning(f"No valid metadata extracted for file {filename}.")
    return metadata

# If confidence falls below this, try broader search or ask for clarification
VERY_LOW_CONFIDENCE_THRESHOLD_NO_ANSWER = 0.3 # If confidence falls below this, definitively state no answer

# Chunking Parameters - REVERTED to more robust values, with special handling for markdown tables
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", 50)) # REVERTED
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", 800))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", 150))
MIN_ALPHANUM_RATIO = float(os.getenv("MIN_ALPHANUM_RATIO", 0.5)) # REVERTED

# Batching for embeddings
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

# Rerank Cache (existing feature)
RERANK_CACHE = {}

# LLM Prompt Templates (Config-
LLM_PROMPT_TEMPLATES = {
    "system_instruction": (
        "You are a highly accurate, concise, and document-grounded AI assistant specializing in providing information about Pyrotech products and HR policies. "
        "Your responses **MUST BE STRICTLY BASED ONLY ON THE PROVIDED CONTEXT AND CHAT HISTORY.** "
        "**DO NOT USE ANY EXTERNAL KNOWLEDGE, INFER INFORMATION, OR MAKE ASSUMPTIONS.** "
        "**Crucially, adhere to numerical values, units, and exact phrasing found in the document.** "
        "If the answer cannot be found EXPLICITLY in the provided context or history, you MUST clearly state: "
        "'I do not have enough information from the provided documents to answer this precisely.' "
        "Do NOT provide general knowledge, speculative answers, or apologies if the specific information is not in the context.\n\n"
        "**Your output format is CRITICAL for readability and accuracy. Always provide answers in a point-wise, neat, and clean manner using Markdown:**\n"
        "* **Use bullet points (`* ` for main points, `  * ` for sub-points) or numbered lists (`1. `) for all factual information.**\n"
        "* **Ensure each distinct piece of information is on a NEW LINE.** Avoid combining multiple facts into a single wrapped paragraph line.\n"
        "* **Insert a blank line (double newline `\\n\\n`) between major sections or distinct bulleted/numbered lists for clarity.**\n"
        "* **For tabular data, always present it as a Markdown table.**\n"
        "* **If a policy or feature has multiple sub-points, use nested bullet points.**\n"
        "* **Output MUST be readable in any markdown renderer or plain terminal.**\n\n"
        "**Context from retrieved documents (each chunk is clearly delimited with source information):**\n"
    ),
    "no_context_fallback": "No specific document context was retrieved for this query. Therefore, I cannot provide a precise answer based on the documents. Please try rephrasing your question or provide more specific keywords.",
    "chat_history_header": "\n\n**Chat History (for conversational context):**\n",
    "no_chat_history": "No prior chat history.\n",
    "incremental_prompt_no_docs": "I do not have enough information from the provided documents to answer this precisely.",
    "clarification_prompt": "I'm having a bit of trouble finding a precise answer. Could you clarify if you're asking about product specifications, HR policies, or something else specific?",
    "query_attribute_extraction_system": (
        "Analyze the provided user query and extract **ONLY** key entities, product names, specific model numbers, "
        "product variants (e.g., power consumption ranges like '10-45W', '50-70W'), or HR policy categories/topics "
        "that could be used for filtering or direct lookup in a document database.\n\n"
        "**Retain the exact, full name of products or policies, avoiding any generalization or abbreviation unless explicitly stated in the text.**\n\n"
        "**CRITICAL: You MUST respond ONLY with a valid JSON object. NO EXCEPTIONS.**\n"
        "**DO NOT include any conversational text, explanations, or code blocks (like ```json) outside the JSON object. Just the JSON.**\n"
        "**The JSON object MUST start with an '{' and end with a '}'.**\n" # <-- MODIFIED: Enforce object, not array
        "Each key-value pair MUST be followed by a comma (','), EXCEPT for the very last key-value pair immediately "
        "before the closing curly brace '}'.\n"
        "No preamble, no postamble, no explanations, no conversational filler. Your response should be nothing but the JSON.\n"
        "If no filterable attributes are identified, return an empty JSON object {}.\n"
        "Ensure all extracted values are strings. If a value is a number, convert it to a string.\n"
        "For model numbers and specific variant identifiers, ensure the exact string is captured.\n\n"
        "Prioritize extracting: `product_name`, `model_number`, `total_power_consumption_w`, `supply_voltage_vac`, "
        "`ip_rating`, `hr_policy_category`, `leave_type`, `safety_topic`, `document_type`, `section_type`.\n"
        "⚠️ Do not include any explanatory or filler text outside the JSON object. Your output must end at the closing brace }.\n\n"
        "INCORRECT (do NOT do this):\n"
        "[\n    \"Medical Products\",\n    \"Covid Products\"\n]\n\n"
        "CORRECT (do this):\n"
        "{\n    \"product_type\": \"Medical Products\",\n    \"product_type_2\": \"Covid Products\"\n}\n"
    ),
    "table_metadata_extraction_system": (
        "You are an AI assistant specialized in extracting structured key-value metadata from text, "
        "especially from markdown tables or semi-structured lists. Your goal is to extract ALL relevant key-value pairs from the "
        "provided text, regardless of document type. Do not limit yourself to product fields; extract any meaningful attributes, identifiers, or specifications.\n\n"
        "**CRITICAL: Always capture the FULL, EXACT value for each key. Do NOT generalize or abbreviate names or values.**\n\n"
        "Respond ONLY with a valid JSON object containing all extracted key-value pairs. If no key-value pairs are identified, return an empty JSON object {}.\n\n"
        "**Instructions:**\n"
        "1. Identify all distinct parameters/keys and their values.\n"
        "2. Convert keys to lowercase, use underscores instead of spaces/special characters.\n"
        "3. Ensure all values are strings. If a value is a number, convert it to a string.\n"
        "4. No explanations, no conversational filler, no preamble/postamble—just the JSON.\n"
        "5. If a value is missing, use null.\n"
        "6. If a field is repeated, use the most specific or complete value.\n"
        "⚠️ Your output must end at the closing brace }.\n"
    ),
    "chat_history_entity_extraction_system": (
        "You are an AI assistant that analyzes chat history to identify the core product, model, or HR policy topic being discussed. "
        "Extract only the most relevant, persistent entity (e.g., a specific product name, model number, or HR policy category) "
        "that the user is continuously asking about. If no clear persistent entity is found, return an empty JSON object {}.\n\n"
        "**Retain the exact, full name of products or policies, avoiding any generalization or abbreviation unless explicitly stated in the text.**\n\n"
        "**CRITICAL: You MUST respond ONLY with a valid JSON object. NO EXCEPTIONS.**\n"
        "**DO NOT include any conversational text, explanations, or code blocks (like ```json) outside the JSON object. Just the JSON.**\n"
        "**The JSON object MUST start with an '{' and end with a '}'.**\n" # <-- MODIFIED: Enforce object, not array
        "Each key-value pair MUST be followed by a comma (','), EXCEPT for the very last key-value pair immediately "
        "before the closing curly brace '}'.\n"
        "No preamble, no postamble, no explanations, no conversational filler. Your response should be nothing but the JSON.\n"
        "If no persistent entity is identified, return an empty JSON object {}.\n"
        "Ensure all extracted values are strings.\n\n"
        "⚠️ Do not include any explanatory or filler text outside the JSON object. Your output must end at the closing brace }.\n\n"
        "INCORRECT (do NOT do this):\n"
        "[\n    \"Fan\",\n    \"Light\"\n]\n\n"
    ),
    "query_attribute_extraction_examples": (
        "Examples of desired output:\n"
        "# Correct answers from documents\n"
        "Document Content: \"What are the key differences between Bitcoin and Blockchain?\"\n"
        "{\"topic\": \"Bitcoin, Blockchain\", \"category\": \"Cryptocurrency\"}\n"
        "Document Content: \"lights with 10-45W or 50-70W power consumption\"\n"
        "{\"product_type\": \"lights\", \"total_power_consumption_w\": \"10-45W, 50-70W\"}\n"
        "Document Content: \"products with IP-66 protection\"\n"
        "{\"ingress_protection\": \"IP-66\"}\n"
        "Document Content: \"street lights with dimensions 472X221X71mm and 60Hz frequency\"\n"
        "{\"product_type\": \"street light\", \"dimensions_lxwxh_mm\": \"472X221X71mm\", \"frequency_hz\": \"60Hz\"}\n"
        "Document Content: \"LED STREET LIGHT (GLASS MODEL) Driver Efficiency\"\n"
        "{\"product_name\": \"LED STREET LIGHT (GLASS MODEL)\", \"attribute\": \"Driver Efficiency\"}\n"
        "Document Content: \"PORTABLE POWER STATION APPLICATION\"\n"
        "{\"product_name\": \"PORTABLE POWER STATION\", \"topic\": \"application\"}\n"
        "Document Content: \"what is Color temperature All-in-One LED Solar Street-light PH-11-A-L-WXOA\"\n"
        "{\"product_name\": \"All-in-One LED Solar Street-light\", \"model_number\": \"PH-11-A-L-WXOA\", \"correlated_color_temperature_k\": \"Color temperature\"}\n"
        "Document Content: \"what is enclosure of led street light ?\"\n"
        "{\"product_type\": \"led street light\", \"attribute\": \"enclosure\"}\n"
        "Document Content: \"PH-11-A-L-WXOB\"\n"
        "{\"model_number\": \"PH-11-A-L-WXOB\"}\n"
        "Document Content: \"7W LED Solar Street-light total power consumption\"\n"
        "{\"product_name\": \"7W LED Solar Street-light\", \"total_power_consumption_w\": \"7W\", \"section_type\": \"technical_specifications\"}\n"
        "Document Content: \"TECHNICAL SPECIFICATIONS\\nTotal Power Consumption (W), +/-10%\\n7W \\u00b110%\\nSupply Voltage (V AC)\\n240V AC \\u00b110%,50Hz\\u00b15%\\nPower Factor\\n>0.95\\nTHD\\n<15%\\nOperating Temperature (\u00b0c)\\n-20\\u00b0 to 50\\u00b0\\nWorking Humidity (RH)\\n5%-90% RH\\nSurge Protection\\n4KV\\nLED Luminous Efficacy (Lm/W)\\n>130\\nSystem Luminous Flux (Lumen)\\n>700\\nCCT (K)\\n5700K-6500K\\nCRI\\n>70\\nBeam Angel\\n120\\u00b0\\nIP\\n54\\nEnclosure\\nDie-Cast Aluminium\\nLED Life (Hrs)\\n>50000\"\n"
        "{\n"
        "    \"total_power_consumption_w\": \"7W \\u00b110%\",\n"
        "    \"supply_voltage_vac\": \"240V AC \\u00b110%, 50Hz\\u00b15%\",\n"
        "    \"power_factor\": \">0.95\",\n"
        "    \"thd\": \"<15%\",\n"
        "    \"operating_temperature_c\": \"-20\\u00b0 to 50\\u00b0\",\n"
        "    \"\"working_humidity_rh\": \"5%-90% RH\",\n"
        "    \"surge_protection_kv\": \"4KV\",\n"
        "    \"led_luminous_efficacy_lm_w\": \">130\",\n"
        "    \"system_luminous_efficacy_lm_w\": \">100\",\n"
        "    \"system_luminous_flux_lumen\": \">700\",\n"
        "    \"cct_k\": \"5700K-6500K\",\n"
        "    \"cri\": \">70\",\n"
        "    \"beam_angle_deg\": \"120\\u00b0\",\n"
        "    \"ip_rating\": \"54\",\n"
        "    \"enclosure_material\": \"Die-Cast Aluminium\",\n"
        "    \"led_life_hours\": \">50000\",\n"
        "    \"section_type\": \"technical_specifications\"\n"
        "}\n"
        "Document Content: \"Product: LED Hand Lamp. Model: PK-11-D-L-WXOA. Total Power: 7W. Voltage: 230V AC.\"\n"
        "{\"product_name\": \"LED Hand Lamp\", \"model_number\": \"PK-11-D-L-WXOA\", \"total_power_consumption_w\": \"7W\", \"supply_voltage_vac\": \"230V AC\", \"section_type\": \"technical_specifications\"}\n"
        "Document Content: \"TECHNICAL SPECIFICATIONS\\nVoltage: 220V\\nPower: 50W\\nProtection: IP65\"\n"
        "{\"supply_voltage_vac\": \"220V\", \"total_power_consumption_w\": \"50W\", \"ip_rating\": \"IP65\", \"section_type\": \"technical_specifications\"}\n"
        "Document Content: \"Dimensions: 100x50x20mm. Weight: 1.5kg. Color: Black.\"\n"
        "{\"dimensions_lxwxh_mm\": \"100x50x20mm\", \"weight_kg\": \"1.5kg\", \"color\": \"Black\", \"section_type\": \"dimensions\"}\n"
        "}\n"
        "# New examples for HR policies\n"
        "Document Content: \"What are the rules for privilege leave?\"\n"
        "{\"hr_policy_category\": \"LEAVES AND HOLIDAYS\", \"leave_type\": \"Privilege Leave\", \"document_type\": \"HR Policy\", \"section_type\": \"hr_leaves_holidays\"}\n"
        "Document Content: \"Tell me about workplace safety.\"\n"
        "{\"hr_policy_category\": \"HEALTH AND SAFETY\", \"safety_topic\": \"Workplace Safety\", \"document_type\": \"HR Policy\", \"section_type\": \"hr_health_safety\"}\n"
        "Document Content: \"What are the office timings?\"\n"
        "{\"hr_policy_category\": \"WORKING HOURS\", \"topic\": \"Office timings\", \"document_type\": \"HR Policy\", "
        "\"section_type\": \"hr_working_hours\"}\n"
        "Document Content: \"What are the different types of drivers in Pyrotech's catalog?\"\n"
        "{\"document_type\": \"Driver Catalogue\", \"topic\": \"types of drivers\"}\n"
        "Document Content: \"What are the specifications of the Blood Bank Refrigerator?\"\n"
        "{\"product_name\": \"Blood Bank Refrigerator\", \"section_type\": \"technical_specifications\"}\n"
        "Document Content: \"Tell me about the BLDC Table Fan.\"\n"
        "{\"product_name\": \"BLDC Table Fan\"}\n"
        "Document Content: \"Tell me about the LED Street Light.\"\n"
        "{\"product_name\": \"LED Street Light\"}\n"
        "Document Content: \"What about the LED Hand Lamp?\"\n"
        "{\"product_name\": \"LED Hand Lamp\"}\n"
        "Document Content: \"Specifications for the LED Highbay Light.\"\n"
        "{\"product_name\": \"LED Highbay Light\"}\n"
        "Document Content: \"What is the capital of France?\"\n"
        "{}\n"
        "Document Content: \"Tell me about Pyrotech's stock price.\"\n"
        "{}\n"
        "Document Content: \"What is the average employee salary?\"\n"
        "{}\n"
        "Document Content: \"Who is the CEO of Pyrotech?\"\n"
        "{}"
    ),
    "answer_validation_system": ( # NEW: Prompt for LLM self-correction/validation
        "You are an AI assistant tasked with validating if a given 'Answer' is **FULLY AND STRICTLY** supported by the 'Context' provided. "
        "Your goal is to identify if the answer contains any information not present in the context, or if it makes inferences or assumptions. "
        "You MUST respond ONLY with a JSON object containing a 'valid' boolean and a 'reason' string.\n\n"
        "**CRITICAL: You MUST respond ONLY with a valid JSON object. NO EXCEPTIONS.**\n"
        "**DO NOT include any conversational text, explanations, or code blocks (like ```json) outside the JSON object. Just the JSON.**\n"
        "**The JSON object MUST start with an '{' or '[' and end with a '}' or ']'.**\n" # Updated for array output
        "Each key-value pair MUST be followed by a comma (','), EXCEPT for the very last key-value pair immediately "
        "before the closing curly brace '}'.\n"
        "No preamble, no postamble, no explanations, no conversational filler. Your response should be nothing but the JSON.\n"
        "If the answer is fully supported by the context, set 'valid' to true. "
        "If ANY part of the answer is NOT explicitly supported by the context, set 'valid' to false and explain why.\n\n"
        "Example of desired output (ONLY the JSON, no ```json or other text):\n"
        "Context: \"The quick brown fox jumps over the lazy dog.\"\n"
        "Answer: \"A brown fox quickly jumped over a dog.\"\n"
        "{\"valid\": true, \"reason\": \"All information is directly supported by the context.\"}\n"
        "Context: \"The quick brown fox jumps over the lazy dog.\"\n"
        "Answer: \"The quick brown fox jumps over the lazy cat.\"\n"
        "{\"valid\": false, \"reason\": \"The animal 'cat' is not mentioned in the context; it says 'dog'.\"}\n"
        "Context: \"Pyrotech's LED Street Light Model X has 100W power consumption.\"\n"
        "Answer: \"The LED Street Light Model X has 100W power consumption and is very efficient.\"\n"
        "{\"valid\": false, \"reason\": \"The context mentions 100W power consumption but does not state it is 'very efficient'.\"}\n"
        "Context: \"The company's leave policy states employees get 15 days of privilege leave per year.\"\n"
        "Answer: \"Employees are entitled to 15 days of privilege leave annually, which is standard.\"\n"
        "{\"valid\": false, \"reason\": \"The context states 15 days of privilege leave, but does not say it is 'standard'.\"}"
        "⚠️ Do not include any explanatory or filler text outside the JSON object. Your output must end at the closing brace } or bracket ]." # Reinforcement
    )
}

# ==============================================================================
# ====== SYSTEM INITIALIZATION & MODEL LOADING (Executes on app startup) ======

# Global cache instantiation for embedding and reranker caches
embedding_cache = EmbeddingCache(db_path=EMBEDDING_CACHE_DB_PATH)
reranker_cache = RerankCache(db_path=RERANK_CACHE_DB_PATH)

# ==============================================================================
embedding_model_status = False
reranker_tokenizer = None
reranker_model = None
secondary_reranker_tokenizer = None # Added secondary reranker tokenizer
secondary_reranker_model = None # Added secondary reranker model
collection = None # Ensure 'collection' is initialized to None globally
SYSTEM_READY = False
INITIALIZATION_ERRORS = []
embedding_cache_conn = None
USE_LOCAL_EMBEDDING_MODEL = False
rerank_cache_conn = None

def init_embedding_cache_db():
    """Initializes the SQLite database for embedding caching."""
    global embedding_cache_conn
    try:
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_DB_PATH), exist_ok=True)
        # Set isolation_level to None to disable implicit transactions for DML statements
        # This gives us explicit control with commit()/rollback()
        embedding_cache_conn = sqlite3.connect(EMBEDDING_CACHE_DB_PATH, check_same_thread=False, isolation_level=None)
        cursor = embedding_cache_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding_vector BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # No commit() needed after DDL if isolation_level is None, but good practice for clarity
        # embedding_cache_conn.commit()
        logger.info(f"Embedding cache database initialized at {EMBEDDING_CACHE_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding cache database: {e}")
        traceback.print_exc()
        embedding_cache_conn = None

def get_embedding_from_cache(text_hash: str) -> Optional[List[float]]:
    """Retrieves an embedding from the cache."""
    if embedding_cache_conn:
        try:
            cursor = embedding_cache_conn.cursor()
            cursor.execute("SELECT embedding_vector FROM embeddings WHERE text_hash = ?", (text_hash,))
            result = cursor.fetchone()
            cursor.close() # Explicitly close the cursor
            if result:
                # Ensure cached embedding matches the expected dimension (768 for nomic-embed-text)
                cached_embedding = json.loads(result[0])
                if len(cached_embedding) != 768: # Updated to 768
                    logger.warning(f"Cached embedding for hash {text_hash} has dimension {len(cached_embedding)}, expected 768. Recalculating.")
                    # Force recalculation if cached embedding has wrong dimension
                    pass 
                else:
                    return cached_embedding # Return the cached embedding
        except Exception as e:
            logger.warning(f"Error retrieving embedding from cache for hash {text_hash}: {e}")
    return None

def store_embedding_in_cache(embedding_hash: str, embedding: List[float]):
        """
        Stores an embedding in the SQLite cache.
        """
        if not embedding_cache_conn:
            logger.warning("Embedding cache connection not established. Skipping cache store.")
            return

        try:
            # Removed explicit BEGIN TRANSACTION;
            # With isolation_level=None, each execute is its own transaction unless explicitly wrapped.
            # We'll rely on the implicit transaction for the single INSERT/REPLACE and then commit.
            cursor = embedding_cache_conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, embedding_vector, timestamp) VALUES (?, ?, ?)",
                (embedding_hash, json.dumps(embedding), datetime.now().isoformat())
            )
            embedding_cache_conn.commit() # This commits the implicit transaction
            cursor.close() # Explicitly close the cursor
            logger.debug(f"Stored embedding for hash {embedding_hash} in cache.")
        except sqlite3.OperationalError as e:
            logger.warning(f"Error storing embedding in cache for hash {embedding_hash}: {e}")
            if embedding_cache_conn:
                # Only rollback if a transaction was implicitly started and failed before commit
                # This might still raise "no transaction is active" if the error happened before the implicit start
                try:
                    embedding_cache_conn.rollback()
                except sqlite3.OperationalError as rb_e:
                    logger.warning(f"Rollback failed for embedding cache (hash {embedding_hash}): {rb_e}")
        except Exception as e:
            logger.error(f"Unexpected error storing embedding in cache for hash {embedding_hash}: {e}")
            if embedding_cache_conn:
                try:
                    embedding_cache_conn.rollback()
                except sqlite3.OperationalError as rb_e:
                    logger.warning(f"Rollback failed for embedding cache (hash {embedding_hash}): {rb_e}")

def init_rerank_cache_db():
    """Initializes the SQLite database for rerank caching."""
    global rerank_cache_conn
    try:
        os.makedirs(os.path.dirname(RERANK_CACHE_DB_PATH), exist_ok=True)
        rerank_cache_conn = sqlite3.connect(RERANK_CACHE_DB_PATH, check_same_thread=False, isolation_level=None)
        cursor = rerank_cache_conn.cursor()
        # MODIFIED: Added model_name to primary key
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rerank_cache (
                 query_hash TEXT NOT NULL,
                 document_hash TEXT NOT NULL,
                 model_name TEXT NOT NULL,
                 rerank_score REAL NOT NULL,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 PRIMARY KEY (query_hash, document_hash, model_name)
               )
        """)
        # rerank_cache_conn.commit() # No commit needed after DDL if isolation_level is None
        logger.info(f"Rerank cache database initialized at {RERANK_CACHE_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize rerank cache database: {e}")
        traceback.print_exc()
        rerank_cache_conn = None

def get_rerank_score_from_cache(query_hash: str, doc_hash: str, model_name: str) -> Optional[float]:
    """Retrieves a rerank score from the cache."""
    if rerank_cache_conn:
        try:
            cursor = rerank_cache_conn.cursor()
            # MODIFIED: Added model_name to query
            cursor.execute("SELECT rerank_score FROM rerank_cache WHERE query_hash = ? AND document_hash = ? AND model_name = ?", (query_hash, doc_hash, model_name))
            result = cursor.fetchone()
            cursor.close() # Explicitly close the cursor
            if result:
                return result[0]
        except Exception as e:
            logger.warning(f"Error retrieving rerank score from cache (query: {query_hash}, doc: {doc_hash}, model: {model_name}): {e}")
    return None

def store_rerank_score_in_cache(query_hash: str, doc_hash: str, model_name: str, score: float):
    """Stores a rerank score in the cache."""
    if rerank_cache_conn:
        try:
            cursor = rerank_cache_conn.cursor()
            # MODIFIED: Added model_name to insert
            cursor.execute(
                "INSERT OR REPLACE INTO rerank_cache (query_hash, doc_hash, model_name, rerank_score) VALUES (?, ?, ?, ?)",
                (query_hash, doc_hash, model_name, score)
            )
            rerank_cache_conn.commit()
            cursor.close() # Explicitly close the cursor
        except sqlite3.OperationalError as e:
            logger.warning(f"Error storing rerank score in cache for hash {query_hash}: {e}")
            if rerank_cache_conn:
                try:
                    rerank_cache_conn.rollback()
                except sqlite3.OperationalError as rb_e:
                    logger.warning(f"Rollback failed for rerank cache (hash {query_hash}): {rb_e}")
        except Exception as e:
            logger.warning(f"Unexpected error storing rerank score in cache for hash {query_hash}: {e}")
            if rerank_cache_conn:
                try:
                    rerank_cache_conn.rollback()
                except sqlite3.OperationalError as rb_e:
                    logger.warning(f"Rollback failed for rerank cache (hash {query_hash}): {rb_e}")

def load_model_offline(model_name: str, model_class, tokenizer_class):
    """Attempts to load a model from the Hugging Face local cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    model_path_segment = model_name.replace("/", "--")
    local_model_dir = os.path.join(cache_dir, model_path_segment)

    if os.path.exists(local_model_dir):
        logger.info(f"Attempting to load model '{model_name}' from local cache: {local_model_dir}")
        try:
            tokenizer = tokenizer_class.from_pretrained(local_model_dir)
            model = model_class.from_pretrained(local_model_dir)
            logger.info(f"Successfully loaded model '{model_name}' from local cache.")
            return tokenizer, model
        except Exception as e:
            logger.warning(f"Failed to load model '{model_name}' from local cache '{local_model_dir}': {e}")
            return None, None
    else:
        logger.info(f"Local cache directory for '{model_name}' not found: {local_model_dir}")
        return None, None


def init_rag_system():
    global collection, embedding_model_status, reranker_tokenizer, reranker_model, \
           secondary_reranker_tokenizer, secondary_reranker_model, SYSTEM_READY, \
           INITIALIZATION_ERRORS, USE_LOCAL_EMBEDDING_MODEL, local_embedding_model

    INITIALIZATION_ERRORS = []
    SYSTEM_READY = False
    USE_LOCAL_EMBEDDING_MODEL = False

    init_embedding_cache_db()
    init_rerank_cache_db()

    try:
        # MOVE THE CLIENT INITIALIZATION HERE:
        logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
        dir_name = os.path.dirname(CHROMA_DB_PATH)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH) # <--- ADD THIS LINE HERE!
        
        # For nomic-embed-text, the expected dimension is 768.
        expected_dim = 768 # Changed to 768
        logger.info(f"Expected embedding dimension for '{EMBEDDING_MODEL_NAME}': {expected_dim}")

        # Try to get the collection. If it exists, check its dimension.
        # If it doesn't exist or has the wrong dimension, delete and recreate.
        try:
            collection = chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            # --- Check if ChromaDB is already populated ---
            if collection.count() > 0:
                print(f"ChromaDB collection '{COLLECTION_NAME}' already contains {collection.count()} documents. Skipping data ingestion.")
                SYSTEM_READY = True # Indicate system is ready as DB is populated
                # return # Exit the function early if data is already present - REMOVED TO ALLOW HEALTH CHECKS
            else:
                print(f"ChromaDB collection '{COLLECTION_NAME}' is empty. Starting data ingestion...")

            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' initialized. Current count: {collection.count()}")

            # Verify the dimension of the existing collection
            # Query a single document to infer the dimension
            if collection.count() > 0:
                sample_doc = collection.peek(limit=1)
                
                embeddings_data = sample_doc.get('embeddings')

                if embeddings_data is not None:
                    # If it's a numpy array, convert it to a list for consistent handling
                    if isinstance(embeddings_data, np.ndarray):
                        embeddings_list = embeddings_data.tolist()
                    else:
                        embeddings_list = embeddings_data

                    if len(embeddings_list) > 0:
                        first_embedding = embeddings_list[0]
                        # Ensure first_embedding is actually a list/array of numbers
                        if isinstance(first_embedding, (list, np.ndarray)):
                            current_dimension = len(first_embedding)
                            if current_dimension != expected_dim:
                                # Removed automatic deletion on dimension mismatch
                                error_message = (f"Existing collection '{COLLECTION_NAME}' has dimension {current_dimension}, expected {expected_dim}. "
                                                 "Please manually delete the ChromaDB directory "
                                                 f"'{CHROMA_DB_PATH}' if you wish to re-initialize the database with the new embedding model, "
                                                 "or revert to the previous embedding model.")
                                logger.critical(error_message,
                                                extra={"collection_name_log": COLLECTION_NAME, "error_detail": str(e), "expected_dimension": expected_dim})
                                INITIALIZATION_ERRORS.append(error_message)
                                SYSTEM_READY = False # Ensure system is not ready if this critical error occurs
                                collection = None # Ensure collection is explicitly None if it's in an invalid state
                                # return # Removed to allow other parts of init to run if possible
                            logger.info(f"Existing ChromaDB collection '{COLLECTION_NAME}' confirmed to be {expected_dim}-dimensional.")
                        else:
                            logger.warning(f"Collection '{COLLECTION_NAME}' is not empty, but first embedding is not a list/array. Type: {type(first_embedding)}. Assuming {expected_dim}-D for now.")
                    else:
                        logger.warning(f"Collection '{COLLECTION_NAME}' is not empty but contains no embeddings to verify dimension. Assuming {expected_dim}-D for now.")
                else:
                    logger.warning(f"Collection '{COLLECTION_NAME}' is not empty but 'embeddings' key is missing or None. Assuming {expected_dim}-D for now.")
            else:
                logger.info(f"Collection '{COLLECTION_NAME}' is empty. Dimension will be set on first add.")

        except chromadb.errors.InvalidDimensionException as e:
            error_message = (f"ChromaDB collection '{COLLECTION_NAME}' has an invalid dimension: {e}. "
                             f"Expected {expected_dim}-dimensional embeddings, but found a different dimension. "
                             "Automatic deletion is disabled. Please manually delete the ChromaDB directory "
                             f"'{CHROMA_DB_PATH}' if you wish to re-initialize the database with the new embedding model, "
                             "or revert to the previous embedding model.")
            logger.critical(error_message,
                            extra={"collection_name_log": COLLECTION_NAME, "error_detail": str(e), "expected_dimension": expected_dim})
            INITIALIZATION_ERRORS.append(error_message)
            SYSTEM_READY = False # Ensure system is not ready if this critical error occurs
            collection = None # Ensure collection is explicitly None if it's in an invalid state
            # return # Removed to allow other parts of init to run if possible
        except Exception as e:
            # Handle other potential errors during collection access
            raise Exception(f"Failed to access or create ChromaDB collection '{COLLECTION_NAME}': {e}") from e


        logger.info(f"Verifying Ollama Embedding Model: {EMBEDDING_MODEL_NAME} at {OLLAMA_EMBED_API_URL}")
        try:
            # MODIFICATION: Use OLLAMA_EMBED_API_URL for the embedding check
            response = requests.post(
                OLLAMA_EMBED_API_URL, 
                json={"model": EMBEDDING_MODEL_NAME, "prompt": "test embedding"},
                timeout=10
            )
            response.raise_for_status()
            
            response_json = response.json()
            embedding = response_json.get("embedding")
            
            if embedding == None: # Corrected from '===' to '=='
                raise Exception("Ollama embedding response missing 'embedding' field. Check model name or Ollama setup.")
            
            # Enforce expected_dim embedding output
            if len(embedding) != expected_dim:
                raise ValueError(f"Only {expected_dim}-dim embeddings are allowed. Ollama model '{EMBEDDING_MODEL_NAME}' returned {len(embedding)}-dim embeddings. Fix your embedding source.")

            embedding_model_status = True
            logger.info(f"Ollama embedding model '{EMBEDDING_MODEL_NAME}' is available and dimension matches (768D).")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.critical(f"Could not connect to Ollama server or timed out at {OLLAMA_EMBED_API_URL}: {e}. Disabling embedding generation.")
            embedding_model_status = False
            INITIALIZATION_ERRORS.append(f"Ollama connection failed: {e}")
        except ValueError as e:
            logger.critical(f"Embedding dimension mismatch with Ollama: {e}. Disabling embedding generation.")
            embedding_model_status = False
            INITIALIZATION_ERRORS.append(f"Ollama dimension mismatch: {e}")
        except Exception as e:
            logger.critical(f"Error checking Ollama embedding model '{EMBEDDING_MODEL_NAME}': {e}. Disabling embedding generation.")
            embedding_model_status = False
            INITIALIZATION_ERRORS.append(f"Ollama model check failed: {e}")
        
        # Removed the fallback to local embedding model as per strict requirement
        if not embedding_model_status:
            logger.critical("Ollama embedding model is not available or has dimension mismatch. Embedding generation will be disabled.")
            local_embedding_model = None


        logger.info(f"Initializing Primary Reranker Model: {RERANKER_MODEL_NAME}")
        try:
            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
            reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
            logger.info("Primary reranker model loaded successfully from Hugging Face.")
        except Exception as e_online:
            logger.warning(f"Failed to load primary reranker model '{RERANKER_MODEL_NAME}' from Hugging Face (online): {e_online}. Attempting local cache.")
            reranker_tokenizer, reranker_model = load_model_offline(RERANKER_MODEL_NAME, AutoModelForSequenceClassification, AutoTokenizer)
            
            if reranker_model is None:
                raise Exception(f"Failed to load primary reranker model '{RERANKER_MODEL_NAME}' from both online and local cache. "
                                f"Ensure the model files are in the Hugging Face cache at '~/.cache/huggingface/transformers/models--{RERANKER_MODEL_NAME.replace('/', '--')}/' "
                                f"or that you have internet access to download it. Original online error: {e_online}")

        # NEW: Initialize Secondary Reranker Model
        logger.info(f"Initializing Secondary Reranker Model: {SECONDARY_RERANKER_MODEL_NAME}")
        try:
            secondary_reranker_tokenizer = AutoTokenizer.from_pretrained(SECONDARY_RERANKER_MODEL_NAME)
            secondary_reranker_model = AutoModelForSequenceClassification.from_pretrained(SECONDARY_RERANKER_MODEL_NAME)
            logger.info("Secondary reranker model loaded successfully from Hugging Face.")
        except Exception as e_online:
            logger.warning(f"Failed to load secondary reranker model '{SECONDARY_RERANKER_MODEL_NAME}' from Hugging Face (online): {e_online}. Attempting local cache.")
            secondary_reranker_tokenizer, secondary_reranker_model = load_model_offline(SECONDARY_RERANKER_MODEL_NAME, AutoModelForSequenceClassification, AutoTokenizer)
            
            if secondary_reranker_model is None:
                logger.error(f"Failed to load secondary reranker model '{SECONDARY_RERANKER_MODEL_NAME}' from both online and local cache. "
                                f"Secondary reranking will be disabled. Original online error: {e_online}")
                INITIALIZATION_ERRORS.append(f"Secondary reranker load failed: {e_online}")


        SYSTEM_READY = True
        logger.info("RAG System components initialized successfully.")

    except Exception as e:
        SYSTEM_READY = False
        INITIALIZATION_ERRORS.append(f"RAG System initialization failed: {e}")
        logger.critical(f"RAG System initialization failed: {e}")
        traceback.print_exc()

# Configure tenacity for retries with exponential backoff
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[500, 502, 503, 504, 429],
    allowed_methods={'POST', 'GET'}
)
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# ==============================================================================
# ====== HELPER FUNCTIONS (Sanitization, Logging, Rate Limiting) ======
# ==============================================================================

def sanitize_text(text: Any) -> str:
    """Sanitizes text by converting to string, handling encoding, and replacing HTML special characters."""
    if text is None:
        return ""
    s = str(text)
    # Convert smart quotes to straight quotes
    s = s.replace('"', '"').replace("'", "'")
    # Replace tabs with spaces to prevent JSON parsing issues
    s = s.replace('\t', '    ') # 4 spaces for a tab
    s = s.encode('utf-8', errors='replace').decode('utf-8')
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip()
    return s

def write_log(filename_prefix: str, content: str, username: str = "system"):
    """Writes content to a log file with a timestamp and username."""
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{username}.txt")
    
    try:
        with open(log_filename, "w", encoding="utf-8", errors='replace') as f:
            f.write(content)
        logger.info(f"Log written to: {log_filename}", extra={"log_file_path": log_filename, "prefix_name": filename_prefix, "log_username": username})
    except Exception as e:
        logger.error(f"ERROR writing log to file '{log_filename}': {e}", extra={"log_file_path": log_filename, "error_detail": str(e)})
        traceback.print_exc()

def log_chat_message(username_str: str, session_id_str: str, role_str: str, message_content: str, session_title: Optional[str] = None):
    """Logs chat messages to the database via auth module."""
    auth_log_chat_message_to_db(username_str, session_id_str, role_str, message_content, session_title)

def is_rate_limited(username: str, max_calls: int = RATE_LIMIT_MAX_CALLS, window_sec: int = RATE_LIMIT_WINDOW_SEC) -> bool:
    """Checks if a user is rate-limited based on recent API calls."""
    now = time.time()
    logs = user_access_log[username]
    user_access_log[username] = [t for t in logs if now - t < window_sec]
    
    if len(user_access_log[username]) >= max_calls:
        logger.warning(f"RATE LIMIT: User '{username}' is rate-limited (too many calls in {window_sec}s).",
                       extra={"rate_limit_user": username, "max_allowed_calls": max_calls, "window_seconds": window_sec})
        return True
    
    user_access_log[username].append(now)
    return False

# ==============================================================================
# ====== CORE RAG & LLM FUNCTIONS (Embedding, Reranking, API Calls) ======
# ==============================================================================

def calculate_alphanum_ratio(text: str) -> float:
    """Calculates the ratio of alphanumeric characters to total characters in a string."""
    if not text:
        return 0.0
    alphanum_count = sum(c.isalnum() for c in text)
    return alphanum_count / len(text)

# Helper function to detect markdown table documents
def is_markdown_table_doc(file_name: str) -> bool:
    """Checks if the filename indicates a structured markdown table document."""
    return "markdown_tables" in file_name.lower()

# List of critical technical phrases to whitelist chunks
CRITICAL_TECHNICAL_PHRASES = [
    "voltage", "wattage", "power factor", "ip rating", "order code",
    "total power consumption", "supply voltage", "luminous efficacy",
    "system luminous flux", "cct", "cri", "beam angle", "enclosure",
    "led life", "dimensions", "frequency", "current", "model number",
    "protection", "operating temperature", "humidity"
]

def contains_critical_phrase(chunk_text: str) -> bool:
    """Checks if a chunk contains any critical technical phrases."""
    chunk_lower = chunk_text.lower()
    for phrase in CRITICAL_TECHNICAL_PHRASES:
        if phrase.lower() in chunk_lower:
            return True
    return False

# Helper function to fix single quotes to double quotes for JSON parsing
def _fix_single_quotes_in_json(text: str) -> str:
    """
    Attempts to convert single quotes to double quotes in a JSON-like string,
    but only where they likely represent string delimiters or keys, not apostrophes.
    """
    # Replace single quotes used as string delimiters or around keys
    # This regex tries to match patterns where single quotes are used like JSON double quotes.
    # It's a heuristic and might not catch all cases or might misfire on some apostrophes.
    fixed_text = re.sub(r"([{,]\s*)'([^']+)'(\s*[:,}\]])", r'\1"\2"\3', text)
    fixed_text = re.sub(r"^\s*'([^']+)'(\s*[:,}\]])", r'"\1"\2', fixed_text) # For starting single-quoted key/value
    fixed_text = re.sub(r"([{,]\s*)'([^']+)'\s*$", r'\1"\2"', fixed_text) # For ending single-quoted value
    fixed_text = re.sub(r"^\s*'([^']+)'\s*$", r'"\1"', fixed_text) # If the whole string is a single-quoted value

    # Escape unescaped single quotes within string values that are now double-quoted
    # This is a tricky one, as it assumes the outer quotes are now double quotes.
    # It tries to escape single quotes that are not already escaped by a backslash
    # and are within what looks like a double-quoted string.
    fixed_text = re.sub(r'("[^"\\]*)(\')([^"\\]*")', r'\1\\\2\3', fixed_text)
    return fixed_text

# Function to safely parse LLM JSON responses with optional schema validation
# Function to safely parse LLM JSON responses with optional schema validation
# Function to safely parse LLM JSON responses with optional schema validation
def safe_parse_llm_json_response(json_string: str, schema: Optional[Dict] = None) -> Any:
    """
    Safely parses a JSON string response from an LLM.
    Attempts to fix common LLM JSON formatting issues and validates against a schema if provided.
    """
    original_json_string = json_string 
    
    # Defensive check: If input is empty, return empty dict immediately
    if not original_json_string or not original_json_string.strip():
        logger.debug("safe_parse_llm_json_response received empty or whitespace-only string. Returning empty dict.")
        return {}

    # Pre-parse filter to remove trailing lists and non-kv lines
    json_string = remove_trailing_lists_and_non_kv(json_string)
    cleaned_json_string = str(json_string).strip()
    
    # --- NEW (Point 9): Log invisible control characters for debugging ---
    for idx, ch in enumerate(cleaned_json_string):
        if unicodedata.category(ch)[0] == "C" and ch not in ['\n', '\r', '\t']: # Exclude common newlines/tabs
            logger.debug(f"Invisible control character detected at index {idx}: U+{ord(ch):04X} in snippet: '{cleaned_json_string[max(0, idx-10):min(len(cleaned_json_string), idx+10)]}'")

    # Remove all non-printable ASCII characters and replace non-breaking spaces with regular spaces
    cleaned_json_string = ''.join(ch if unicodedata.category(ch)[0] != "C" or ch in ['\n', '\r', '\t'] else ' ' for ch in cleaned_json_string)
    cleaned_json_string = cleaned_json_string.replace('\xa0', ' ') # Replace non-breaking space specifically
    cleaned_json_string = cleaned_json_string.replace('\t', ' ') # Replace tabs with spaces

    # --- NEW (Point 1): Smart quotes already handled in sanitize_text, but ensure here too for safety ---
    cleaned_json_string = cleaned_json_string.replace('"', '"').replace("'", "'").replace('"', '"').replace("'", "'")
    cleaned_json_string = cleaned_json_string.replace('"', '"').replace("'", "'").replace('"', '"').replace("'", "'")
    cleaned_json_string = cleaned_json_string.replace('`', '"') # Replace backticks with double quotes

    # Handle common invalid escape sequences directly.
    cleaned_json_string = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cleaned_json_string)

    # --- FIX FOR ERROR B: Malformed LLM output with dictionary inside list ---
    # This pattern matches: ["key": "value", ...]
    # And transforms it into: {"key": "value", ...}
    # This fix should only apply if the *outermost* structure is a list with key-value pairs directly inside.
    # The regex is adjusted to be more specific to this malformation.
    if re.match(r'^\s*\[\s*"[^"]+":\s*".+?"', cleaned_json_string, re.DOTALL): # <-- MODIFIED REGEX
        logger.warning(f"Detected malformed LLM list-dict JSON. Attempting to convert to object. Snippet: '{cleaned_json_string[:100]}'")
        try:
            # Remove the outer brackets and add curly braces
            # This assumes the content within the brackets is already comma-separated key-value pairs
            # e.g., ["key1": "val1", "key2": "val2"] -> {"key1": "val1", "key2": "val2"}
            cleaned_json_string = "{" + cleaned_json_string.strip()[1:-1].strip() + "}"
            logger.debug(f"Converted malformed list-dict to object: '{cleaned_json_string[:100]}'")
        except Exception as e:
            logger.error(f"Failed to convert malformed list-dict JSON: {e}. Original snippet: '{cleaned_json_string[:100]}'")
            # If conversion fails, proceed with original cleaned_json_string, hoping other fallbacks work.


    # --- ADD HARD JSON BLOCK CUTTER HERE (1/3) ---
    # Cut everything after the last closing brace/bracket to isolate a valid block
    # This regex looks for the last occurrence of a balanced {} or [] block.
    last_json_match = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', cleaned_json_string, re.DOTALL)
    if last_json_match:
        # Prefer full object/array match if available
        if last_json_match.group(1): # It's an object {}
            cleaned_json_string = last_json_match.group(1)
        elif last_json_match.group(2): # It's an array []
            cleaned_json_string = last_json_match.group(2)
        logger.debug(f"Hard JSON block cutter applied. Snippet: {cleaned_json_string[:200]}...")
    else:
        logger.debug("Hard JSON block cutter found no full JSON match, proceeding with original string.")

    # --- Attempt 1: Direct JSON parse (fastest if clean) ---
    try:
        # No need for unicodedata.normalize here if done at the start
        # No need for .replace here if done at the start
        json_to_parse = cleaned_json_string
        # NEW LINE 1: Remove unencodable characters (already done above, but harmless for double-check)
        json_to_parse = json_to_parse.encode('utf-8', 'ignore').decode('utf-8') 
        # NEW LINE 2: Escape newlines, carriage returns and tabs (already done above, but harmless for double-check)
        json_to_parse = json_to_parse.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') 

        parsed_json = json.loads(json_to_parse)
        if schema and jsonschema:
            try:
                jsonschema.validate(instance=parsed_json, schema=schema)
                logger.debug("LLM JSON response (direct parse) validated successfully against schema.")
                return parsed_json
            except jsonschema.exceptions.ValidationError as ve:
                logger.warning(f"LLM JSON response validation error (direct parse): {ve.message}. Trying other methods.",
                               extra={"json_validation_error": ve.message, "invalid_json_string": json_to_parse[:200]})
        else:
            return parsed_json
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON decode failed: {e}. Trying structural recovery. Snippet: '{json_to_parse[max(0, e.pos-50):e.pos+50]}...'")
        
        # --- Structural Recovery (Optional Advanced) - Integrated here ---
        try:
            fixed_for_incomplete = json_to_parse
            # If the error is "Expecting value" or "Expecting ',' delimiter" and it's near the end,
            # try to drop the last incomplete field or trailing comma.
            if e.msg.startswith("Expecting value") or e.msg.startswith("Expecting ',' delimiter"):
                # Try to find the last colon followed by an incomplete value and remove it
                # or a trailing comma without a subsequent key-value pair
                fixed_for_incomplete = re.sub(r',\s*"[^"]*"\s*:\s*$', '', json_to_parse[:e.pos])
                fixed_for_incomplete = re.sub(r',\s*$', '', fixed_for_incomplete) # Remove trailing comma if any
                # Add back content after error pos, hoping it's a closing brace/bracket
                fixed_for_incomplete += json_to_parse[e.pos:]

                # Re-apply the hard cutter after trimming an incomplete field
                last_json_match_fixed = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', fixed_for_incomplete, re.DOTALL)
                if last_json_match_fixed:
                    if last_json_match_fixed.group(1):
                        fixed_for_incomplete = last_json_match_fixed.group(1)
                    elif last_json_match_fixed.group(2):
                        fixed_for_incomplete = last_json_match_fixed.group(2)

            parsed_json = json.loads(fixed_for_incomplete)
            if schema and jsonschema:
                jsonschema.validate(instance=parsed_json, schema=schema)
            logger.warning(f"Partial JSON recovered (direct parse fallback with structural recovery): {parsed_json}. Returning partial data.",
                           extra={"partial_json_recovered": parsed_json})
            return parsed_json
        except json.JSONDecodeError as inner_e:
            logger.warning(f"Structural recovery failed (direct parse fallback): {inner_e}. Continuing. Snippet: '{fixed_for_incomplete[max(0, inner_e.pos-50):inner_e.pos+50]}'")
        except Exception as inner_exp:
            logger.warning(f"An unexpected error occurred during structural recovery (direct parse fallback): {inner_exp}. Continuing. Snippet: '{fixed_for_incomplete[:100]}'")
    except Exception as e:
        logger.warning(f"An unexpected error occurred during direct JSON parsing: {e}. Trying other methods. Snippet: '{json_to_parse[:100]}...'")


    # --- Attempt 2: Extract from Markdown code block (if not already found) ---
    # This specifically looks for a JSON object or array embedded within markdown code fences.
    json_code_block_match = re.search(r'```json\s*\n(.*?)\n```', original_json_string, re.DOTALL)
    if json_code_block_match:
        extracted_block_content = json_code_block_match.group(1).strip()
        logger.debug(f"Extracted JSON from markdown code block. Snippet: '{extracted_block_content[:100]}...'")
        
        # Apply pre-processing to the extracted block content
        extracted_block_content = re.sub(r'//.*$', '', extracted_block_content, flags=re.MULTILINE) # Remove JS-style comments
        extracted_block_content = re.sub(r',\s*([}\]])', r'\1', extracted_block_content) # Remove trailing commas
        
        # Apply general cleaning to extracted block
        extracted_block_content = unicodedata.normalize('NFKC', extracted_block_content)
        extracted_block_content = extracted_block_content.replace('"', '"').replace("'", "'").replace('"', '"').replace("'", "'")
        extracted_block_content = extracted_block_content.replace('`', '"')
        extracted_block_content = "".join(
           ch if unicodedata.category(ch)[0] != "C" or ch in ['\n', '\r', '\t'] else ' '
           for ch in extracted_block_content
        )

        extracted_block_content = extracted_block_content.replace('\xa0', ' ')
        extracted_block_content = extracted_block_content.replace('\t', ' ')
        extracted_block_content = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', extracted_block_content)

        # --- ADD HARD JSON BLOCK CUTTER HERE (2/3) ---
        last_json_match_block = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', extracted_block_content, re.DOTALL)
        if last_json_match_block:
            if last_json_match_block.group(1):
                extracted_block_content = last_json_match_block.group(1)
            elif last_json_match_block.group(2):
                extracted_block_content = last_json_match_block.group(2)
            logger.debug(f"Hard JSON block cutter applied to extracted block. Snippet: {extracted_block_content[:200]}...")
        else:
            logger.debug("Hard JSON block cutter found no full JSON match in extracted block, proceeding with original block.")

        try:
            parsed_json = json.loads(extracted_block_content)
            if schema and jsonschema:
                try:
                    jsonschema.validate(instance=parsed_json, schema=schema)
                    logger.debug("LLM JSON response (from code block) validated successfully against schema.")
                    return parsed_json
                except jsonschema.exceptions.ValidationError as ve:
                    logger.warning(f"LLM JSON response validation error for code block: {ve.message}. Trying other methods.",
                                   extra={"json_validation_error": ve.message, "invalid_json_string": extracted_block_content[:200]})
            else:
                return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON from embedded block: {e}. Trying structural recovery. Snippet: '{extracted_block_content[max(0, e.pos-50):e.pos+50]}...'")
            # --- Structural Recovery for embedded block ---
            try:
                fixed_for_incomplete_block = extracted_block_content
                if e.msg.startswith("Expecting value") or e.msg.startswith("Expecting ',' delimiter"):
                    fixed_for_incomplete_block = re.sub(r',\s*"[^"]*"\s*:\s*$', '', extracted_block_content[:e.pos])
                    fixed_for_incomplete_block = re.sub(r',\s*$', '', fixed_for_incomplete_block)
                    fixed_for_incomplete_block += extracted_block_content[e.pos:]

                    last_json_match_inner_block = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', fixed_for_incomplete_block, re.DOTALL)
                    if last_json_match_inner_block:
                        if last_json_match_inner_block.group(1):
                            fixed_for_incomplete_block = last_json_match_inner_block.group(1)
                        elif last_json_match_inner_block.group(2):
                            fixed_for_incomplete_block = last_json_match_inner_block.group(2)

                parsed_json = json.loads(fixed_for_incomplete_block)
                if schema and jsonschema:
                    jsonschema.validate(instance=parsed_json, schema=schema)
                logger.warning(f"Partial JSON recovered (embedded block fallback with structural recovery): {parsed_json}. Returning partial data.",
                               extra={"partial_json_recovered": parsed_json})
                return parsed_json
            except json.JSONDecodeError as sub_e:
                logger.warning(f"Embedded block structural recovery failed: {sub_e}. Continuing. Snippet: '{fixed_for_incomplete_block[max(0, sub_e.pos-50):sub_e.pos+50]}'")
            except Exception as sub_exp:
                logger.warning(f"An unexpected error occurred during embedded block structural recovery: {sub_exp}. Continuing. Snippet: '{fixed_for_incomplete_block[:100]}'")
        except Exception as e:
            logger.warning(f"An unexpected error occurred during JSON parsing from code block: {e}. Trying other methods. Snippet: '{extracted_block_content[:100]}...'")
     

    # --- Attempt 3: Extract any embedded JSON block (not necessarily in markdown) ---
    # This specifically looks for a JSON object or array embedded within other text.
    embedded_json_match = re.search(r'(\{.*?\}|\[.*?\])', original_json_string, re.DOTALL)
    if embedded_json_match:
        extracted_content = embedded_json_match.group(0) # Get the full matched object or array
        logger.debug(f"Found general embedded JSON. Snippet: '{extracted_content[:100]}...'")
        
        # Apply pre-processing to the extracted block content
        extracted_content = re.sub(r'//.*$', '', extracted_content, flags=re.MULTILINE)
        extracted_content = re.sub(r',\s*([}\]])', r'\1', extracted_content)
        
        # Apply general cleaning
        extracted_content = unicodedata.normalize('NFKC', extracted_content)
        extracted_content = extracted_content.replace('"', '"').replace("'", "'").replace('"', '"').replace("'", "'")
        extracted_content = extracted_content.replace('`', '"')
        extracted_content = "".join(
            ch if unicodedata.category(ch)[0] != "C" or ch in ['\n', '\r', '\t'] else ' '
            for ch in extracted_content
        )

        extracted_content = extracted_content.replace('\xa0', ' ')
        extracted_content = extracted_content.replace('\t', ' ')
        extracted_content = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', extracted_content)

        # --- ADD HARD JSON BLOCK CUTTER HERE (3/3) ---
        last_json_match_embedded = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', extracted_content, re.DOTALL)
        if last_json_match_embedded:
            if last_json_match_embedded.group(1):
                extracted_content = last_json_match_embedded.group(1)
            elif last_json_match_embedded.group(2):
                extracted_content = last_json_match_embedded.group(2)
            logger.debug(f"Hard JSON block cutter applied to general embedded content. Snippet: {extracted_content[:200]}...")
        else:
            logger.debug("Hard JSON block cutter found no full JSON match in general embedded content, proceeding.")

        try:
            parsed_json = json.loads(extracted_content)
            if schema and jsonschema:
                try:
                    jsonschema.validate(instance=parsed_json, schema=schema)
                    logger.debug("LLM JSON response (general embedded block) validated successfully against schema.")
                    return parsed_json
                except jsonschema.exceptions.ValidationError as ve:
                    logger.warning(f"LLM JSON response validation error for general embedded block: {ve.message}. Trying other methods.",
                                   extra={"json_validation_error": ve.message, "invalid_json_string": extracted_content[:200]})
            else:
                return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON from general embedded block: {e}. Trying structural recovery. Snippet: '{extracted_content[max(0, e.pos-50):e.pos+50]}...'")
            # --- Structural Recovery for general embedded block ---
            try:
                fixed_for_incomplete_general = extracted_content
                if e.msg.startswith("Expecting value") or e.msg.startswith("Expecting ',' delimiter"):
                    fixed_for_incomplete_general = re.sub(r',\s*"[^"]*"\s*:\s*$', '', extracted_content[:e.pos])
                    fixed_for_incomplete_general = re.sub(r',\s*$', '', fixed_for_incomplete_general)
                    fixed_for_incomplete_general += extracted_content[e.pos:]

                    last_json_match_inner_general = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', fixed_for_incomplete_general, re.DOTALL)
                    if last_json_match_inner_general:
                        if last_json_match_inner_general.group(1):
                            fixed_for_incomplete_general = last_json_match_inner_general.group(1)
                        elif last_json_match_inner_general.group(2):
                            fixed_for_incomplete_general = last_json_match_inner_general.group(2)

                parsed_json = json.loads(fixed_for_incomplete_general)
                if schema and jsonschema:
                    jsonschema.validate(instance=parsed_json, schema=schema)
                logger.warning(f"Partial JSON recovered (general embedded block fallback with structural recovery): {parsed_json}. Returning partial data.",
                               extra={"partial_json_recovered": parsed_json})
                return parsed_json
            except json.JSONDecodeError as sub_e:
                logger.warning(f"General embedded structural recovery failed: {sub_e}. Continuing. Snippet: '{fixed_for_incomplete_general[max(0, sub_e.pos-50):sub_e.pos+50]}'")
            except Exception as sub_exp:
                logger.warning(f"An unexpected error occurred during general embedded structural recovery: {sub_exp}. Continuing. Snippet: '{fixed_for_incomplete_general[:100]}'")


    # --- Attempt 4: Robust single-quote replacement (applied to the *original* cleaned string if no code block) ---
    # This is the existing Strategy 2 from previous version.
    temp_json_string = _fix_single_quotes_in_json(cleaned_json_string)

    # Re-apply Hard JSON Block Cutter after single quote fix
    last_json_match_sq_fix = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', temp_json_string, re.DOTALL)
    if last_json_match_sq_fix:
        if last_json_match_sq_fix.group(1):
            temp_json_string = last_json_match_sq_fix.group(1)
        elif last_json_match_sq_fix.group(2):
            temp_json_string = last_json_match_sq_fix.group(2)
        logger.debug(f"Hard JSON block cutter applied after single quote fix. Snippet: {temp_json_string[:200]}...")
    else:
        logger.debug("Hard JSON block cutter found no full JSON match after single quote fix, proceeding.")

    try:
        parsed_json = json.loads(temp_json_string)
        if schema and jsonschema:
            try:
                jsonschema.validate(instance=parsed_json, schema=schema)
                logger.debug("LLM JSON response (single quote fix) validated successfully against schema.")
                return parsed_json
            except jsonschema.exceptions.ValidationError as ve:
                logger.warning(f"LLM JSON response validation error after single quote fix: {ve.message}. Trying other methods.",
                               extra={"json_validation_error": ve.message, "invalid_json_string": temp_json_string[:200]})
        else:
            return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode JSON after single quote fix: {e}. Trying structural recovery. Snippet: '{temp_json_string[max(0, e.pos-50):e.pos+50]}...'")
        # --- Structural Recovery for single quote fix ---
        try:
            fixed_for_incomplete_final = temp_json_string
            if e.msg.startswith("Expecting value") or e.msg.startswith("Expecting ',' delimiter"):
                fixed_for_incomplete_final = re.sub(r',\s*"[^"]*"\s*:\s*$', '', temp_json_string[:e.pos])
                fixed_for_incomplete_final = re.sub(r',\s*$', '', fixed_for_incomplete_final)
                fixed_for_incomplete_final += temp_json_string[e.pos:]

                last_json_match_inner_final = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', fixed_for_incomplete_final, re.DOTALL)
                if last_json_match_inner_final:
                    if last_json_match_inner_final.group(1):
                        fixed_for_incomplete_final = last_json_match_inner_final.group(1)
                    elif last_json_match_inner_final.group(2):
                        fixed_for_incomplete_final = last_json_match_inner_final.group(2)
            try:
                parsed_json = json.loads(fixed_for_incomplete_final)
                if schema and jsonschema:
                    jsonschema.validate(instance=parsed_json, schema=schema)
                logger.warning(f"Partial JSON recovered (single quote fix fallback with structural recovery): {parsed_json}. Returning partial data.",
                               extra={"partial_json_recovered": parsed_json})
                return parsed_json
            except json.JSONDecodeError as sub_e:
                logger.warning(f"Single quote fix structural recovery failed: {sub_e}. Continuing. Snippet: '{fixed_for_incomplete_final[max(0, sub_e.pos-50):sub_e.pos+50]}'")
            except Exception as sub_exp:
                logger.warning(f"An unexpected error occurred during single quote fix structural recovery: {sub_exp}. Continuing. Snippet: '{fixed_for_incomplete_final[:100]}'")
        except Exception as e:
            logger.warning(f"An unexpected error occurred during JSON parsing after single quote fix: {e}. Trying other methods. Snippet: '{temp_json_string[:100]}...'")


    # --- Last resort: Try brace matching and manual extraction (least robust) ---
    # This is the existing Strategy 3 from previous version.
    # It should only be reached if all other, more precise methods failed.
    first_open_idx = -1
    for char_idx, char in enumerate(cleaned_json_string):
        if char == '{' or char == '[':
            first_open_idx = char_idx
            break

    if first_open_idx != -1:
        balance = 0
        in_string = False
        escape_next = False
        json_end_idx = -1
        
        opening_char = cleaned_json_string[first_open_idx]
        closing_char = '}' if opening_char == '{' else ']'

        for char_idx in range(first_open_idx, len(cleaned_json_string)):
            char = cleaned_json_string[char_idx]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if in_string:
                if char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == opening_char:
                    balance += 1
                elif char == closing_char:
                    balance -= 1
            
            if balance == 0 and not in_string:
                json_end_idx = char_idx + 1
                break
        
        if json_end_idx != -1:
            json_to_parse = cleaned_json_string[first_open_idx:json_end_idx].strip()
            # Apply the same pre-processing steps as above, in case they were missed or re-introduced
            json_to_parse = re.sub(r'//.*$', '', json_to_parse, flags=re.MULTILINE)
            json_to_parse = re.sub(r',\s*([}\]])', r'\1', json_to_parse)
            json_to_parse = _fix_single_quotes_in_json(json_to_parse) # Apply single quote fix
            json_to_parse = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_to_parse)


            try:
                # No need for unicodedata.normalize here if done at the start
                # No need for .replace here if done at the start
                parsed_json = json.loads(json_to_parse)
                if schema and jsonschema:
                    try:
                        jsonschema.validate(instance=parsed_json, schema=schema)
                        logger.debug("LLM JSON response (brace/bracket matching) validated successfully against schema.")
                        return parsed_json
                    except jsonschema.exceptions.ValidationError as ve:
                        logger.warning(f"LLM JSON response validation error (brace/bracket matching): {ve.message}. Trying next method.",
                                       extra={"json_validation_error": ve.message, "invalid_json_string": json_to_parse[:200]})
                else:
                    return parsed_json
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to decode JSON from brace/bracket matching: {e}. Snippet: '{json_to_parse[max(0, e.pos-50):e.pos+50]}...'")
                # --- Structural Recovery for brace matching ---
                try:
                    fixed_for_incomplete_brace = json_to_parse
                    if e.msg.startswith("Expecting value") or e.msg.startswith("Expecting ',' delimiter"):
                        fixed_for_incomplete_brace = re.sub(r',\s*"[^"]*"\s*:\s*$', '', json_to_parse[:e.pos])
                        fixed_for_incomplete_brace = re.sub(r',\s*$', '', fixed_for_incomplete_brace)
                        fixed_for_incomplete_brace += json_to_parse[e.pos:]

                        last_json_match_inner_brace = re.search(r'(\{.*?[\}\]])(?!.*\{)|(\[.*?\])(?!.*\[)', fixed_for_incomplete_brace, re.DOTALL)
                        if last_json_match_inner_brace:
                            if last_json_match_inner_brace.group(1):
                                fixed_for_incomplete_brace = last_json_match_inner_brace.group(1)
                            elif last_json_match_inner_brace.group(2):
                                fixed_for_incomplete_brace = last_json_match_inner_brace.group(2)

                    parsed_json = json.loads(fixed_for_incomplete_brace)
                    if schema and jsonschema:
                        jsonschema.validate(instance=parsed_json, schema=schema)
                    logger.warning(f"Partial JSON recovered (brace matching fallback with structural recovery): {parsed_json}. Returning partial data.",
                                   extra={"partial_json_recovered": parsed_json})
                    return parsed_json
                except json.JSONDecodeError as sub_e:
                    logger.warning(f"Brace matching structural recovery failed: {sub_e}. Continuing. Snippet: '{fixed_for_incomplete_brace[max(0, sub_e.pos-50):sub_e.pos+50]}'")
                except Exception as sub_exp:
                    logger.warning(f"An unexpected error occurred during brace matching structural recovery: {sub_exp}. Continuing. Snippet: '{fixed_for_incomplete_brace[:100]}'")
            except Exception as e:
                logger.warning(f"An unexpected error occurred during JSON parsing (brace/bracket matching): {e}. Snippet: '{json_to_parse[:100]}...'")
    
    logger.error(f"No valid JSON found or parsed from LLM response after trying all strategies. Raw response: '{original_json_string[:500]}...'",
                 extra={"raw_json_snippet": original_json_string[:500]})
    # Fallback: extract only valid key-value pairs
    fallback_kv = extract_kv_pairs(json_string)
    if fallback_kv:
        logger.warning(f"Fallback: Extracted only valid key-value pairs from malformed JSON: {fallback_kv}")
        return fallback_kv
    return {} # Return empty dict if no valid JSON is found after all attempts


def flatten_metadata(metadata: Dict, parent_key: str = '') -> Dict:
    """
    Recursively flattens a nested dictionary into a single-level dictionary
    where all values are strings, ints, floats, or booleans.
    Keys are concatenated with underscores.
    Lists are joined into comma-separated strings.
    Converts None values to empty strings to avoid ChromaDB TypeErrors.
    """
    flattened = {}
    for key, value in metadata.items():
        new_key = f"{parent_key}_{key}" if parent_key else key
        # Sanitize key to be alphanumeric and underscores only
        clean_new_key = re.sub(r'[^a-zA-Z0-9_]', '', new_key.lower().replace(' ', '_'))
        if not clean_new_key: # Skip if key becomes empty after sanitization
            continue

        if isinstance(value, dict):
            flattened.update(flatten_metadata(value, clean_new_key))
        elif isinstance(value, list):
            # Join list items into a single string, converting None to empty string
            flattened[clean_new_key] = ", ".join(str(item) if item is not None else "" for item in value)
        elif value is None:
            # Convert None to empty string
            flattened[clean_new_key] = ""
        elif isinstance(value, (str, int, float, bool)):
            flattened[clean_new_key] = value
        else:
            # For any other complex type, convert to string
            flattened[clean_new_key] = str(value)
            logger.warning(f"Flattening metadata: Converted unsupported type {type(value)} to string for key '{clean_new_key}'.")
    return flattened

# Define generalized combination rules for metadata post-processing
# Order matters: more specific combinations first
COMBINATION_RULES = [
    # Product Name Combinations - Prioritize longer, more specific names
    {
        "target_key": "product_name",
        "source_keys": ["blood_bank_refrigerator", "bank_refrigerator"],
        "separator": " ",
        "condition": lambda m: "blood_bank_refrigerator" in m, # Prioritize full name
        "priority": 10 # Higher priority
    },
    {
        "target_key": "product_name",
        "source_keys": ["bldc", "pedestal_fan", "model"],
        "separator": " ",
        "condition": lambda m: "bldc" in m and "pedestal_fan" in m and "model" in m,
        "priority": 9
    },
    {
        "target_key": "product_name",
        "source_keys": ["bldc", "pedestal_fan"],
        "separator": " ",
        "condition": lambda m: "bldc" in m and "pedestal_fan" in m,
        "priority": 8
    },
    {
        "target_key": "product_name",
        "source_keys": ["bldc", "table_fan", "model"],
        "separator": " ",
        "condition": lambda m: "bldc" in m and "table_fan" in m and "model" in m,
        "priority": 9
    },
    {
        "target_key": "product_name",
        "source_keys": ["bldc", "table_fan"],
        "separator": " ",
        "condition": lambda m: "bldc" in m and "table_fan" in m,
        "priority": 8
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "street_lights_venus_series"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "street_lights_venus_series" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "hand_lamp"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "hand_lamp" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "high_bay_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "high_bay_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "emergency_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "emergency_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "garden_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "garden_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "inground_burial"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "inground_burial" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "surface_mounted_down_lights"], # Note: 'down_lights' due to sanitization
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "surface_mounted_down_lights" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "solar_light_all_in_one"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "solar_light_all_in_one" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "street_light_earth_series"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "street_light_earth_series" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "spot_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "spot_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "canopy_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "canopy_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "traffic_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "traffic_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "aviation_lights"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "aviation_lights" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "bulk_head_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "bulk_head_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "flamproof_well_glass"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "flamproof_well_glass" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "non_flameproof_well_glass"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "non_flameproof_well_glass" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "helmet_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "helmet_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "triproof_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "triproof_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "flamproof_tube_light"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "flamproof_tube_light" in m,
        "priority": 7
    },
    {
        "target_key": "product_name",
        "source_keys": ["led", "decorative_post_of_lantern"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "decorative_post_of_lantern" in m,
        "priority": 7
    },
    # General "LED Product Name" combination (more generic, put after specific ones)
    {
        "target_key": "product_name",
        "source_keys": ["led", "product_type"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "product_type" in m,
        "priority": 6
    },
    {
        "target_key": "product_name",
        "source_keys": ["model", "product_type"],
        "separator": " ",
        "condition": lambda m: "model" in m and "product_type" in m and len(str(m["model"])) > len(str(m["product_type"])),
        "priority": 5 # Prefer model if more descriptive
    },
    {
        "target_key": "product_name",
        "source_keys": ["model"],
        "separator": "",
        "condition": lambda m: "model" in m and "product_name" not in m, # Use model as product_name if no other product_name
        "priority": 1
    },
    {
        "target_key": "product_name",
        "source_keys": ["product"],
        "separator": "",
        "condition": lambda m: "product" in m and "product_name" not in m,
        "priority": 1
    },
    
    # Technical Specification Combinations
    {
        "target_key": "power_factor",
        "source_keys": ["power", "factor"],
        "separator": " ",
        "condition": lambda m: str(m.get("power", "")).lower() == "power" and "factor" in m,
        "priority": 3
    },
    {
        "target_key": "input_voltage",
        "source_keys": ["input", "voltage"],
        "separator": " ",
        "condition": lambda m: str(m.get("input", "")).lower() == "input" and "voltage" in m,
        "priority": 3
    },
    {
        "target_key": "color_rendering_index",
        "source_keys": ["color", "rendering_index"],
        "separator": " ",
        "condition": lambda m: str(m.get("color", "")).lower() == "color" and "rendering_index" in m,
        "priority": 3
    },
    {
        "target_key": "lamp_life_hours",
        "source_keys": ["lamp", "life"],
        "separator": " ",
        "condition": lambda m: str(m.get("lamp", "")).lower() == "lamp" and "life" in m,
        "priority": 3
    },
    {
        "target_key": "protection_grade",
        "source_keys": ["protection", "grade"],
        "separator": " ",
        "condition": lambda m: str(m.get("protection", "")).lower() == "protection" and "grade" in m,
        "priority": 3
    },
    {
        "target_key": "operating_temperature",
        "source_keys": ["operating", "temperature"],
        "separator": " ",
        "condition": lambda m: str(m.get("operating", "")).lower() == "operating" and "temperature" in m,
        "priority": 3
    },
    {
        "target_key": "working_humidity",
        "source_keys": ["working", "humidity"],
        "separator": " ",
        "condition": lambda m: str(m.get("working", "")).lower() == "working" and "humidity" in m,
        "priority": 3
    },
    {
        "target_key": "luminous_flux",
        "source_keys": ["led", "luminous_flux_lumen"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "luminous_flux_lumen" in m,
        "priority": 3
    },
    {
        "target_key": "luminous_efficacy",
        "source_keys": ["led", "luminous_efficacy_lm_w"],
        "separator": " ",
        "condition": lambda m: str(m.get("led", "")).lower() == "led" and "luminous_efficacy_lm_w" in m,
        "priority": 3
    },
    {
        "target_key": "motor_type",
        "source_keys": ["motor", "type"],
        "separator": " ",
        "condition": lambda m: str(m.get("motor", "")).lower() == "motor" and "type" in m,
        "priority": 3
    },
    {
        "target_key": "no_of_blades",
        "source_keys": ["no", "of_blades"],
        "separator": " ",
        "condition": lambda m: str(m.get("no", "")).lower() == "no" and "of_blades" in m,
        "priority": 3
    },
    {
        "target_key": "cct_k",
        "source_keys": ["cct", "k"], # For cases like "CCT (K)"
        "separator": "",
        "condition": lambda m: "cct" in m and "k" in m and not str(m["cct"]).isalpha(), # Only if 'k' is a unit, not a word
        "priority": 3
    },
    {
        "target_key": "cri",
        "source_keys": ["cri", "value"], # If CRI is split from its value
        "separator": ": ",
        "condition": lambda m: str(m.get("cri", "")).lower() == "cri" and "value" in m,
        "priority": 3
    },
    {
        "target_key": "beam_angle",
        "source_keys": ["beam", "angle"],
        "separator": " ",
        "condition": lambda m: str(m.get("beam", "")).lower() == "beam" and "angle" in m,
        "priority": 3
    },
    {
        "target_key": "dimensions_mm",
        "source_keys": ["dimension", "mm"],
        "separator": "",
        "condition": lambda m: "dimension" in m and "mm" in m,
        "priority": 3
    },
    # General "Technical Highlights" combination
    {
        "target_key": "technical_highlights",
        "source_keys": ["technical", "highlights"],
        "separator": " ",
        "condition": lambda m: str(m.get("technical", "")).lower() == "technical" and "highlights" in m,
        "priority": 2
    },
    # General "Green and Environment Protection"
    {
        "target_key": "environmental_protection",
        "source_keys": ["green", "and_environment_protection"],
        "separator": " ",
        "condition": lambda m: str(m.get("green", "")).lower() == "green" and "and_environment_protection" in m,
        "priority": 2
    },
    # Contact Information
    {
        "target_key": "contact_phone",
        "source_keys": ["phone", "number"],
        "separator": ": ",
        "condition": lambda m: "phone" in m and "number" in m,
        "priority": 2
    },
    {
        "target_key": "contact_email",
        "source_keys": ["email", "address"],
        "separator": ": ",
        "condition": lambda m: "email" in m and "address" in m,
        "priority": 2
    },
    {
        "target_key": "full_address",
        "source_keys": ["madri", "udaipur", "rajasthan", "india"],
        "separator": ", ",
        "condition": lambda m: all(k in m for k in ["madri", "udaipur", "rajasthan", "india"]),
        "priority": 2
    }
]

def _combine_metadata_fields(metadata: Dict) -> Dict:
    """
    Applies a set of predefined rules to combine fragmented metadata fields
    into more meaningful, higher-level concepts.
    This function now performs multiple passes, prioritizing rules with higher 'priority'.
    """
    processed_metadata = metadata.copy()
    
    # Sort rules by priority (higher priority first)
    sorted_rules = sorted(COMBINATION_RULES, key=lambda x: x.get("priority", 0), reverse=True)

    # Apply rules iteratively until no more changes are made
    changed = True
    while changed:
        changed = False
        for rule in sorted_rules:
            source_keys_exist = all(key in processed_metadata for key in rule["source_keys"])
            
            condition_met = True
            if "condition" in rule:
                try:
                    condition_met = rule["condition"](processed_metadata)
                except KeyError:
                    condition_met = False # If condition relies on a key that's not there

            if source_keys_exist and condition_met:
                values_to_combine = [str(processed_metadata[key]) for key in rule["source_keys"]]
                
                combined_value = rule["separator"].join(values_to_combine).strip()
                
                # Special handling for "LED LED" or similar duplications
                if rule["target_key"] == "product_name":
                    if combined_value.lower().startswith("led led"):
                        combined_value = combined_value[4:].strip() # Remove redundant "LED "
                    elif combined_value.lower().startswith("bldc bldc"):
                        combined_value = combined_value[5:].strip() # Remove redundant "BLDC "
                    
                    # More sophisticated product name merging
                    # If target_key is product_name and a product_name already exists,
                    # prefer the longer/more specific one.
                    existing_product_name = processed_metadata.get("product_name", "")
                    if existing_product_name and rule["target_key"] == "product_name":
                        if len(combined_value) > len(existing_product_name):
                            processed_metadata[rule["target_key"]] = combined_value
                            changed = True
                        else:
                            # If existing is longer or equally long and seems more specific, keep it
                            # e.g., "BLDC Table Fan" vs "Table Fan"
                            if "bldc" in existing_product_name.lower() and "bldc" not in combined_value.lower():
                                pass # Keep existing
                            elif "led" in existing_product_name.lower() and "led" not in combined_value.lower():
                                pass # Keep existing
                            else:
                                processed_metadata[rule["target_key"]] = combined_value
                                changed = True
                    else:
                        processed_metadata[rule["target_key"]] = combined_value
                        changed = True
                
                logger.debug(f"Combined '{rule['source_keys']}' into '{rule['target_key']}': {processed_metadata[rule['target_key']]}")
                
                # Remove original source keys if they were successfully combined
                for key_to_delete in rule["source_keys"]:
                    if key_to_delete in processed_metadata:
                        del processed_metadata[key_to_delete]
    
    return processed_metadata

# Add this function somewhere in your helper functions section, e.g., after flatten_metadata
def _validate_and_correct_extracted_metadata(extracted_data: Dict) -> Dict:
    """
    Performs sanity checks on extracted metadata to correct common LLM misinterpretations.
    """
    corrected_data = extracted_data.copy()

    # Define common misinterpretations and their likely correct keys
    # Key: (pattern_to_match_in_value, correct_key_if_match_found, original_incorrect_key)
    misinterpretations = [
        (r'%-?\d+%\s*RH', 'working_humidity_rh', 'power_factor'), # e.g., "5%-90% RH"
        (r'\d+K', 'cct_k', 'surge_protection'), # e.g., "5700K - 6500K"
        (r'IP\d+', 'ip_rating', 'beam_angle'), # e.g., "IP65"
        (r'(?i)aluminium|aluminum|steel|plastic|die-cast', 'enclosure_material', 'beam_angle'), # e.g., "Die-Cast Aluminium"
        (r'(?i)watts?|w', 'total_power_consumption_w', 'some_other_power_key'), # e.g., "100W"
    ]

    for incorrect_key, correct_key, original_source_key_hint in misinterpretations:
        if incorrect_key in corrected_data:
            value = str(corrected_data[incorrect_key])
            for pattern, target_key, _ in misinterpretations: # Iterate through patterns to find a fit
                if re.search(pattern, value, re.IGNORECASE):
                    if target_key != incorrect_key: # Only move if it's a different, more appropriate key
                        # Check if the value is already correctly assigned to the target_key
                        if target_key not in corrected_data or \
                           (target_key in corrected_data and value != corrected_data[target_key]):
                            logger.warning(f"Metadata correction: Moving value '{value}' from '{incorrect_key}' to '{target_key}'.")
                            corrected_data[target_key] = value
                        del corrected_data[incorrect_key] # Remove the incorrect entry
                        break # Move to next incorrect_key after a correction
    
    # Specific check for 'beam_angle' being 'Die' or similar fragments
    if 'beam_angle' in corrected_data and isinstance(corrected_data['beam_angle'], str):
        if corrected_data['beam_angle'].lower() in ['die', 'cast', 'aluminium', 'aluminum']:
            logger.warning(f"Metadata correction: Removing incorrect 'beam_angle' value '{corrected_data['beam_angle']}'.")
            del corrected_data['beam_angle']
    
    # If 'die_material' exists but 'enclosure_material' doesn't, try to merge
    if 'die_material' in corrected_data and 'enclosure_material' not in corrected_data:
        logger.warning(f"Metadata correction: Moving 'die_material' to 'enclosure_material'.")
        corrected_data['enclosure_material'] = corrected_data['die_material']
        del corrected_data['die_material']

    return corrected_data


# Function to extract product metadata using LLM
def extract_product_metadata(text_content: str, filename: str, elements: List[Any]) -> Dict:
    """
    Extracts product-related metadata from text content using an LLM.
    Prioritizes structured data (tables, lists) and uses LLM for general text.
    Applies post-processing to combine fragmented fields.
    Robust to all LLM response types and noisy outputs.
    """
    metadata = {"source_file_name": filename, "processing_date": datetime.now().isoformat()}
    
    # --- Document Type Detection (Generalized) ---
    # This section dynamically assigns document types and section types based on content and filename.
    filename_lower = filename.lower()
    text_content_lower = text_content.lower()

    # HR Policy documents
    hr_keywords = ["hr policy", "human resources", "leave policy", "safety policy", "code of conduct", "employee handbook"]
    if any(keyword in filename_lower for keyword in hr_keywords) or \
       any(keyword in text_content_lower for keyword in hr_keywords):
        metadata["document_type"] = "HR Policy"
        # Further classify HR section_type
        if "leave" in text_content_lower or "holiday" in text_content_lower:
            metadata["section_type"] = "hr_leaves_holidays"
            metadata["policy_type"] = "leave"
            metadata["tags"] = ", ".join(["pto", "vacation", "holiday"])
        elif "safety" in text_content_lower or "health" in text_content_lower:
            metadata["section_type"] = "hr_health_safety"
            metadata["policy_type"] = "safety"
            metadata["tags"] = ", ".join(["workplace safety", "health", "hazard"])
        elif "working hours" in text_content_lower:
            metadata["section_type"] = "hr_working_hours"
            metadata["policy_type"] = "working_hours"
            metadata["tags"] = ", ".join(["timings", "schedule"])
        elif "benefits" in text_content_lower:
            metadata["section_type"] = "hr_employee_benefits"
            metadata["policy_type"] = "benefits"
            metadata["tags"] = ", ".join(["compensation", "perks"])
        elif "discipline" in text_content_lower or "code of conduct" in text_content_lower:
            metadata["section_type"] = "hr_discipline_policy"
            metadata["policy_type"] = "conduct"
            metadata["tags"] = ", ".join(["ethics", "behavior", "rules"])
        else:
            metadata["section_type"] = "hr_general"
            metadata["policy_type"] = "general_hr"
            metadata["tags"] = ", ".join(["general"])
        return metadata

    # Product Datasheet/Catalogue documents (Generalized detection)
    product_keywords = ["led", "fan", "light", "series", "product", "technical", "specification", "datasheet", "catalogue"]
    if any(keyword in filename_lower for keyword in product_keywords) or \
       any(keyword in text_content_lower for keyword in product_keywords):
        metadata["document_type"] = "Product Datasheet"
        # More specific product type from filename if possible
        if "street light" in filename_lower:
            metadata["product_type"] = "LED Street Light"
        elif "ceiling fan" in filename_lower:
            metadata["product_type"] = "BLDC Ceiling Fan"
        elif "emergency light" in filename_lower:
            metadata["product_type"] = "LED Emergency Light"
        # Add more specific product type detections as needed
    
    # Default to "General Document" if not classified above
    if "document_type" not in metadata:
        metadata["document_type"] = "General Document"


    # Prioritize extracting from structured elements if available
    structured_text = ""
    for el in elements:
        if isinstance(el, (Table, ListItem)):
            structured_text += str(el) + "\n"
        elif isinstance(el, (NarrativeText, Text, Title)):
            # If it's a short text element that looks like a key-value pair, add it
            if len(str(el).split()) < 10 and (":" in str(el) or " - " in str(el)):
                structured_text += str(el) + "\n"
    
    if structured_text.strip():
        llm_prompt_messages = [
            {"role": "system", "content": LLM_PROMPT_TEMPLATES["table_metadata_extraction_system"]},
            {"role": "user", "content": structured_text}
        ]
        llm_response = ask_llm_model_phi3(llm_prompt_messages)
        extracted_data = safe_parse_llm_json_response(llm_response)
        
        # --- FIX START (around line 1747) ---
        # Ensure extracted_data is a dictionary. If LLM returned a list with one dict, unwrap it.
        if isinstance(extracted_data, list) and len(extracted_data) == 1 and isinstance(extracted_data[0], dict):
            extracted_data = extracted_data[0]
            logger.info("Unwrapped single dictionary from LLM list response for structured text metadata extraction.")
        
        if isinstance(extracted_data, dict): # Ensure extracted_data is a dictionary
            if extracted_data:
                from rag_llm_json_utils import filter_noisy_metadata
                extracted_data = filter_noisy_metadata(extracted_data)
                logger.debug(f"Filtered metadata from structured text: {extracted_data}")
                # Map 'technical_highlights' to 'technical_specifications'
                if extracted_data.get("section_type") == "technical_highlights":
                    extracted_data["section_type"] = "technical_specifications"
                metadata.update(extracted_data)
                # Apply generalized combination rules
                metadata = _combine_metadata_fields(metadata)
                metadata = _validate_and_correct_extracted_metadata(metadata)
                # No return here, allow general text extraction to potentially add more
        else:
            logger.warning(f"LLM returned a non-dictionary type ({type(extracted_data)}) for structured text metadata extraction. Skipping update. Raw: {llm_response[:100]}...")
        # --- FIX END ---

    # Fallback to general text extraction if no structured data yielded results
    if text_content.strip():
        llm_prompt_messages = [
            {"role": "system", "content": LLM_PROMPT_TEMPLATES["table_metadata_extraction_system"]},
            {"role": "user", "content": text_content[:MAX_CHUNK_CHARS * 2]} # Send a larger snippet for general extraction
        ]
        llm_response = ask_llm_model_phi3(llm_prompt_messages)
        extracted_data = safe_parse_llm_json_response(llm_response)
        
        # --- FIX START (around line 1777) ---
        # Ensure extracted_data is a dictionary. If LLM returned a list with one dict, unwrap it.
        if isinstance(extracted_data, list) and len(extracted_data) == 1 and isinstance(extracted_data[0], dict):
            extracted_data = extracted_data[0]
            logger.info("Unwrapped single dictionary from LLM list response for general text metadata extraction.")
        
        if isinstance(extracted_data, dict): # Ensure extracted_data is a dictionary
            if extracted_data:
                logger.debug(f"Extracted metadata from general text: {extracted_data}")
                # Map 'technical_highlights' to 'technical_specifications'
                if extracted_data.get("section_type") == "technical_highlights":
                    extracted_data["section_type"] = "technical_specifications"
                metadata.update(extracted_data)
                # Apply generalized combination rules
                metadata = _combine_metadata_fields(metadata)
                metadata = _validate_and_correct_extracted_metadata(metadata)
        else:
            logger.warning(f"LLM returned an unexpected non-dictionary type ({type(extracted_data)}) for general text metadata extraction. Skipping update. Raw: {llm_response[:100]}...")
        # --- FIX END ---

    # Default section type if not already set or refined (Generalized section type assignment)
    if "section_type" not in metadata:
        # Chunk Diversification Strategy - Map "technical_highlights" to "technical_specifications"
        if "technical highlights" in text_content_lower or "technical specification" in text_content_lower or "product specification" in text_content_lower:
            metadata["section_type"] = "technical_specifications"
        elif "features" in text_content_lower:
            metadata["section_type"] = "features"
        elif "application" in text_content_lower:
            metadata["section_type"] = "application"
        elif "dimensions" in text_content_lower:
            metadata["section_type"] = "dimensions"
        elif "installation" in text_content_lower:
            metadata["section_type"] = "installation"
        elif "maintenance" in text_content_lower:
            metadata["section_type"] = "maintenance"
        elif "working hours" in text_content_lower:
            metadata["section_type"] = "hr_working_hours"
        elif "leave" in text_content_lower or "holiday" in text_content_lower:
            metadata["section_type"] = "hr_leaves_holidays"
        elif "safety" in text_content_lower or "health" in text_content_lower:
            metadata["section_type"] = "hr_health_safety"
        elif "discipline" in text_content_lower:
            metadata["section_type"] = "hr_discipline_policy"
        elif "general rules" in text_content_lower:
            metadata["section_type"] = "hr_general_rules"
        elif "retirement" in text_content_lower:
            metadata["section_type"] = "hr_retirement"
        elif "termination" in text_content_lower:
            metadata["section_type"] = "hr_termination"
        elif "about us" in text_content_lower or "company background" in text_content_lower:
            metadata["section_type"] = "about_us"
        elif "quality" in text_content_lower:
            metadata["section_type"] = "our_quality"
        elif "product overview" in text_content_lower:
            metadata["section_type"] = "product_overview"
        elif "ordering code" in text_content_lower:
            metadata["section_type"] = "ordering"
        else:
            metadata["section_type"] = "general" # Fallback if no specific section identified

    # Add product_name based on filename if not already extracted (Generalized filename parsing)
    if "product_name" not in metadata:
        # Attempt to extract product name from filename if it's not a generic table file
        if not is_markdown_table_doc(filename):
            # This regex is more robust for filenames like "LED_Street_Lights_Venus_Series_1_308f88c7_markdown_tables.json"
            # It captures the main product name part before any numerical suffixes or UUIDs
            name_match = re.match(r'([a-zA-Z0-9_]+(?:_[a-zA-Z0-9_]+)*?)(?:_\d+)?(?:_[a-f0-9]{8,32})?(?:_markdown_tables)?\.json', filename, re.IGNORECASE)
            if name_match:
                product_name = name_match.group(1).replace('_', ' ').strip()
                # Filter out generic terms that shouldn't be product names
                if product_name and product_name.lower() not in ["markdown", "tables", "blog", "career", "contact", "us", "led", "bldc", "fan", "light", "series", "update", "driver", "catalogue", "emergency"]:
                    metadata["product_name"] = product_name
                    logger.debug(f"Inferred product_name '{product_name}' from filename '{filename}'.")

    return metadata

# IMPROVEMENT 1: Chunking Enhancements (Generalized chunking logic)
def chunk_document_text(
    text_content: str,
    filename: str,
    elements: List[Any], # Added elements parameter
    min_chars_per_chunk: int = MIN_CHUNK_CHARS,
    max_chars_per_chunk: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    metadata: Dict = None
) -> List[Dict]:
    """
    Splits a large text document into smaller, overlapping chunks,
    prioritizing semantic breaks and structured content.
    Each returned chunk is a dictionary containing 'content' and 'metadata'.
    """
    if not text_content or not isinstance(text_content, str):
        logger.warning("chunk_document_text received empty or invalid content. Returning empty list.")
        return []

    is_md_table_doc = is_markdown_table_doc(filename)
    
    current_min_chars = min_chars_per_chunk
    current_min_alphanum_ratio = MIN_ALPHANUM_RATIO

    if is_md_table_doc:
        # Loosen filters for markdown table documents specifically
        current_min_chars = 20
        current_min_alphanum_ratio = 0.2
        logger.info(f"Applying loosened filters for markdown table document '{filename}': min_chars={current_min_chars}, min_alphanum_ratio={current_min_alphanum_ratio}")
        if metadata is None:
            metadata = {}
        metadata["source_doc_type"] = "markdown_table"
    else:
        logger.info(f"Using default filters for document '{filename}': min_chars={current_min_chars}, min_alphanum_ratio={current_min_alphanum_ratio}")

    cleaned_text_content = text_content.strip()
    # Aggressive HTML tag removal
    cleaned_text_content = re.sub(r'<[^>]+>', '', cleaned_text_content) # Remove all HTML tags
    cleaned_text_content = re.sub(r'\s*\n\s*\n\s*', '\n\n', cleaned_text_content)
    cleaned_text_content = re.sub(r'(?i)(product category|product category)\s*', '', cleaned_text_content)
    cleaned_text_content = re.sub(r'\s*\(W\)\s*\+/\-5W\s*', ' (W)+/-5W ', cleaned_text_content)
    cleaned_text_content = re.sub(r'\"([^\"]+)\"\s*,\s*\"([^\"]+)\"', r'\1: \2', cleaned_text_content)
    cleaned_text_content = re.sub(r'^\s*[\-\*\•]\s*', '- ', cleaned_text_content, flags=re.MULTILINE)
    cleaned_text_content = re.sub(r'^\s*\d+\.\s*', 'Numbered List Item: ', cleaned_text_content, flags=re.MULTILINE)
    cleaned_text_content = re.sub(r'\s+', ' ', cleaned_text_content).strip()

    if not cleaned_text_content:
        logger.warning("chunk_document_text: Text content is empty after preprocessing. Returning empty list.")
        return []

    processed_chunks_with_metadata = []
    
    section_headers_pattern = r'(?i)\n(TECHNICAL\s+SPECIFICATIONS?|PRODUCT\s+SPECIFICATIONS?|FEATURES|APPLICATION|ELECTRICAL\s+CHARACTERISTICS|LAMP\s+CHARACTERISTICS|DIMENSIONS|INSTALLATION|MAINTENANCE|WORKING\s+HOURS|LEAVE/HOLIDAYS|STATUTORY\s+BENEFITS|EMPLOYEE\s+BENEFITS|HEALTH\s+&\s+SAFETY|DISCIPLINE\s+POLICY|GENERAL\s+RULES|RETIREMENT|TERMINATION|ABOUT\s+US|OUR\s+QUALITY|PRODUCT\s+OVERVIEW|TECHNICAL\s+HIGHLIGHTS)\n'
    markdown_table_pattern = r'(^\|.+\|$\n^\|[ :\-|=]+\|$\n(?:^\|.+\|$\n?)+)'
    split_pattern = r'(' + markdown_table_pattern + r'|' + section_headers_pattern + r'|\n{2,})'
    
    parts = re.split(split_pattern, cleaned_text_content, flags=re.DOTALL | re.MULTILINE)
    
    current_buffer = []
    current_buffer_len = 0
    current_section_type = "general"
    
    # Noise Filtering - Expanded patterns based on observed data (Generalized noise patterns)
    known_noise_patterns = [
        r"(?i)^more$",  r"(?i)^home$", r"(?i)^services$", r"(?i)^portfolio$", r"(?i)^team$",
        r"(?i)^blog$", r"(?i)^gallery$", r"(?i)^career$", r"(?i)^contact us$", r"(?i)subscribe to our newsletter.*",
        r"(?i)^certificats$", r"(?i)^achievements$", r"(?i)useful\s+links", r"(?i)our\s+infrastructure",
        r"(?i)our\s+exhibitions", r"(?i)company\s+background", r"(?i)about\s+us",
        r"(?i)industrial\s*\|\s*lighting\s*\|\s*medical\s*\|\s*products\s*\|\s*covid\s*\|\s*products\s*\|\s*our\s*\|\s*newsletter", # Specific observed noise
        r"(?i)plug\s*sensors\s*pir\s*occupancy\s*sensor\s*covid\s*products\s*hand\s*held\s*sanitizer.*", # Specific observed noise
        r"(?i)led\s*decorative\s*post\s*of\s*lantern\s*industrial\s*lighting.*", # Specific observed noise
        r"(?i)solar\s*street\s*light\s*solar\s*garden\s*light\s*solar\s*flood\s*light.*", # Another observed noise pattern
        r"(?i)led\s*street\s*light\s*led\s*flood\s*light\s*led\s*high\s*bay\s*light.*", # Another observed noise pattern
        r"(?i)product\s*category\s*.*", # General product category headers that are often just noise
        r"(?i)parameter\s*\|\s*value\s*\|\s*---\s*\|\s*---", # Common table header noise
        r"(?i)more\s*details\s*here", r"(?i)click\s*to\s*view", r"(?i)learn\s*more", # Call to action phrases
        r"(?i)all\s*rights\s*reserved", r"(?i)privacy\s*policy", r"(?i)terms\s*of\s*service" # Footer/legal text
    ]
    
    for i, part in enumerate(parts):
        if part is None or not part.strip():
            continue

        part_strip = part.strip()
        chunk_lower = part_strip.lower()

        is_markdown_table_match = re.match(markdown_table_pattern, part_strip, flags=re.DOTALL | re.MULTILINE)
        is_section_header_match = re.match(section_headers_pattern, '\n' + part_strip + '\n', flags=re.IGNORECASE)

        if is_markdown_table_match:
            if current_buffer:
                chunk_content = "\n\n".join(current_buffer).strip()
                alphanum_ratio = calculate_alphanum_ratio(chunk_content)
                
                is_noise = False
                matched_pattern = None
                for pattern in known_noise_patterns:
                    if re.search(pattern, chunk_content, re.IGNORECASE):
                        is_noise = True
                        matched_pattern = pattern
                        break

                # Sparse Embedding Handling: Preserve critical technical chunks
                if (not is_noise and len(chunk_content) >= current_min_chars and alphanum_ratio >= current_min_alphanum_ratio) or \
                   (contains_critical_phrase(chunk_content) and len(chunk_content) > 50): # NEW: Sparse Embedding Handling
                    if contains_critical_phrase(chunk_content) and alphanum_ratio < current_min_alphanum_ratio:
                        logger.debug(f"Keeping low-alphanum chunk with critical phrase: {chunk_content[:100]}...")
                    chunk_meta = (metadata or {}).copy()
                    if current_section_type == "technical_highlights": # Chunk Diversification Strategy
                        chunk_meta['section_type'] = "technical_specifications"
                    else:
                        chunk_meta['section_type'] = current_section_type
                    if is_md_table_doc:
                        chunk_meta["source_doc_type"] = "markdown_table"
                    
                    # Add chunk_id, chunk_length, chunk_hash
                    chunk_meta["chunk_id"] = str(uuid.uuid4())
                    chunk_meta["chunk_length"] = len(chunk_content)
                    chunk_meta["chunk_hash"] = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()

                    processed_chunks_with_metadata.append({'content': chunk_content, 'metadata': chunk_meta})
                else:
                    logger.debug(f"Rejected due to noise/length/alphanum (noise: {is_noise}, len: {len(chunk_content)}, ratio: {alphanum_ratio:.2f}): {chunk_content[:100]}...")

                current_buffer = []
                current_buffer_len = 0
            
            table_content = part_strip
            table_rows = table_content.strip().split('\n')
            
            if len(table_rows) >= 2 and table_rows[0].strip().startswith('|') and re.match(r'^\|[ :\-|=]+\|$', table_rows[1].strip()):
                header_row = table_rows[0].strip()
                data_rows = table_rows[2:]
                
                headers = [h.strip() for h in header_row.strip('|').split('|')]
                
                for row_idx, row in enumerate(data_rows):
                    row_strip = row.strip()
                    if not row_strip.startswith('|') or not row_strip.endswith('|'):
                        continue
                    
                    values = [v.strip() for v in row_strip.strip('|').split('|')]
                    
                    row_data = {}
                    for h_idx, header in enumerate(headers):
                        if h_idx < len(values):
                            sanitized_header = re.sub(r'[^a-zA-Z0-9_]', '', header.lower().replace(' ', '_'))
                            if sanitized_header:
                                row_data[sanitized_header] = values[h_idx]

                    row_chunk_content = row_strip
                    
                    if not row_chunk_content.strip() or calculate_alphanum_ratio(row_chunk_content) < 0.1:
                        logger.debug(f"Skipping garbage table row: {row_chunk_content[:50]}...")
                        continue

                    row_chunk_meta = (metadata or {}).copy()
                    row_chunk_meta['section_type'] = 'table_row'
                    row_chunk_meta["source_doc_type"] = "markdown_table"
                    row_chunk_meta.update(row_data)
                    
                    # Smarter Section Type Assignment for Tables:
                    # If table content strongly suggests technical specs, override 'table_row' or 'table'
                    table_lower = row_chunk_content.lower()
                    if any(phrase in table_lower for phrase in ["voltage", "wattage", "power consumption", "ip rating", "cct", "cri", "luminous efficacy", "dimensions"]):
                        row_chunk_meta['section_type'] = 'technical_specifications'
                    elif "features" in table_lower:
                        row_chunk_meta['section_type'] = 'features'
                    elif "application" in table_lower:
                        row_chunk_meta['section_type'] = 'application'
                    elif "ordering code" in table_lower:
                        row_chunk_meta['section_type'] = 'ordering'

                    # Add chunk_id, chunk_length, chunk_hash
                    row_chunk_meta["chunk_id"] = str(uuid.uuid4())
                    row_chunk_meta["chunk_length"] = len(row_chunk_content)
                    row_chunk_meta["chunk_hash"] = hashlib.md5(row_chunk_content.encode('utf-8')).hexdigest()

                    processed_chunks_with_metadata.append({'content': row_chunk_content, 'metadata': row_chunk_meta})
                    logger.debug(f"Added table row chunk (len: {len(row_chunk_content)}) from '{filename}' with section_type: '{row_chunk_meta['section_type']}'.")
            else:
                if not table_content.strip() or calculate_alphanum_ratio(table_content) < 0.1:
                    logger.debug(f"Skipping garbage table content: {table_content[:50]}...")
                else:
                    table_meta = (metadata or {}).copy()
                    table_meta['section_type'] = 'table'
                    if is_md_table_doc:
                        table_meta["source_doc_type"] = "markdown_table"
                    
                    # Smarter Section Type Assignment for Tables (for the whole table chunk)
                    table_lower = table_content.lower()
                    if any(phrase in table_lower for phrase in ["voltage", "wattage", "power consumption", "ip rating", "cct", "cri", "luminous efficacy", "dimensions"]):
                        table_meta['section_type'] = 'technical_specifications'
                    elif "features" in table_lower:
                        table_meta['section_type'] = 'features'
                    elif "application" in table_lower:
                        table_meta['section_type'] = 'application'
                    elif "ordering code" in table_lower:
                        table_meta['section_type'] = 'ordering'

                    # Add chunk_id, chunk_length, chunk_hash
                    table_meta["chunk_id"] = str(uuid.uuid4())
                    table_meta["chunk_length"] = len(table_content)
                    table_meta["chunk_hash"] = hashlib.md5(table_content.encode('utf-8')).hexdigest()

                    processed_chunks_with_metadata.append({'content': table_content, 'metadata': table_meta})
                    logger.debug(f"Added table chunk (len: {len(table_content)}) from '{filename}' with section_type: '{table_meta['section_type']}'.")
            
            current_section_type = 'general'

        elif is_section_header_match:
            if current_buffer:
                chunk_content = "\n\n".join(current_buffer).strip()
                alphanum_ratio = calculate_alphanum_ratio(chunk_content)
                
                is_noise = False
                matched_pattern = None
                for pattern in known_noise_patterns:
                    if re.search(pattern, chunk_content, re.IGNORECASE):
                        is_noise = True
                        matched_pattern = pattern
                        break
                
                # Sparse Embedding Handling: Preserve critical technical chunks
                if (not is_noise and len(chunk_content) >= current_min_chars and alphanum_ratio >= current_min_alphanum_ratio) or \
                   (contains_critical_phrase(chunk_content) and len(chunk_content) > 50): # NEW: Sparse Embedding Handling
                    if contains_critical_phrase(chunk_content) and alphanum_ratio < current_min_alphanum_ratio:
                        logger.debug(f"Keeping low-alphanum chunk with critical phrase: {chunk_content[:100]}...")
                    chunk_meta = (metadata or {}).copy()
                    if current_section_type == "technical_highlights": # Chunk Diversification Strategy
                        chunk_meta['section_type'] = "technical_specifications"
                    else:
                        chunk_meta['section_type'] = current_section_type
                    if is_md_table_doc:
                        chunk_meta["source_doc_type"] = "markdown_table"

                    # Add chunk_id, chunk_length, chunk_hash
                    chunk_meta["chunk_id"] = str(uuid.uuid4())
                    chunk_meta["chunk_length"] = len(chunk_content)
                    chunk_meta["chunk_hash"] = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()

                    processed_chunks_with_metadata.append({'content': chunk_content, 'metadata': chunk_meta})
                else:
                    logger.debug(f"Rejected due to noise/length/alphanum (noise: {is_noise}, len: {len(chunk_content)}, ratio: {alphanum_ratio:.2f}): {chunk_content[:100]}...")
                    
                    if processed_chunks_with_metadata and len(processed_chunks_with_metadata[-1]['content']) + len(chunk_content) + 2 <= max_chars_per_chunk:
                        processed_chunks_with_metadata[-1]['content'] += "\n\n" + chunk_content
                        if processed_chunks_with_metadata[-1]['metadata'].get('section_type') != chunk_meta.get('section_type'):
                            processed_chunks_with_metadata[-1]['metadata']['section_type'] = 'mixed'
                
                current_buffer = []
                current_buffer_len = 0

            header_text = is_section_header_match.group(1).strip().lower().replace(' ', '_')
            if 'technical_specifications' in header_text or 'product_specifications' in header_text or 'technical_highlights' in header_text: # Chunk Diversification Strategy
                current_section_type = 'technical_specifications'
            elif 'features' in header_text:
                current_section_type = 'features'
            elif 'application' in header_text:
                current_section_type = 'application'
            elif 'working_hours' in header_text:
                current_section_type = 'hr_working_hours'
            elif 'leave/holidays' in header_text:
                current_section_type = 'hr_leaves_holidays'
            elif 'statutory_benefits' in header_text:
                current_section_type = 'hr_statutory_benefits'
            elif 'employee_benefits' in header_text:
                current_section_type = 'hr_employee_benefits'
            elif 'health_&_safety' in header_text:
                current_section_type = 'hr_health_safety'
            elif 'discipline_policy' in header_text:
                current_section_type = 'hr_discipline_policy'
            elif 'general_rules' in header_text:
                current_section_type = 'hr_general_rules'
            elif 'retirement' in header_text:
                current_section_type = 'hr_retirement'
            elif 'termination' in header_text:
                current_section_type = 'hr_termination'
            elif 'about_us' in header_text:
                current_section_type = 'about_us'
            elif 'our_quality' in header_text:
                current_section_type = 'our_quality'
            elif 'product_overview' in header_text:
                current_section_type = 'product_overview'
            else:
                current_section_type = 'general'

            current_buffer.append(part_strip)
            current_buffer_len += len(part_strip) + 2

        else:
            if current_buffer_len + len(part_strip) + 2 > max_chars_per_chunk and current_buffer:
                chunk_content = "\n\n".join(current_buffer).strip()
                alphanum_ratio = calculate_alphanum_ratio(chunk_content)
                
                is_noise = False
                matched_pattern = None
                for pattern in known_noise_patterns:
                    if re.search(pattern, chunk_content, re.IGNORECASE):
                        is_noise = True
                        matched_pattern = pattern
                        break
                
                # Sparse Embedding Handling: Preserve critical technical chunks
                if (not is_noise and len(chunk_content) >= current_min_chars and alphanum_ratio >= current_min_alphanum_ratio) or \
                   (contains_critical_phrase(chunk_content) and len(chunk_content) > 50): # NEW: Sparse Embedding Handling
                    if contains_critical_phrase(chunk_content) and alphanum_ratio < current_min_alphanum_ratio:
                        logger.debug(f"Keeping low-alphanum chunk with critical phrase: {chunk_content[:100]}...")
                    chunk_meta = (metadata or {}).copy()
                    if is_md_table_doc:
                        if "application" in chunk_lower:
                            chunk_meta["section_type"] = "applications"
                        elif "ordering code" in chunk_lower:
                            chunk_meta["section_type"] = "ordering"
                        elif "features" in chunk_lower:
                            chunk_meta["section_type"] = "features"
                        elif "technical specification" in chunk_lower or "parameter" in chunk_lower or "technical highlight" in chunk_lower: # Chunk Diversification Strategy
                            chunk_meta["section_type"] = "technical_specifications"
                        else:
                            chunk_meta["section_type"] = current_section_type
                        chunk_meta["source_doc_type"] = "markdown_table"
                    else:
                        if current_section_type == "technical_highlights": # Chunk Diversification Strategy
                            chunk_meta['section_type'] = "technical_specifications"
                        else:
                            chunk_meta['section_type'] = current_section_type

                    # Add chunk_id, chunk_length, chunk_hash
                    chunk_meta["chunk_id"] = str(uuid.uuid4())
                    chunk_meta["chunk_length"] = len(chunk_content)
                    chunk_meta["chunk_hash"] = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()

                    processed_chunks_with_metadata.append({'content': chunk_content, 'metadata': chunk_meta})
                else:
                    logger.debug(f"Rejected due to noise/length/alphanum (noise: {is_noise}, len: {len(chunk_content)}, ratio: {alphanum_ratio:.2f}): {chunk_content[:100]}...")
                    
                    if processed_chunks_with_metadata and len(processed_chunks_with_metadata[-1]['content']) + len(chunk_content) + 2 <= max_chars_per_chunk:
                        processed_chunks_with_metadata[-1]['content'] += "\n\n" + chunk_content
                        if processed_chunks_with_metadata[-1]['metadata'].get('section_type') != chunk_meta.get('section_type'):
                            processed_chunks_with_metadata[-1]['metadata']['section_type'] = 'mixed'
                
                overlap_text = ""
                if processed_chunks_with_metadata and overlap_chars > 0:
                    last_chunk_content = processed_chunks_with_metadata[-1]['content']
                    overlap_candidate = last_chunk_content[-min(len(last_chunk_content), overlap_chars):].strip()
                    if nlp:
                        sentences_in_overlap = [s.text for s in nlp(overlap_candidate).sents]
                        overlap_text = " ".join(sentences_in_overlap[-3:]).strip() if sentences_in_overlap else ""
                    else:
                        lines = overlap_candidate.split('\n')
                        overlap_text = "\n".join(lines[-3:]).strip() if lines else ""
                    
                    if len(overlap_text) < overlap_chars / 2 and len(overlap_candidate) > overlap_chars / 2:
                        overlap_text = overlap_candidate

                current_buffer = []
                current_buffer_len = 0
                if overlap_text:
                    current_buffer.append(overlap_text)
                    current_buffer_len += len(overlap_text) + 2

            current_buffer.append(part_strip)
            current_buffer_len += len(part_strip) + 2

    if current_buffer:
        final_buffered_chunk = "\n\n".join(current_buffer).strip()
        alphanum_ratio = calculate_alphanum_ratio(final_buffered_chunk)
        
        is_noise = False
        matched_pattern = None
        for pattern in known_noise_patterns:
            if re.search(pattern, final_buffered_chunk, re.IGNORECASE):
                is_noise = True
                matched_pattern = pattern
                break
        
        # Sparse Embedding Handling: Preserve critical technical chunks
        if (not is_noise and len(final_buffered_chunk) >= current_min_chars and alphanum_ratio >= current_min_alphanum_ratio) or \
           (contains_critical_phrase(final_buffered_chunk) and len(final_buffered_chunk) > 50): # NEW: Sparse Embedding Handling
            if contains_critical_phrase(final_buffered_chunk) and alphanum_ratio < current_min_alphanum_ratio:
                logger.debug(f"Keeping low-alphanum chunk with critical phrase: {final_buffered_chunk[:100]}...")
            chunk_meta = (metadata or {}).copy()
            if is_md_table_doc:
                if "application" in final_buffered_chunk.lower():
                    chunk_meta["section_type"] = "applications"
                elif "ordering code" in final_buffered_chunk.lower():
                    chunk_meta["section_type"] = "ordering"
                elif "features" in final_buffered_chunk.lower():
                    chunk_meta["section_type"] = "features"
                elif "technical specification" in final_buffered_chunk.lower() or "parameter" in final_buffered_chunk.lower() or "technical highlight" in final_buffered_chunk.lower(): # Chunk Diversification Strategy
                    chunk_meta["section_type"] = "technical_specifications"
                else:
                    chunk_meta['section_type'] = current_section_type
                chunk_meta["source_doc_type"] = "markdown_table"
            else:
                if current_section_type == "technical_highlights": # Chunk Diversification Strategy
                    chunk_meta['section_type'] = "technical_specifications"
                else:
                    chunk_meta['section_type'] = current_section_type
            
            # Add chunk_id, chunk_length, chunk_hash
            chunk_meta["chunk_id"] = str(uuid.uuid4())
            chunk_meta["chunk_length"] = len(final_buffered_chunk)
            chunk_meta["chunk_hash"] = hashlib.md5(final_buffered_chunk.encode('utf-8')).hexdigest()

            processed_chunks_with_metadata.append({'content': final_buffered_chunk, 'metadata': chunk_meta})
        else:
            logger.debug(f"Rejected due to noise/length/alphanum (noise: {is_noise}, len: {len(final_buffered_chunk)}, ratio: {alphanum_ratio:.2f}): {final_buffered_chunk[:100]}...")
            
            if processed_chunks_with_metadata and len(processed_chunks_with_metadata[-1]['content']) + len(final_buffered_chunk) + 2 <= max_chars_per_chunk:
                processed_chunks_with_metadata[-1]['content'] += "\n\n" + final_buffered_chunk
                if processed_chunks_with_metadata[-1]['metadata'].get('section_type') != chunk_meta.get('section_type'):
                    processed_chunks_with_metadata[-1]['metadata']['section_type'] = 'mixed'

    final_chunks_output = []
    for chunk_dict in processed_chunks_with_metadata:
        chunk_content = chunk_dict['content']
        chunk_meta = chunk_dict['metadata']

        if len(chunk_content) < current_min_chars and final_chunks_output and \
           len(final_chunks_output[-1]['content']) + len(chunk_content) + 2 <= max_chars_per_chunk:
            final_chunks_output[-1]['content'] += "\n\n" + chunk_content
            # Update metadata of the combined chunk
            if final_chunks_output[-1]['metadata'].get('section_type') != chunk_meta.get('section_type'):
                final_chunks_output[-1]['metadata']['section_type'] = 'mixed'
            final_chunks_output[-1]['metadata']['chunk_length'] = len(final_chunks_output[-1]['content'])
            final_chunks_output[-1]['metadata']['chunk_hash'] = hashlib.md5(final_chunks_output[-1]['content'].encode('utf-8')).hexdigest()
        elif len(chunk_content) > max_chars_per_chunk:
            sub_chunks_from_large = []
            if nlp:
                doc = nlp(chunk_content)
                sentences = [sent.text for sent in doc.sents]
                sub_chunk_buffer = ""
                for sentence in sentences:
                    s_strip = sentence.strip()
                    if not s_strip: continue
                    if len(sub_chunk_buffer) + len(s_strip) + 1 > max_chars_per_chunk and sub_chunk_buffer:
                        sub_chunks_from_large.append(sub_chunk_buffer)
                        sub_chunk_buffer = s_strip
                    else:
                        sub_chunk_buffer += (" " if sub_chunk_buffer else "") + s_strip
                if sub_chunk_buffer: sub_chunks_from_large.append(sub_chunk_buffer)
            else:
                lines = chunk_content.split('\n')
                sub_chunk_buffer = ""
                for line in lines:
                    l_strip = line.strip()
                    if not l_strip: continue
                    if len(sub_chunk_buffer) + len(l_strip) + 1 > max_chars_per_chunk and sub_chunk_buffer:
                        sub_chunks_from_large.append(sub_chunk_buffer)
                        sub_chunk_buffer = l_strip
                    else:
                        sub_chunk_buffer += ("\n" if sub_chunk_buffer else "") + l_strip
                if sub_chunk_buffer: sub_chunks_from_large.append(sub_chunk_buffer)

            for sc in sub_chunks_from_large:
                alphanum_ratio_sc = calculate_alphanum_ratio(sc)
                
                is_noise_sc = False
                matched_pattern_sc = None
                for pattern in known_noise_patterns:
                    if re.search(pattern, sc, re.IGNORECASE):
                        is_noise_sc = True
                        matched_pattern_sc = pattern
                        break

                # Sparse Embedding Handling: Preserve critical technical chunks
                if (not is_noise_sc and len(sc) >= current_min_chars and alphanum_ratio_sc >= current_min_alphanum_ratio) or \
                   (contains_critical_phrase(sc) and len(sc) > 50): # NEW: Sparse Embedding Handling
                    if contains_critical_phrase(sc) and alphanum_ratio_sc < current_min_alphanum_ratio:
                        logger.debug(f"Keeping low-alphanum sub-chunk with critical phrase: {sc[:100]}...")
                    new_meta = chunk_meta.copy()
                    if new_meta.get('section_type') == "technical_highlights": # Chunk Diversification Strategy
                        new_meta['section_type'] = "technical_specifications"
                    else:
                        new_meta['section_type'] = new_meta.get('section_type', 'split_large_chunk')
                    if is_md_table_doc:
                        new_meta["source_doc_type"] = "markdown_table"
                    
                    # Add chunk_id, chunk_length, chunk_hash
                    new_meta["chunk_id"] = str(uuid.uuid4())
                    new_meta["chunk_length"] = len(sc)
                    new_meta["chunk_hash"] = hashlib.md5(sc.encode('utf-8')).hexdigest()

                    final_chunks_output.append({'content': sc, 'metadata': new_meta})
                else:
                    logger.debug(f"Rejected sub-chunk due to noise/length/alphanum (noise: {is_noise_sc}, len: {len(sc)}, ratio: {alphanum_ratio_sc:.2f}): {sc[:100]}...")
                    
                    if final_chunks_output and len(final_chunks_output[-1]['content']) + len(sc) + 2 <= max_chars_per_chunk:
                        final_chunks_output[-1]['content'] += "\n\n" + sc
                        # Update metadata of the combined chunk
                        if final_chunks_output[-1]['metadata'].get('section_type') != new_meta.get('section_type'):
                            final_chunks_output[-1]['metadata']['section_type'] = 'mixed'
                        final_chunks_output[-1]['metadata']['chunk_length'] = len(final_chunks_output[-1]['content'])
                        final_chunks_output[-1]['metadata']['chunk_hash'] = hashlib.md5(final_chunks_output[-1]['content'].encode('utf-8')).hexdigest()
        else:
            final_chunks_output.append(chunk_dict)

    final_chunks_output = [c for c in final_chunks_output if c['content'].strip()]

    logger.debug(f"Initial text length: {len(cleaned_text_content)} chars.")
    logger.debug(f"Generated {len(final_chunks_output)} chunks for document.")
    if final_chunks_output:
        for idx, chunk_dict in enumerate(final_chunks_output[:3]):
            logger.debug(f"Chunk {idx+1} (len {len(chunk_dict['content'])}, type: {chunk_dict['metadata'].get('section_type')}): {chunk_dict['content'][:200]}...")
        if len(final_chunks_output) > 3:
            logger.debug("...")
    else:
        logger.debug("No chunks were generated despite content.")

    return final_chunks_output

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3), before_sleep=before_sleep_log(logger, logging.WARNING),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def _generate_ollama_embeddings_single_batch(texts: List[str]) -> List[List[float]]:
    """Helper to generate embeddings for a single batch using Ollama."""
    if not texts:
        return []

    logger.info(f"Generating embeddings using Ollama for {len(texts)} texts.")
    embeddings = []
    for text in texts:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cached_embedding = get_embedding_from_cache(text_hash) # Call 1
        if cached_embedding:
            # Enforce 768-dim for cached embeddings as well (for nomic-embed-text)
            if len(cached_embedding) != 768: # Updated to 768
                logger.warning(f"Cached embedding for hash {text_hash} has dimension {len(cached_embedding)}, expected 768. Recalculating.")
                # Force recalculation if cached embedding has wrong dimension
                pass 
            else:
                embeddings.append(cached_embedding)
                logger.debug(f"Cache hit for embedding: {text_hash}")
                continue
        
        try:
            response = session.post(
                OLLAMA_EMBED_API_URL,
                json={"model": EMBEDDING_MODEL_NAME, "prompt": text},
                timeout=300
            )
            response.raise_for_status()
            
            response_json = response.json()
            embedding = response_json.get("embedding")
            
            if embedding == None:
                raise ValueError(f"Ollama embedding response missing 'embedding' field for text: {text[:50]}...")
            
            # Enforce 768-dim embedding output (for nomic-embed-text)
            if len(embedding) != 768: # Updated to 768
                raise ValueError(f"❌ Only 768-dim embeddings are allowed. Ollama model '{EMBEDDING_MODEL_NAME}' returned {len(embedding)}-dim embeddings. Fix your embedding source.")

            embeddings.append(embedding)
            store_embedding_in_cache(text_hash, embedding)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embeddings from Ollama for text '{text[:50]}...': {e}", extra={"text_snippet_log": text[:50], "error_detail": str(e)})
            raise
        except ValueError as e:
            logger.error(f"Invalid embedding response from Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during embedding generation for text '{text[:50]}...': {e}", extra={"error_detail": str(e)})
            raise
    return embeddings

def generate_ollama_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts, handling batching and caching."""
    if not SYSTEM_READY or not embedding_model_status:
        logger.error("RAG system is not ready or embedding model is not available. Skipping embedding generation.",
                     extra={"system_ready_status": SYSTEM_READY, "embedding_model_available": embedding_model_status})
        return []

    all_embeddings = []
    num_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    
    for i in range(num_batches):
        batch_texts = texts[i * EMBEDDING_BATCH_SIZE : (i + 1) * EMBEDDING_BATCH_SIZE]
        try:
            batch_embeddings = _generate_ollama_embeddings_single_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i+1}/{num_batches} for embeddings. Generated {len(batch_embeddings)} embeddings.")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {i+1}: {e}. Stopping embedding generation for this document.",
                         extra={"batch_number": i+1, "error_detail": str(e)})
            return []
    return all_embeddings


class EmbeddingFunctionWrapper:
    """Wrapper class to make generate_ollama_embeddings compatible with expected interface."""
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embeddings = generate_ollama_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return generate_ollama_embeddings(texts)

def fuse_adjacent_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Fuses adjacent chunks from the same document and section type into a single chunk.
    This helps in creating more coherent context for the LLM.
    """
    fused = []
    prev = None

    for curr in chunks:
        if prev and \
           curr['metadata'].get('section_type') == prev['metadata'].get('section_type') and \
           curr['metadata'].get('source_file_name') == prev['metadata'].get('source_file_name'):
            prev['content'] += "\n\n" + curr['content']
            prev['score'] = (prev['score'] + curr['score']) / 2
            # Update chunk_length and chunk_hash for the fused chunk
            prev['metadata']['chunk_length'] = len(prev['content'])
            prev['metadata']['chunk_hash'] = hashlib.md5(prev['content'].encode('utf-8')).hexdigest()
            logger.debug(f"Fused adjacent chunks from '{prev['metadata'].get('source_file_name')}' (section: {prev['metadata'].get('section_type')}). New length: {len(prev['content'])}.")
        else:
            if prev:
                fused.append(prev)
            prev = curr
    
    if prev:
        fused.append(prev)
    
    logger.info(f"Fused {len(chunks)} chunks into {len(fused)} chunks.")
    return fused

# Reranker Upgrade
def rerank_documents(query: str, documents: List[Dict], all_retrieved_embeddings: Optional[List[List[float]]] = None):
    """
    Reranks documents based on a cross-encoder model.
    Applies reranker cutoff and soft-filtering.
    Uses an on-disk cache for reranker scores.
    Includes boosting for technical sections and fusing adjacent chunks.
    Receives all_retrieved_embeddings (same order as documents) for advanced reranking logic.
    """
    global reranker_tokenizer, reranker_model, secondary_reranker_tokenizer, secondary_reranker_model
    if not SYSTEM_READY or reranker_model is None or reranker_tokenizer is None:
        logger.warning("Primary reranker model not ready. Skipping reranking.", extra={"system_ready_status": SYSTEM_READY})
        return documents

    if not documents:
        return []

    query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
    
    # --- Stage 1: Primary Reranking ---
    primary_reranked_docs_with_scores = []
    texts_to_primary_rerank = []
    original_indices_primary = []
    for doc in documents:
        doc.setdefault("score", 0.5)  # or 0.0 if you prefer

    # Reranking Efficiency - Pre-filter before reranking
    high_similarity_docs = [doc for doc in documents if doc.get("score", 0) > SIMILARITY_THRESHOLD * 1.5]
    docs_for_primary_reranking_pool = []
    if len(high_similarity_docs) >= 5:
        docs_for_primary_reranking_pool = high_similarity_docs[:10]
        logger.info(f"Pre-filtered {len(high_similarity_docs)} high-similarity docs, selecting top 10 for primary reranking pool.")
    else:
        docs_for_primary_reranking_pool = documents # Use all if not enough high similarity docs
        logger.info(f"Less than 5 high-similarity docs ({len(high_similarity_docs)}), using all {len(documents)} initial docs for primary reranking pool.")

    # If all_embeddings provided, attach to docs for downstream use (MMR, advanced rerank)
    if all_retrieved_embeddings is not None and len(all_retrieved_embeddings) == len(documents):
        for i, doc in enumerate(documents):
            doc["embedding"] = all_retrieved_embeddings[i]


    for i, doc in enumerate(docs_for_primary_reranking_pool): # Iterate over the pre-filtered pool
        doc_content = doc["content"]
        doc_hash = hashlib.md5(doc_content.encode('utf-8')).hexdigest()
        
        cached_score = get_rerank_score_from_cache(query_hash, doc_hash, RERANKER_MODEL_NAME)
        if cached_score is not None:
            primary_reranked_docs_with_scores.append({
                "content": doc_content,
                "metadata": doc["metadata"],
                "score": cached_score,
                "cached": True,
                "embedding": doc.get("embedding")
            })
            logger.debug(f"Primary Rerank cache hit for doc {i} (score: {cached_score:.4f})")
        else:
            texts_to_primary_rerank.append([query, doc_content])
            original_indices_primary.append(i) # Store index relative to docs_for_primary_reranking_pool
            logger.debug(f"Primary Rerank cache miss for doc {i}. Adding to batch.")

    if texts_to_primary_rerank:
        logger.info(f"Primary reranking {len(texts_to_primary_rerank)} documents for query '{query[:30]}...' using {RERANKER_MODEL_NAME}")
        try:
            inputs = reranker_tokenizer([ (q_doc[0], q_doc[1]) for q_doc in texts_to_primary_rerank ],
                                        padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                scores = reranker_model(**inputs).logits.squeeze(-1).numpy()
            
            ranking_scores = scores.tolist()

            for i, original_idx in enumerate(original_indices_primary):
                doc_content = docs_for_primary_reranking_pool[original_idx]["content"] # Get from pool
                doc_metadata = docs_for_primary_reranking_pool[original_idx]["metadata"] # Get from pool
                doc_hash = hashlib.md5(doc_content.encode('utf-8')).hexdigest()
                score = ranking_scores[i]

                # Apply boosting for technical sections *before* adding to primary_reranked_docs_with_scores
                if doc_metadata.get("section_type") == "technical_specifications":
                    score *= 1.3  # Boost technical chunks

                primary_reranked_docs_with_scores.append({
                    "content": doc_content,
                    "metadata": doc_metadata,
                    "score": score,
                    "cached": False,
                    "embedding": docs_for_primary_reranking_pool[original_idx].get("embedding")
                })
                store_rerank_score_in_cache(query_hash, doc_hash, RERANKER_MODEL_NAME, score)
        except Exception as e:
            logger.error(f"ERROR during primary document reranking: {e}", extra={"query_snippet_log": query[:50], "error_detail": str(e)})
            traceback.print_exc()
            return documents # Fallback to original documents if primary reranking fails

    primary_reranked_docs_with_scores.sort(key=lambda x: x["score"], reverse=True)

    # Filter primary reranked docs by threshold
    filtered_primary_reranked_docs = [doc for doc in primary_reranked_docs_with_scores if doc["score"] >= MIN_RERANK_SCORE_THRESHOLD]
    
    # Select top N for secondary reranking
    docs_for_secondary_reranking = filtered_primary_reranked_docs[:NUM_DOCS_FOR_SECOND_RERANK]
    logger.info(f"Selected {len(docs_for_secondary_reranking)} documents for secondary reranking.")

    # --- Stage 2: Secondary Reranking (if secondary model is available) ---
    if secondary_reranker_model and secondary_reranker_tokenizer and docs_for_secondary_reranking:
        secondary_reranked_docs_with_scores = []
        texts_to_secondary_rerank = []
        original_docs_secondary = [] # Keep original docs to map back

        for doc in docs_for_secondary_reranking:
            doc_content = doc["content"]
            doc_hash = hashlib.md5(doc_content.encode('utf-8')).hexdigest()
            
            cached_score = get_rerank_score_from_cache(query_hash, doc_hash, SECONDARY_RERANKER_MODEL_NAME)
            if cached_score is not None:
                secondary_reranked_docs_with_scores.append({
                    "content": doc_content,
                    "metadata": doc["metadata"],
                    "score": cached_score,
                    "cached": True
                })
                logger.debug(f"Secondary Rerank cache hit for doc (score: {cached_score:.4f})")
            else:
                texts_to_secondary_rerank.append([query, doc_content])
                original_docs_secondary.append(doc)
                logger.debug(f"Secondary Rerank cache miss. Adding to batch.")

        if texts_to_secondary_rerank:
            logger.info(f"Secondary reranking {len(texts_to_secondary_rerank)} documents using {SECONDARY_RERANKER_MODEL_NAME}")
            try:
                inputs = secondary_reranker_tokenizer([ (q_doc[0], q_doc[1]) for q_doc in texts_to_secondary_rerank ],
                                                    padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    scores = secondary_reranker_model(**inputs).logits.squeeze(-1).numpy()
                
                ranking_scores = scores.tolist()

                for i, doc_original in enumerate(original_docs_secondary):
                    doc_content = doc_original["content"]
                    doc_metadata = doc_original["metadata"]
                    doc_hash = hashlib.md5(doc_content.encode('utf-8')).hexdigest()
                    score = ranking_scores[i]

                    # Apply boosting for technical sections (again, if desired, or adjust weights)
                    if doc_metadata.get("section_type") == "technical_specifications":
                        score *= 1.3

                    secondary_reranked_docs_with_scores.append({
                        "content": doc_content,
                        "metadata": doc_metadata,
                        "score": score,
                        "cached": False
                    })
                    store_rerank_score_in_cache(query_hash, doc_hash, SECONDARY_RERANKER_MODEL_NAME, score)
            except Exception as e:
                logger.error(f"ERROR during secondary document reranking: {e}", extra={"query_snippet_log": query[:50], "error_detail": str(e)})
                traceback.print_exc()
                # Fallback to primary reranked docs if secondary reranking fails
                secondary_reranked_docs_with_scores = docs_for_secondary_reranking
        
        secondary_reranked_docs_with_scores.sort(key=lambda x: x["score"], reverse=True)
        final_reranked_docs = secondary_reranked_docs_with_scores
    else:
        logger.info("Secondary reranker not available or no documents for secondary reranking. Using primary reranked documents as final.")
        final_reranked_docs = filtered_primary_reranked_docs

    # Apply soft-filtering after final reranking
    filtered_final_reranked_docs = [doc for doc in final_reranked_docs if doc["score"] >= MIN_RERANK_SCORE_THRESHOLD]
    
    MIN_DOCS_FOR_CONTEXT_SOFT_LIMIT = 1
    if len(filtered_final_reranked_docs) < MIN_DOCS_FOR_CONTEXT_SOFT_LIMIT and len(final_reranked_docs) > len(filtered_final_reranked_docs):
        num_to_add = MIN_DOCS_FOR_CONTEXT_SOFT_LIMIT - len(filtered_final_reranked_docs)
        for doc in final_reranked_docs:
            if doc not in filtered_final_reranked_docs:
                filtered_final_reranked_docs.append(doc)
                num_to_add -= 1
                if num_to_add <= 0:
                    break
        filtered_final_reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Soft-filtered: Added documents to reach minimum context size ({MIN_DOCS_FOR_CONTEXT_SOFT_LIMIT}).",
                    extra={"initial_filtered_count": len(filtered_final_reranked_docs) - num_to_add, "final_filtered_count": len(filtered_final_reranked_docs)})
    
    logger.info(f"Final reranked to {len(filtered_final_reranked_docs)} documents after thresholding (score >= {MIN_RERANK_SCORE_THRESHOLD}) and soft-filtering.",
                extra={"final_reranked_doc_count": len(filtered_final_reranked_docs), "rerank_threshold": MIN_RERANK_SCORE_THRESHOLD})
    
    return fuse_adjacent_chunks(filtered_final_reranked_docs[:TOP_N_RERANKED_RESULTS])


def compute_confidence_score(reranked_chunks: List[Dict]) -> float:
    """
    Computes a confidence score based on the reranked chunks' scores and metadata.
    A higher score indicates more relevant and certain retrieved information.
    """
    if not reranked_chunks:
        return 0.0
    
    avg_score = sum([doc["score"] for doc in reranked_chunks]) / len(reranked_chunks)
    num_docs = len(reranked_chunks)
    
    boost = 0.0
    for doc in reranked_chunks:
        section_type = doc.get("metadata", {}).get("section_type")
        if section_type in ["technical_specifications", "features", "table", "table_row", "technical_highlights"]: # Added table_row
            boost += 0.15
        if "total_power_consumption_w" in doc.get("metadata", {}):
            boost += 0.1
        if "model_number" in doc.get("metadata", {}): # Boost if specific model numbers are present
            boost += 0.1
        if "hr_policy_category" in doc.get("metadata", {}): # Boost HR policy chunks
            boost += 0.05
    
    raw_score = avg_score + 0.05 * num_docs + boost
    
    # Confidence Score Normalization - Sigmoid normalization
    # Adjust 0.7 and 10 based on desired curve shape for your scores
    confidence = 1 / (1 + math.exp(-(raw_score - 0.7) * 10)) 
    
    return min(confidence, 1.0) # Cap at 1.0 for a cleaner confidence score


# Hybrid Search Optimization
def hybrid_search(query: str, username: str, n_results: int, query_filters: dict) -> List[Dict]:
    """
    Performs a hybrid search combining BM25 (keyword) and vector similarity.
    """
    if not BM25Okapi:
        logger.warning("BM25Okapi not available. Falling back to pure vector search in hybrid_search.",
                       extra={"reason": "BM25Okapi_not_imported"})
        # Fallback to pure vector search if BM25 is not available
        query_embedding_list = generate_ollama_embeddings([query])
        if not query_embedding_list:
            return []
        
        query_results = collection.query(
            query_embeddings=query_embedding_list,
            n_results=n_results,
            where=query_filters,
            include=['documents', 'embeddings', 'metadatas']
        )
        retrieved_doc_texts = query_results.get('documents', [[]])[0]
        retrieved_embeddings = query_results.get('embeddings', [[]])[0]
        retrieved_metadatas = query_results.get('metadatas', [[]])[0]

        initial_retrieved_docs = []
        if retrieved_doc_texts:
            query_embedding_vector = array(query_embedding_list[0])
            for doc_text, doc_emb_list_from_db, doc_meta in zip(retrieved_doc_texts, retrieved_embeddings, retrieved_metadatas):
                doc_embedding_vector = array(doc_emb_list_from_db)
                norm_query = norm(query_embedding_vector)
                norm_doc = norm(doc_embedding_vector)
                sim_score = dot(query_embedding_vector, doc_embedding_vector) / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0.0
                initial_retrieved_docs.append({"content": doc_text, "score": sim_score, "metadata": doc_meta})
        
        initial_retrieved_docs.sort(key=lambda item: item["score"], reverse=True)
        return initial_retrieved_docs


    logger.info(f"Performing hybrid search for query: '{query[:50]}...'")

    # 1. Get all documents matching filters for BM25 corpus
    # NOTE: ChromaDB's get() can be slow for very large collections. For production, consider
    # fetching only documents relevant to the user/company_data, or optimize this step.
    try:
        all_docs_in_filter = collection.get(where=query_filters, include=["documents", "metadatas"])
        if not all_docs_in_filter["documents"]:
            logger.info("No documents found for BM25 corpus after applying filters.")
            return [] # No documents to search
        
        bm25_corpus_documents = all_docs_in_filter["documents"]
        bm25_corpus_metadatas = all_docs_in_filter["metadatas"]
    except Exception as e:
        logger.error(f"Error fetching documents for BM25 corpus with filters {query_filters}: {e}",
                     extra={"query_filters_log": query_filters, "error_detail": str(e)})
        # Fallback to pure vector search if BM25 corpus creation fails
        query_embedding_list = generate_ollama_embeddings([query])
        if not query_embedding_list:
            return []
        query_results = collection.query(
            query_embeddings=query_embedding_list,
            n_results=n_results,
            where=query_filters,
            include=['documents', 'embeddings', 'metadatas']
        )
        retrieved_doc_texts = query_results.get('documents', [[]])[0]
        retrieved_embeddings = query_results.get('embeddings', [[]])[0]
        retrieved_metadatas = query_results.get('metadatas', [[]])[0]

        initial_retrieved_docs = []
        if retrieved_doc_texts:
            query_embedding_vector = array(query_embedding_list[0])
            for doc_text, doc_emb_list_from_db, doc_meta in zip(retrieved_doc_texts, retrieved_embeddings, retrieved_metadatas):
                doc_embedding_vector = array(doc_emb_list_from_db)
                norm_query = norm(query_embedding_vector)
                norm_doc = norm(doc_embedding_vector)
                sim_score = dot(query_embedding_vector, doc_embedding_vector) / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0.0
                initial_retrieved_docs.append({"content": doc_text, "score": sim_score, "metadata": doc_meta})
        
        initial_retrieved_docs.sort(key=lambda item: item["score"], reverse=True)
        return initial_retrieved_docs


    # Create BM25 corpus
    tokenized_corpus = [doc.split() for doc in bm25_corpus_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Get top BM25 docs
    tokenized_query = query.split()
    doc_scores_bm25 = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores (0-1 range)
    if doc_scores_bm25.max() > 0:
        bm25_scores_normalized = doc_scores_bm25 / doc_scores_bm25.max()
    else:
        bm25_scores_normalized = doc_scores_bm25 # All zeros if max is zero
    
    bm25_results = []
    for i, score in enumerate(bm25_scores_normalized):
        bm25_results.append({
            "content": bm25_corpus_documents[i],
            "metadata": bm25_corpus_metadatas[i],
            "score": float(score) # Ensure it's a float
        })
    
    bm25_results.sort(key=lambda x: x["score"], reverse=True)
    top_bm25_docs = bm25_results[:n_results]
    logger.debug(f"Top {len(top_bm25_docs)} BM25 documents retrieved.")

    # 2. Get top vector similarity docs with the same filters
    query_embedding_list = generate_ollama_embeddings([query])
    if not query_embedding_list:
        logger.error("Could not generate query embedding for vector search in hybrid_search.")
        return top_bm25_docs # Fallback to just BM25 if embedding fails

    query_results_vector = collection.query(
        query_embeddings=query_embedding_list,
        n_results=n_results,
        where=query_filters,
        include=['documents', 'embeddings', 'metadatas']
    )
    
    vector_doc_texts = query_results_vector.get('documents', [[]])[0]
    vector_embeddings = query_results_vector.get('embeddings', [[]])[0]
    vector_metadatas = query_results_vector.get('metadatas', [[]])[0]

    vector_similarity_results = []
    if vector_doc_texts:
        query_embedding_vector = array(query_embedding_list[0])
        for doc_text, doc_emb_list_from_db, doc_meta in zip(vector_doc_texts, vector_embeddings, vector_metadatas):
            doc_embedding_vector = array(doc_emb_list_from_db)
            norm_query = norm(query_embedding_vector)
            norm_doc = norm(doc_embedding_vector)
            sim_score = dot(query_embedding_vector, doc_embedding_vector) / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0.0
            vector_similarity_results.append({"content": doc_text, "score": sim_score, "metadata": doc_meta})

    vector_similarity_results.sort(key=lambda x: x["score"], reverse=True)
    top_vector_docs = vector_similarity_results[:n_results]
    logger.debug(f"Top {len(top_vector_docs)} vector similarity documents retrieved.")

    # 3. Combine and re-rank (simple fusion for now, reranker will do the heavy lifting)
    # Use a set to avoid duplicates based on content
    combined_docs_map = {}
    for doc in top_bm25_docs + top_vector_docs:
        doc_hash = hashlib.md5(doc["content"].encode('utf-8')).hexdigest()
        if doc_hash not in combined_docs_map:
            combined_docs_map[doc_hash] = doc
        else:
            # If a document appears in both, take the higher score
            combined_docs_map[doc_hash]["score"] = max(combined_docs_map[doc_hash]["score"], doc["score"])
    
    combined_docs = list(combined_docs_map.values())
    
    # Fuse with preference for exact matches (adjusted from original request to be applied here)
    for doc in combined_docs:
        bm25_score = next((b.get("score", 0) for b in top_bm25_docs if b["content"] == doc["content"]), 0)
        vector_score = next((v.get("score", 0) for v in top_vector_docs if v["content"] == doc["content"]), 0)
        
        # Normalize scores (already done for BM25, ensure vector score is also 0-1)
        # Assuming vector_score is already normalized by ChromaDB (cosine similarity)
        
        doc["score"] = (BM25_WEIGHT * bm25_score) + (VECTOR_WEIGHT * vector_score)

    combined_docs.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Hybrid search combined to {len(combined_docs)} unique documents.")
    
    return combined_docs


# Toggle to enable/disable LLM-based query rewriting
ENABLE_QUERY_REWRITE = True

def rewrite_query_if_needed(original_query: str, llm_call_fn=None) -> str:
    """
    Optionally rewrites a vague query into a detailed technical question using an LLM.
    Falls back to the original query if LLM is unavailable or fails.
    """
    try:
        if llm_call_fn is None or not ENABLE_QUERY_REWRITE:
            return original_query
        # Get the prompt template for query rewriting
        query_rewriting_prompt_template_raw = LLM_PROMPT_TEMPLATES.get("query_rewriting_prompt")
        if query_rewriting_prompt_template_raw is not None:
            if isinstance(query_rewriting_prompt_template_raw, list):
                query_rewriting_prompt_string = "\n".join(
                    [entry.get("content", "") for entry in query_rewriting_prompt_template_raw if isinstance(entry, dict)]
                )
                logger.warning("LLM_PROMPT_TEMPLATES['query_rewriting_prompt'] was a list. Flattened to string for caching/LLM call.")
            elif isinstance(query_rewriting_prompt_template_raw, str):
                query_rewriting_prompt_string = query_rewriting_prompt_template_raw
            else:
                logger.error("Query rewriting prompt template is neither a list nor a string. Cannot proceed with rewriting.")
                return original_query
            prompt = query_rewriting_prompt_string.format(query_to_rewrite=original_query)
        else:
            prompt = f"Rewrite this vague query into a detailed technical question: {original_query}"
        rewritten = llm_call_fn(prompt)
        if rewritten and isinstance(rewritten, str):
            logger.info(f"Rewritten Query: {rewritten}")
            return rewritten
        return original_query
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}. Using original query.")
        return original_query

def retrieve_context_from_vector_db(
    query: str,
    query_attributes: Dict[str, Any], 
    collection: Any, 
    embedding_fn: Any, 
    rerank_fn: Callable[[str, List[Dict], Optional[List[List[float]]]], List[Dict]], 
    llm_call_fn: Callable[[Any], str], 
    top_k: int = 10,
    initial_search_k: int = 50, 
    is_fallback_pass: bool = False, 
    current_username: str = "company_data", 
) -> Tuple[str, List, float, List, Dict]:
    """
    Retrieve relevant document context from a ChromaDB vector store, with reranking, hybrid search, and robust attribute filtering.
    All LLM calls (e.g., for attribute extraction) are made via the provided llm_call_fn parameter, which must be supplied by the caller.
    """
    # Use the passed query_attributes as the working extracted_query_params
    extracted_query_params = query_attributes 

    if not collection or not embedding_fn:
        logger.error("ChromaDB collection or embedding function not initialized. Cannot retrieve context.")
        return LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"], [], 0.0, [], {}

    start_retrieval_time = time.perf_counter()
    query_text = query 

    # --- CRITICAL FIX: Initialize all_retrieved_embeddings here ---
    all_retrieved_embeddings: List[List[float]] = []
    all_retrieved_embeddings_from_db: List[List[float]] = []  # Fix for line 3676 error
    debug_info = {}  # Fix for line 3906 error
    initial_retrieved_docs = []  # Always initialize before retrieval logic

    # --- Step 0: Use provided query_attributes, do NOT re-extract ---
    # query_attributes is assumed to be extracted once per query, upstream. Never re-extract here.

    # --- Normalize query attribute keys before filtering ---
    if not isinstance(query_attributes, dict):
        logger.error(
            f"Type Mismatch Error: Expected 'query_attributes' to be a dictionary, "
            f"but received type {type(query_attributes)}. Value: {query_attributes}. "
            f"Initializing to an empty dictionary for metadata normalization to prevent crash."
        )
        query_attributes_for_normalization = {}
    else:
        query_attributes_for_normalization = query_attributes
    strict_metadata_params = normalize_query_attributes(query_attributes)

    # --- Step 1: Generate Query Embedding ONCE and reuse ---
    query_embedding = embedding_cache.get_or_set(query_text, lambda text: embedding_fn.embed_query(text))

    if not query_embedding:
        logger.error("Could not generate query embedding. Returning empty context.", extra={"query_text_snippet": query_text[:50]})
        return LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"], [], 0.0, [], {}

    # --- Step 2: Initial Broad Retrieval (Semantic + BM25 Hybrid) ---
    effective_num_results = 8 

    # --- Diagnostic Logging: Enumerate unique 'user' and 'section_type' values ---
    try:
        user_ids = collection.get(include=["metadatas"])["metadatas"][0]
        unique_users = set()
        unique_section_types = set()
        for meta in user_ids:
            if isinstance(meta, dict):
                if "user" in meta:
                    unique_users.add(str(meta["user"]))
                if "section_type" in meta:
                    unique_section_types.add(str(meta["section_type"]))
        logger.info(f"Unique users in collection: {sorted(unique_users)}")
        logger.info(f"Unique section_types in collection: {sorted(unique_section_types)}")
    except Exception as e:
        logger.warning(f"Could not enumerate unique users/section_types: {e}")

    # Dynamically enumerate all unique users from collection metadata and normalize them
    try:
        meta = collection.get(include=["metadatas"])
        valid_users = list({str(m.get("user", "company_data")).strip().lower() for m in meta["metadatas"] if isinstance(m, dict) and "user" in m})
        logger.info(f"[RAG] Dynamically detected valid users for filtering: {valid_users}")
    except Exception as e:
        logger.warning(f"[RAG] Could not enumerate valid users for filtering: {e}")
        valid_users = [current_username, "company_data"]

    # Apply section_type filter for initial broad search if query is technical
    technical_keywords = ["spec", "technical", "parameter", "model", "voltage", "wattage", "datasheet", "features", "dimensions", "ip rating"]
    if any(kw in query_text.lower() for kw in technical_keywords):
        final_where_clause = {"$and": [
            {"user": {"$in": valid_users}},
            {"section_type": {"$in": ["technical_specifications", "table"]}}
        ]}
        logger.info(f"Enforced technical_specifications/table filter for initial broad search due to technical keywords: {final_where_clause}")
    else:
        final_where_clause = {"user": {"$in": valid_users}}
        logger.info(f"No strong technical keywords, initial broad search will not filter by section_type. Using: {final_where_clause}")

    # --- Strict Filtering: Try initial filter ---
    matching_ids_strict = []
    try:
        matching_ids_strict = collection.get(where=final_where_clause)["ids"]
        logger.info(f"Docs matching strict filter: {len(matching_ids_strict)}")
        if not matching_ids_strict:
            logger.warning("[BM25] No documents matched strict where clause. Relaxing filters.")
            # Relaxed filter: only by user (all users found in collection)
            relaxed_user_where = {"user": {"$in": list(unique_users)}} if unique_users else {"user": current_username}
            matching_ids_relaxed = collection.get(where=relaxed_user_where)["ids"]
            logger.info(f"Docs matching relaxed user filter: {len(matching_ids_relaxed)}")
            if matching_ids_relaxed:
                # Use relaxed filter for retrieval
                docs_relaxed = collection.get(ids=matching_ids_relaxed, include=["documents", "metadatas"])
                # Format docs_relaxed into initial_retrieved_docs format
                initial_retrieved_docs = []
                for i, doc in enumerate(docs_relaxed["documents"]):
                    meta = docs_relaxed["metadatas"][i]
                    initial_retrieved_docs.append({"content": doc, "metadata": meta})
            else:
                logger.warning("No docs matched relaxed filter. Will fallback to top-N semantic docs.")
                # Fallback: get top-N semantic docs (no filter)
                semantic_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
                initial_retrieved_docs = []
                if semantic_results and semantic_results.get('documents') and semantic_results.get('metadatas'):
                    for i in range(len(semantic_results['documents'][0])):
                        doc_content = semantic_results['documents'][0][i]
                        doc_meta = semantic_results['metadatas'][0][i]
                        initial_retrieved_docs.append({"content": doc_content, "metadata": doc_meta})
                logger.warning("No documents remaining after all filtering. Returning top semantic docs as fallback.")
        else:
            # Use strict filter for retrieval
            docs_strict = collection.get(ids=matching_ids_strict, include=["documents", "metadatas"])
            initial_retrieved_docs = []
            for i, doc in enumerate(docs_strict["documents"]):
                meta = docs_strict["metadatas"][i]
                initial_retrieved_docs.append({"content": doc, "metadata": meta})
    except Exception as e:
        logger.warning(f"[BM25] Fallback triggered due to error: {e}")
        initial_retrieved_docs = []

    # If still empty, fallback to top-N semantic docs
    if not initial_retrieved_docs:
        logger.warning("No documents remaining after all filtering. Returning top semantic docs as fallback.")
        semantic_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        if semantic_results and semantic_results.get('documents') and semantic_results.get('metadatas'):
            for i in range(len(semantic_results['documents'][0])):
                doc_content = semantic_results['documents'][0][i]
                doc_meta = semantic_results['metadatas'][0][i]
                initial_retrieved_docs.append({"content": doc_content, "metadata": doc_meta})

    try:
        if use_hybrid_search and BM25Okapi:
            # hybrid_search should return documents with 'content', 'metadata', 'score' and 'embedding' if available
            initial_retrieved_docs = hybrid_search(query, current_username, n_results=effective_num_results, query_filters=final_where_clause)
            # Extract embeddings from hybrid search results if they were included
            all_retrieved_embeddings_from_db = [doc["embedding"] for doc in initial_retrieved_docs if "embedding" in doc and doc["embedding"] is not None]
        else:
            # Perform semantic search and request doc embeddings
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_num_results,
                where=final_where_clause,
                include=['documents', 'metadatas', 'distances', 'embeddings']
            )
            
            if semantic_results and semantic_results.get('documents') and semantic_results.get('metadatas'):
                for i in range(len(semantic_results['documents'][0])):
                    doc_content = semantic_results['documents'][0][i]
                    doc_meta = semantic_results['metadatas'][0][i]
                    doc_distance = semantic_results['distances'][0][i]
                    doc_embedding = None
                    if 'embeddings' in semantic_results and semantic_results['embeddings'] and len(semantic_results['embeddings'][0]) > i:
                        doc_embedding = semantic_results['embeddings'][0][i]
                        all_retrieved_embeddings_from_db.append(doc_embedding) 
                    initial_retrieved_docs.append({
                        "content": doc_content,
                        "metadata": doc_meta,
                        "score": doc_distance, 
                        "embedding": doc_embedding
                    })
            logger.info(f"Semantic search retrieved {len(initial_retrieved_docs)} documents.")
            # Sort by distance (lower is better)
            initial_retrieved_docs.sort(key=lambda item: item["score"]) 
            
            # Convert distance to similarity score (0-1, higher is better) for consistency with reranker
            # Assuming distance is cosine distance (0 to 2), similarity = 1 - (distance / 2)
            for doc in initial_retrieved_docs:
                doc["score"] = 1 - (doc["score"] / 2) 
    except Exception as e:
        logger.error(f"Error during initial retrieval: {e}. Proceeding with fallback logic.", exc_info=True)
        # Fallback to pure vector search if initial retrieval fails
        query_embedding_list = generate_ollama_embeddings([query])
        if not query_embedding_list:
            return LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"], [], 0.0, [], {}
        query_results = collection.query(
            query_embeddings=query_embedding_list,
            n_results=effective_num_results,
            where=final_where_clause,
            include=['documents', 'embeddings', 'metadatas']
        )
        retrieved_doc_texts = query_results.get('documents', [[]])[0]
        retrieved_embeddings = query_results.get('embeddings', [[]])[0]
        retrieved_metadatas = query_results.get('metadatas', [[]])[0]

        initial_retrieved_docs = []
        if retrieved_doc_texts:
            query_embedding_vector = array(query_embedding_list[0])
            for doc_text, doc_emb_list_from_db, doc_meta in zip(retrieved_doc_texts, retrieved_embeddings, retrieved_metadatas):
                doc_embedding_vector = array(doc_emb_list_from_db)
                norm_query = norm(query_embedding_vector)
                norm_doc = norm(doc_embedding_vector)
                sim_score = dot(query_embedding_vector, doc_embedding_vector) / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0.0
                initial_retrieved_docs.append({"content": doc_text, "score": sim_score, "metadata": doc_meta})
        
        initial_retrieved_docs.sort(key=lambda item: item["score"], reverse=True)

    bm25_docs = []
    if BM25Okapi:
        try:
            filtered_ids = collection.get(where=final_where_clause)['ids']
            if not filtered_ids:
                logger.warning("🔍 BM25: No matching documents. Skipping BM25 retrieval.")
            else:
                all_chroma_docs = collection.get(
                    ids=filtered_ids,
                    include=['documents', 'metadatas']
                )
                tokenized_corpus = [doc.split() for doc in all_chroma_docs['documents'] if doc.strip()]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query_text.lower().split()
                doc_scores = bm25.get_scores(tokenized_query)

                # Combine scores with original documents
                scored_bm25_docs = []
                for i, score in enumerate(doc_scores):
                    meta = all_chroma_docs['metadatas'][i]
                    if meta.get('user') == current_username or meta.get('user') == 'company_data':
                        scored_bm25_docs.append({
                            "content": all_chroma_docs['documents'][i],
                            "metadata": meta,
                            "score": score
                        })
        except Exception as e:
            logger.warning(f"Error during BM25 search: {e}")
        # Sort BM25 results by score (descending)
            scored_bm25_docs.sort(key=lambda x: x['score'], reverse=True)
            bm25_docs = scored_bm25_docs[:effective_num_results] 
            logger.info(f"BM25 search retrieved {len(bm25_docs)} documents.")
        except Exception as e:
            logger.warning(f"Error during BM25 search: {e}. Skipping BM25 for this query.", exc_info=True)

    # Combine semantic and BM25 results, remove duplicates
    candidate_docs_map = {doc["metadata"].get("chunk_hash", doc["content"]): doc for doc in initial_retrieved_docs}
    for doc in bm25_docs:
        # Use a unique identifier, e.g., chunk_hash or content itself
        doc_id = doc["metadata"].get("chunk_hash", doc["content"])
        if doc_id not in candidate_docs_map:
            candidate_docs_map[doc_id] = doc
        else:
            # If a document is found in both, combine their scores or prioritize semantic
            # For simplicity, we'll keep the one already there (semantic)
            pass
    initial_retrieved_docs = list(candidate_docs_map.values()) 
    logger.info(f"Combined semantic and BM25 results: {len(initial_retrieved_docs)} unique documents.")

    # --- Step 3: Post-retrieval Filtering (Python-based, fuzzy matching) ---
    filtered_docs = []
    strict_metadata_params = query_attributes.copy() 

    # Ensure section_type fallback includes similar values like technical_specifications, technical_highlights, and table.
    if "section_type" in strict_metadata_params:
        requested_section_type = strict_metadata_params["section_type"].lower()
        # Define a mapping for related section types
        section_type_mapping = {
            "technical_specifications": ["technical_specifications", "technical_highlights", "table"],
            "hr_leaves_holidays": ["hr_leaves_holidays", "leave_policy", "holidays"],
            # Add more mappings as needed
        }
        # If the requested section type has variations, use them for filtering
        strict_metadata_params["section_type"] = section_type_mapping.get(requested_section_type, [requested_section_type])
        logger.info(f"Normalized section_type filter to: {strict_metadata_params['section_type']}")

    if strict_metadata_params:
        logger.info(f"Applying strict post-retrieval fuzzy filtering with params: {strict_metadata_params}")
        for doc in initial_retrieved_docs: 
            doc_metadata = doc.get('metadata', {})
            reasons_for_exclusion = fuzzy_match_metadata(doc_metadata, strict_metadata_params, logger)
            
            if not reasons_for_exclusion: 
                filtered_docs.append(doc)
            else:
                logger.debug(f"Chunk filtered out by strict metadata: {reasons_for_exclusion} | Doc Meta: {doc_metadata} | Content: {doc['content'][:80]}...")
    else:
        filtered_docs = initial_retrieved_docs 
        logger.info("No strict metadata parameters for post-filtering. All initial candidates passed to reranking.")
    
    logger.info(f"Post-filtered docs (strict): {len(filtered_docs)} / {len(initial_retrieved_docs)}")

    # Fallback if strict filtering yields no results
    if not filtered_docs and strict_metadata_params and not is_fallback_pass: 
        logger.warning("Strict filtering returned 0 documents. Attempting relaxed filtering.")
        relaxed_filtered_docs = []
        
        # Create a relaxed version of strict_metadata_params
        relaxed_params = strict_metadata_params.copy()
        # Example relaxation: remove 'model_number' if it was present, or lower fuzzy thresholds
        relaxed_params.pop('model_number', None) 
        
        for doc in initial_retrieved_docs: 
            doc_metadata = doc.get('metadata', {})
            # Use a more relaxed fuzzy match threshold (e.g., 70 instead of 85)
            # Or skip certain strict checks
            reasons_for_exclusion = fuzzy_match_metadata(doc_metadata, relaxed_params, logger)
            
            if not reasons_for_exclusion:
                relaxed_filtered_docs.append(doc)
            else:
                logger.debug(f"Chunk filtered out by relaxed metadata: {reasons_for_exclusion} | Doc Meta: {doc_metadata} | Content: {doc['content'][:80]}...")
        
        filtered_docs = relaxed_filtered_docs
        logger.info(f"Post-filtered docs (relaxed): {len(filtered_docs)} / {len(initial_retrieved_docs)}")

    if not filtered_docs:
        logger.warning("[retrieve_context_from_vector_db] No relevant chunks retrieved after filtering. Fallback logic may be triggered.")
        logger.warning("No documents remaining after all filtering. Returning initial candidates for best effort.")
        # If all filtering fails, return the top N semantic results as a last resort
        initial_retrieved_docs.sort(key=lambda d: d['score']) 
        final_docs_for_context = initial_retrieved_docs[:top_k]
        context_final_str = format_docs_for_llm(final_docs_for_context)
        # Calculate a low confidence score if no docs passed filters
        confidence_score = 0.1 
        source_filenames = list(set([d["metadata"].get("source_file_name", "N/A") for d in final_docs_for_context]))
        return context_final_str, final_docs_for_context, confidence_score, source_filenames, query_attributes 
    
    # --- Step 4: Use cached doc embeddings for filtered docs; only generate if missing ---
    embeddings_for_reranking = []
    docs_needing_embedding_generation = []
    doc_indices_needing_embedding = []
    for idx, doc in enumerate(filtered_docs):
        if doc.get("embedding") is not None:
            embeddings_for_reranking.append(doc["embedding"])
        else:
            docs_needing_embedding_generation.append(doc["content"])
            doc_indices_needing_embedding.append(idx)
    # Only generate embeddings for docs missing them
    if docs_needing_embedding_generation:
        # This assumes embedding_fn is capable of batching, or generate_ollama_embeddings is used.
        # If embedding_fn is a single-doc embedder, you'd loop.
        try:
            if hasattr(embedding_fn, 'embed_documents'):
                generated_embs = embedding_fn.embed_documents(docs_needing_embedding_generation)
            else:
                # Fallback to single document embedding if batch method not available
                generated_embs = [embedding_fn.embed_query(doc) for doc in docs_needing_embedding_generation]
        except Exception as e:
            logger.error(f"Error generating embeddings for filtered docs: {e}")
            generated_embs = []
        for i, emb in enumerate(generated_embs):
            # Insert at the correct position
            insert_idx = doc_indices_needing_embedding[i]
            embeddings_for_reranking.insert(insert_idx, emb)
            filtered_docs[insert_idx]["embedding"] = emb
    if not embeddings_for_reranking or len(embeddings_for_reranking) != len(filtered_docs):
        logger.warning("Failed to obtain all embeddings for filtered documents. Reranking might be impacted or skipped.")
        final_reranked_docs = filtered_docs
    else:
        final_docs_for_reranking = filtered_docs 

        # Optional: Apply MMR on the filtered set if desired, using embeddings_for_reranking
        if use_mmr and len(final_docs_for_reranking) > rerank_top_n:
            if callable(maximal_marginal_relevance):
                logger.info(f"Applying ChromaDB's built-in MMR for diversity on filtered docs (factor: {MMR_DIVERSITY_FACTOR})...",
                            extra={"mmr_diversity_factor_value": MMR_DIVERSITY_FACTOR})
                
                query_embedding_for_mmr = array(query_embedding) 

                selected_indices = maximal_marginal_relevance(
                    query_embedding_for_mmr,
                    embeddings_for_reranking,
                    k=rerank_top_n,
                    lambda_mult=MMR_DIVERSITY_FACTOR
                )
                temp_selected_docs_after_mmr = [final_docs_for_reranking[i] for i in selected_indices]
                
                if len(temp_selected_docs_after_mmr) > 0:
                    final_docs_for_reranking = temp_selected_docs_after_mmr
                    logger.info(f"ChromaDB MMR selected {len(final_docs_for_reranking)} documents from filtered set for reranking.")
                else:
                    logger.warning("MMR on filtered docs did not select any. Proceeding with all filtered documents for reranking.")
            else:
                logger.warning("maximal_marginal_relevance is not callable. Skipping MMR reranking.")
        
        # Always rerank the final_docs_for_reranking (which might be MMR-selected or all filtered docs).
        # Pass all_retrieved_embeddings to rerank_fn, as it might need them for its internal logic.
        if callable(rerank_fn) and final_docs_for_reranking:
            try:
                final_reranked_docs = rerank_fn(query, final_docs_for_reranking, all_retrieved_embeddings)
            except Exception as e:
                logger.error(f"Error during reranking: {e}. Using filtered documents directly.")
                final_reranked_docs = final_docs_for_reranking
        else:
            if not callable(rerank_fn):
                logger.warning("Rerank function is not callable. Skipping reranking and using filtered documents directly.")
            final_reranked_docs = final_docs_for_reranking or filtered_docs 

    debug_info["reranked_docs"] = final_reranked_docs[:5] 

    # --- Step 5: Select Top N Reranked Results ---
    # The rerank_documents function already sorts by 'final_rerank_score'
    final_docs_for_context = final_reranked_docs[:TOP_N_RERANKED_RESULTS]
    logger.info(f"Selected top {len(final_docs_for_context)} documents after reranking.")

    # --- Step 6: Format Context for LLM & Calculate Confidence ---
    context_final_str = format_docs_for_llm(final_docs_for_context)
    
    # Calculate confidence based on the top reranked document's score
    # Use .get to avoid KeyError if 'final_rerank_score' is missing (fallback to 'score' or 0.0)
    confidence_score = final_docs_for_context[0].get("final_rerank_score", final_docs_for_context[0].get("score", 0.0)) if final_docs_for_context else 0.0
    
    source_filenames = list(set([d["metadata"].get("source_file_name", "N/A") for d in final_docs_for_context]))

    # Extract unique metadata from final retrieved documents for response
    retrieved_unique_metadata = defaultdict(set)
    for item in final_docs_for_context:
        for k, v in item["metadata"].items():
            # Exclude internal/ChromaDB specific metadata fields
            if k not in ["filename", "user", "source_file_name", "processing_date", "chunk_order", "_embeddings", "_id", "_collection", "chunk_length", "chunk_hash", "source_doc_type"]:
                sanitized_value = sanitize_text(str(v)).lower()
                if sanitized_value:
                    retrieved_unique_metadata[k].add(sanitized_value)
    final_retrieved_metadata_for_response = {k: list(v) for k, v in retrieved_unique_metadata.items()}

    end_retrieval_time = time.perf_counter()
    retrieval_duration = (end_retrieval_time - start_retrieval_time) * 1000
    logger.info(f"Final RAG context string generated (first 100 chars): '{context_final_str[:100]}...' in {retrieval_duration:.2f} ms",
                extra={"retrieval_time_ms_log": retrieval_duration, "context_length_chars": len(context_final_str)})

    return context_final_str, final_docs_for_context, confidence_score, source_filenames, final_retrieved_metadata_for_response


def ask_llm_model_phi3(messages: Any, temperature: float = 0.0, max_tokens: int = 150, stream: bool = False, model: str = "llama3:8b", api_base: str = OLLAMA_API_URL) -> str:
    """
    Sends a list of messages to the Ollama LLM model and returns the response.
    Includes retry logic for network requests and output validation.
    Adds defensive checks for empty prompts and logs outgoing payloads.
    """
    if not SYSTEM_READY:
        logger.error("RAG system not ready, skipping LLM call.", extra={"rag_system_status": SYSTEM_READY})
        return "I'm sorry, the AI system is not fully initialized at the moment. Please try again in a few moments."

    # Normalize messages to list of dicts
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, dict):
        messages = [messages]
    elif not isinstance(messages, list):
        logger.error(f"ask_llm_model_phi3 received unexpected 'messages' type: {type(messages)}. Forcing to empty list.")
        messages = []

    # Defensive check: Only access .get if dict
    if not messages or not any(isinstance(msg, dict) and msg.get("content", "").strip() for msg in messages):
        logger.warning("No valid content found in messages for LLM request.")
        return ""

    # Filter out malformed messages
    chat_messages = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            logger.warning(f"Skipping malformed message: {msg}")

    # ... rest of your function continues to make the API call

    # Log the outgoing payload
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages
    }
    logger.debug(f"Sending prompt to Ollama:\n{json.dumps(payload, indent=2)}")

    try:
        logger.info(f"Sending request to Ollama API ({OLLAMA_API_URL})...",
                    extra={"ollama_api_url_log": OLLAMA_API_URL, "model_name_log": OLLAMA_MODEL_NAME})
        start_llm_call_time = time.perf_counter()
        response = session.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL_NAME,
                "messages": messages,
                "stream": False
            },
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()
        
        json_response = response.json()
        raw_answer = json_response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        end_llm_call_time = time.perf_counter()
        llm_call_duration = (end_llm_call_time - start_llm_call_time) * 1000
        logger.info(f"LLM Raw Response received: {raw_answer} in {llm_call_duration:.2f} ms",
                    extra={"response_content_full": raw_answer, "llm_call_time_ms": llm_call_duration})
        
        return raw_answer
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get response from Ollama: {e}", extra={"error_detail": str(e)})
        traceback.print_exc()
        return f"I'm sorry, I couldn't get a response from the AI model. Please ensure Ollama is running and the model '{OLLAMA_MODEL_NAME}' is available. Error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call: {e}", extra={"error_detail": str(e)})
        traceback.print_exc()
        return f"An unexpected error occurred while processing your request: {e}"


def build_llm_prompt(context: str, query: str, chat_history: List[Dict]) -> List[Dict]:
    """
    Constructs the prompt messages for the LLM, incorporating system instructions,
    retrieved context, and chat history. Uses config-driven templates.
    """
    messages = []

    system_instruction = LLM_PROMPT_TEMPLATES["system_instruction"]
    if context and context.strip() != LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"]:
        system_instruction += context
    else:
        # This branch should ideally not be hit if iterative retrieval works well
        system_instruction = LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"]
        messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": query})
        return messages

    system_instruction += LLM_PROMPT_TEMPLATES["chat_history_header"]
    if not chat_history:
        system_instruction += LLM_PROMPT_TEMPLATES["no_chat_history"]
    
    messages.append({"role": "system", "content": system_instruction})

    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    return messages

# NEW: Function to validate LLM's answer against context
def validate_llm_answer(query: str, llm_answer: str, context: str) -> Tuple[bool, str]:
    """
    Validates if the LLM's generated answer is fully supported by the provided context.
    Uses a separate LLM call for self-correction/validation.
    """
    if not llm_answer.strip() or not context.strip():
        return False, "Answer or context is empty for validation."

    validation_prompt_messages = [
        {"role": "system", "content": LLM_PROMPT_TEMPLATES["answer_validation_system"]},
        {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{llm_answer}"}
    ]
    
    validation_response_raw = ask_llm_model_phi3(validation_prompt_messages)
    
    validation_schema = {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "reason": {"type": "string"}
        },
        "required": ["valid", "reason"]
    }
    
    validation_result = safe_parse_llm_json_response(validation_response_raw, schema=validation_schema)

    if validation_result and isinstance(validation_result.get("valid"), bool):
        return validation_result["valid"], validation_result.get("reason", "Validation performed.")
    else:
        logger.warning(f"LLM answer validation returned invalid JSON or missing 'valid' field. Assuming valid for now. Raw: {validation_response_raw[:100]}...")
        return True, "Validation result unclear, assumed valid." # Default to True if validation fails

# MODIFIED: Main RAG query processing function to include iterative retrieval and self-correction
def ask_rag_system(query: str, username: str, chat_history: List[Dict]) -> Tuple[str, List, List, List, Dict]:
    """
    Main function to ask the RAG system a query.
    Implements iterative retrieval and LLM self-correction for higher accuracy.
    """
    if not SYSTEM_READY:
        logger.error("RAG system not ready. Cannot process query.", extra={"rag_system_status": SYSTEM_READY})
        return "I'm sorry, the AI system is currently unavailable. Please try again later.", [], [], [], {}

    logger.info(f"Processing query: '{query}' for user: '{username}'")

    # --- Extract query attributes (metadata) ONCE ---
    try:
        raw_metadata = extract_metadata_with_llm(query)
        query_attributes = parse_metadata(raw_metadata)
        logger.info(f"Extracted query_attributes: {query_attributes}")
    except Exception as e:
        logger.error(f"Failed to extract metadata from query: {e}", extra={"query": query})
        query_attributes = {}

    # --- First Pass Retrieval ---
    context_str, retrieved_docs, confidence, source_filenames, retrieved_metadata = \
        retrieve_context_from_vector_db(
            query=query,
            query_attributes=query_attributes,
            collection=collection,
            embedding_fn=embedding_fn,
            rerank_fn=rerank_fn,
            llm_call_fn=ask_llm_model_phi3,
            is_fallback_pass=False,
            current_username=username
        )

    final_answer = ""
    final_source_docs = retrieved_docs
    final_source_filenames = source_filenames
    final_retrieved_metadata = retrieved_metadata

    # --- Iterative Retrieval Logic ---
    if confidence < LOW_CONFIDENCE_THRESHOLD_CLARIFY:
        logger.info(f"First pass confidence ({confidence:.2f}) is low. Attempting second, broader retrieval pass.")
        # Second Pass: Broader search
        context_str_fallback, retrieved_docs_fallback, confidence_fallback, source_filenames_fallback, retrieved_metadata_fallback = \
            retrieve_context_from_vector_db(
                query=query,
                query_attributes=query_attributes,
                collection=collection,
                embedding_fn=embedding_fn,
                rerank_fn=rerank_fn,
                llm_call_fn=ask_llm_model_phi3,
                is_fallback_pass=True,
                current_username=username
            )

        if confidence_fallback > confidence: # If fallback improved confidence significantly
            logger.info(f"Fallback pass improved confidence from {confidence:.2f} to {confidence_fallback:.2f}. Using fallback context.")
            context_str = context_str_fallback
            retrieved_docs = retrieved_docs_fallback
            confidence = confidence_fallback
            source_filenames = source_filenames_fallback
            retrieved_metadata = retrieved_metadata_fallback
        else:
            logger.info(f"Fallback pass did not significantly improve confidence. Sticking with first pass or no context.")

    # --- Determine Final Response based on Confidence ---
    if confidence < VERY_LOW_CONFIDENCE_THRESHOLD_NO_ANSWER:
        logger.warning(f"Very low confidence ({confidence:.2f}) after all retrieval attempts. Returning 'no information' response.")
        final_answer = LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"]
        final_source_docs = []
        final_source_filenames = []
        final_retrieved_metadata = {}
    elif confidence < LOW_CONFIDENCE_THRESHOLD_CLARIFY:
        logger.info(f"Low confidence ({confidence:.2f}) after retrieval. Suggesting clarification to user.")
        final_answer = LLM_PROMPT_TEMPLATES["clarification_prompt"]
        final_source_docs = [] # No documents if asking for clarification
        final_source_filenames = []
        final_retrieved_metadata = {}
    else:
        # Sufficient confidence, proceed with LLM generation
        logger.info(f"Sufficient confidence ({confidence:.2f}). Generating LLM answer with strict prompt template.")

        # Strict prompt template for context-grounded answer
        strict_prompt = (
            "You are a helpful product assistant for Pyrotech Electronics Pvt. Ltd.\n"
            "Your task is to answer user queries based *only* on the provided \"Document Data\".\n"
            "If the answer cannot be found or fully supported by the \"Document Data\", state \"Information not found in the provided documents.\"\n"
            "Do not make up information. Do not use external knowledge.\n\n"
            f"User Query: {query}\n\n"
            f"Document Data:\n{context_str}\n\n"
            "Answer:"
        )
        logger.info(f"Sending strict prompt to LLM:\n{strict_prompt[:500]}...")
        raw_llm_response = ask_llm_model_phi3(strict_prompt)
        final_answer = raw_llm_response

        # --- LLM Self-Correction/Validation ---
        logger.info("Performing LLM self-correction/validation on generated answer.")
        is_valid, validation_reason = validate_llm_answer(query, final_answer, context_str)

        if not is_valid:
            logger.warning(f"LLM answer failed self-validation: {validation_reason}. Reverting to 'no information'.")
            final_answer = LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"]
            final_source_docs = []
            final_source_filenames = []
            final_retrieved_metadata = {}
        else:
            logger.info(f"LLM answer passed self-validation: {validation_reason}.")
            final_source_docs = retrieved_docs # Keep the docs if answer is valid
            final_source_filenames = source_filenames
            final_retrieved_metadata = retrieved_metadata

    logger.info(f"Final RAG answer: '{final_answer[:100]}...'")
    return final_answer, final_source_docs, [], final_source_filenames, final_retrieved_metadata


def ingest_txt_documents_from_folder(folder_path: str, username: str = "company_data"):
    """
    Ingests TXT/CSV/JSON/PDF/DOCX documents from a specified folder (and its subfolders)
    into the RAG knowledge base. Uses ProcessPoolExecutor for CPU-bound tasks.
    """
    logger.info(f"Starting ingestion of documents from: '{folder_path}' for user '{username}'")
    
    if not SYSTEM_READY or not collection:
        logger.error("RAG system not ready for ingestion. Skipping document ingestion.",
                     extra={"system_ready_status": SYSTEM_READY, "collection_available": bool(collection)})
        return

    documents_to_process = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            # Added .pdf and .docx
            if file_extension in [".txt", ".csv", ".json", ".pdf", ".docx"]:
                documents_to_process.append((file_path, filename, file_extension))
    
    logger.info(f"Found {len(documents_to_process)} documents to ingest (including subfolders).")
    
    if not documents_to_process:
        logger.info("No documents found to ingest in this directory or its subdirectories.")
        return

    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = []
        for doc_path, doc_filename, doc_extension in documents_to_process:
            futures.append(executor.submit(_process_single_document_for_ingestion, doc_path, doc_filename, doc_extension, username))
        
        for future in futures:
            try:
                result_msg = future.result()
                logger.info(result_msg)
            except Exception as e:
                logger.error(f"ERROR during batch processing of a document in batching function: {e}", extra={"error_detail": str(e)})
                traceback.print_exc()
        logger.info("Batch processing complete.")


def extract_document_metadata(filename: str, text_content: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Extracts metadata for a document, first by attempting to use an LLM with a retry mechanism,
    and then falling back to regex-based extraction from the filename if the LLM fails.
    """
    from rag_utils import extract_metadata_with_llm
    from rag_ingest import extract_product_name_and_power_from_filename

    # Attempt to extract metadata using the LLM with retries
    for attempt in range(max_retries):
        try:
            llm_metadata = extract_metadata_with_llm(text_content)
            if llm_metadata and llm_metadata.get("product_name"):
                logger.info(f"Successfully extracted metadata for '{filename}' using LLM on attempt {attempt + 1}.")
                llm_metadata["source_file_name"] = filename # Ensure filename is correctly set
                return llm_metadata
        except Exception as e:
            logger.warning(f"LLM metadata extraction attempt {attempt + 1} for '{filename}' failed: {e}")
    
    # Fallback to regex-based extraction if LLM fails
    logger.warning(f"LLM metadata extraction failed after {max_retries} attempts. Falling back to filename-based extraction for '{filename}'.")
    product_name, power_consumption = extract_product_name_and_power_from_filename(filename)
    # Clean up fallback values: never set 'unknown', use '' or 0.0 if missing
    if not product_name or product_name.lower() == 'unknown':
        product_name = infer_product_name(text_content, filename)
        if not product_name or product_name.lower() == 'unknown':
            product_name = ''
    if not power_consumption or power_consumption == 'unknown':
        from rag_utils import parse_power_consumption
        power_consumption = parse_power_consumption(filename)
        if power_consumption is None:
            power_consumption = 0.0
    fallback_metadata = {
        "product_name": product_name,
        "power_consumption": power_consumption,
        "section_type": "general", # Default section type for fallback
        "source_file_name": filename,
    }
    logger.info(f"Fallback metadata used for '{filename}': {fallback_metadata}")
    return fallback_metadata

def _process_single_document_for_ingestion(file_path: str, filename: str, file_extension: str, username: str) -> str:
    """
    Processes a single document (TXT, CSV, JSON, PDF, DOCX) for ingestion into ChromaDB.
    Handles text extraction, chunking, embedding, and metadata association.
    Includes idempotency check.
    """
    logger.info(f"Processing document for storage: '{filename}' by user: '{username}'", extra={"doc_filename_log": filename, "ingestion_user": username})
    text_content = ""
    elements = []
    try:
        # Text extraction logic remains the same...
        if file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text_content = f.read()
            text_content = raw_text_content.strip()
            logger.info(f"Read TXT file: {file_path}", extra={"file_path_log": file_path})
            elements = unstructured_partition_text(text=text_content)

            # --- SKIP LLM for structured .txt with TECHNICAL SPECIFICATIONS ---
            if "TECHNICAL SPECIFICATIONS" in text_content.upper():
                logger.info(f"Skipping LLM for '{filename}' — structured specs detected.")
                from rag_ingest import extract_table_fields, extract_product_name_and_power_from_filename, extract_product_name
                fields = extract_table_fields(text_content)
                product_name, power_consumption = extract_product_name_and_power_from_filename(filename)
                extracted_metadata = {
                    "product_name": product_name,
                    "power_consumption": power_consumption,
                    "section_type": "technical_specifications",
                    "source_file_name": filename,
                    "chunk_id": str(uuid.uuid4()),
                    "table_fields": fields,
                    "user": username,
                }
                # Avoid "unknown" for numeric fields
                if extracted_metadata.get("power_consumption") == "unknown":
                    extracted_metadata["power_consumption"] = None
                if extracted_metadata.get("wattage") == "unknown":
                    extracted_metadata["wattage"] = None
                # Enforce presence of key fields
                if not extracted_metadata.get("product_name"):
                    extracted_metadata["product_name"] = extract_product_name(text_content, filename)
                if not extracted_metadata.get("section_type"):
                    extracted_metadata["section_type"] = "technical_specifications"
            else:
                # fallback: will use LLM as before
                pass

        elif file_extension == ".csv":
            elements = unstructured_partition_csv(filename=file_path)
            text_content_raw = "\n\n".join([
                str(el) for el in elements
                if isinstance(el, (Text, NarrativeText, Title, Table, ListItem)) and str(el).strip()
            ])
            text_content = text_content_raw.strip()

            logger.info(f"Partitioned CSV file: {file_path}", extra={"file_path_log": file_path})

        elif file_extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            processed_json_data = _process_json_for_html_tables(json_data)
            
            text_content_parts = []
            markdown_table_found = False

            if isinstance(processed_json_data, list):
                for item in processed_json_data:
                    if isinstance(item, str):
                        if re.match(r'^\|.*\|$\n^\|[-:|=]+\|$\n(?:^\|.*\|$\n?)+', item, re.DOTALL | re.MULTILINE):
                            text_content_parts.append(item)
                            markdown_table_found = True
                        else:
                            text_content_parts.append(item)
                    elif isinstance(item, dict):
                        if 'type' in item and item['type'] == 'table' and 'content' in item and \
                           isinstance(item['content'], str) and \
                           re.match(r'^\|.*\|$\n^\|[-:|=]+\|$\n(?:^\|.*\|$\n?)+', item['content'], re.DOTALL | re.MULTILINE):
                            text_content_parts.append(item['content'])
                            markdown_table_found = True
                        else:
                            try:
                                flat_dict_str = json.dumps(item)
                                text_content_parts.append(", ".join(f"{k}: {str(v)}" for k, v in json.loads(flat_dict_str).items()))
                            except Exception as e_inner:
                                logger.warning(f"Could not parse dictionary item as flat dict, treating as raw string. Error: {e_inner}", extra={"item_content_snippet_log": str(item)[:100]})
                                text_content_parts.append(str(item))
                    else:
                        logger.warning(f"Skipping unexpected type in JSON list item: {type(item)}. Content: {item}", extra={"item_type_log": type(item).__name__, "item_content_snippet_log": str(item)[:100]})
                
                text_content = "\n\n".join(text_content_parts)

            elif isinstance(processed_json_data, dict):
                dict_content_parts = []
                for key, value in processed_json_data.items():
                    if isinstance(value, str):
                        if re.match(r'^\|.*\|$\n^\|[-:|=]+\|$\n(?:^\|.*\|$\n?)+', value, re.DOTALL | re.MULTILINE):
                            dict_content_parts.append(value)
                            markdown_table_found = True
                        else:
                            dict_content_parts.append(f"{key}: {value}")
                    elif isinstance(value, dict):
                        if 'type' in value and value['type'] == 'table' and 'content' in value and \
                           isinstance(value['content'], str) and \
                           re.match(r'^\|.*\|$\n^\|[-:|=]+\|$\n(?:^\|.*\|$\n?)+', value['content'], re.DOTALL | re.MULTILINE):
                            dict_content_parts.append(value['content'])
                            markdown_table_found = True
                        else:
                            try:
                                dict_content_parts.append(f"{key}: {json.dumps(value)}")
                            except Exception as e_inner:
                                logger.warning(f"Could not parse nested dictionary value, treating as raw string. Error: {e_inner}", extra={"dict_key_log": key, "dict_value_snippet_log": str(value)[:100]})
                                dict_content_parts.append(f"{key}: {str(value)}")
                    elif isinstance(value, list):
                        try:
                            dict_content_parts.append(f"{key}: {json.dumps(value)}")
                        except Exception as e_inner:
                            logger.warning(f"Could not parse nested list value, treating as raw string. Error: {e_inner}", extra={"dict_key_log": key, "dict_value_snippet_log": str(value)[:100]})
                            dict_content_parts.append(f"{key}: {str(value)}")
                    else:
                        dict_content_parts.append(f"{key}: {str(value)}")
                
                if markdown_table_found:
                    text_content = "\n\n".join(dict_content_parts)
                else:
                    text_content = json.dumps(processed_json_data, indent=2)

            elif isinstance(processed_json_data, str):
                text_content = processed_json_data
            
            else:
                text_content = json.dumps(processed_json_data, indent=2)
                logger.warning(f"JSON file '{filename}' did not contain a list, dict, or string at top level after processing. Treating as raw JSON string. Type: {type(processed_json_data).__name__}", extra={"doc_filename_log": filename, "json_data_type": type(processed_json_data).__name__})

            logger.info(f"Read and processed JSON file: {file_path}", extra={"file_path_log": file_path})
            # No elements from unstructured for JSON in this path for now
            # extracted_metadata = {"source_file_name": filename} # This might be redundant if extract_product_metadata adds it.
            # extracted_metadata.update(extract_product_metadata(text_content, filename, elements)) # Pass elements if available

        elif file_extension in [".pdf", ".docx"]:
            logger.info(f"Partitioning {file_extension} file using unstructured.partition.auto: {file_path}", extra={"file_path_log": file_path})
            elements = partition_auto(filename=file_path)
            text_content = "\n\n".join([str(el) for el in elements if str(el).strip()])
            text_content = text_content.strip()

        else:
            return f"WARNING: Skipping unsupported file type '{file_extension}' for ingestion: {filename}"

        # --- CENTRALIZED TEXT PREPROCESSING AFTER text_content IS ASSEMBLED ---
        # This block replaces the duplicated code in each 'if/elif' branch.
        text_content = re.sub(r'<[^>]+>', '', text_content) # Remove all HTML tags
        text_content = re.sub(r'\s*\n\s*\n\s*', '\n\n', text_content) # Normalize multiple newlines
        text_content = re.sub(r'(?i)(product category|product category)\s*', '', text_content) # Specific removals
        text_content = re.sub(r'\s*\(W\)\s*\+/\-5W\s*', ' (W)+/-5W ', text_content) # Specific replacements
        text_content = re.sub(r'\"([^\"]+)\"\s*,\s*\"([^\"]+)\"', r'\1: \2', text_content) # Normalize quoted pairs
        text_content = re.sub(r'^\s*[\-\*\•]\s*', '- ', text_content, flags=re.MULTILINE) # Normalize list bullets
        text_content = re.sub(r'^\s*\d+\.\s*', 'Numbered List Item: ', text_content, flags=re.MULTILINE) # Normalize numbered lists
        text_content = re.sub(r'\s+', ' ', text_content).strip() # Normalize all whitespace to single space

        logger.debug(f"Text content from '{filename}' after preprocessing (first 500 chars):\n{text_content[:500]}...",
                     extra={"doc_filename_log": filename, "content_snippet_log": text_content[:500]})

        # Extract metadata after text_content and elements (if any) are ready
        # extracted_metadata is initialized as {} at the start, then updated.
        extracted_metadata.update(extract_product_metadata(text_content, filename, elements))
        
        # Ensure 'source_file_name' is always in metadata
        extracted_metadata["source_file_name"] = filename

        if not text_content.strip():
            return f"Processed {filename}: Warning: Document contains no usable text after parsing and preprocessing. Skipping."

        logger.info(f"Chunking text from '{filename}' with content length {len(text_content)}...",
                            extra={"doc_filename_log": filename, "content_length_chars": len(text_content)})
        
        chunks_with_metadata = chunk_document_text(text_content, filename, elements,
                                            min_chars_per_chunk=MIN_CHUNK_CHARS, 
                                            max_chars_per_chunk=MAX_CHUNK_CHARS, 
                                            overlap_chars=OVERLAP_CHARS,
                                            metadata=extracted_metadata) # Pass extracted_metadata here
        
        logger.debug(f"[{filename}] Chunking {len(text_content)} chars -> {len(chunks_with_metadata)} chunks.",
                        extra={"doc_filename_log": filename, "original_chars_count": len(text_content), "num_chunks_created": len(chunks_with_metadata)})

        if not chunks_with_metadata:
            return f"Processed {filename}: Error: No usable text chunks generated from {filename} after all attempts. Document might be genuinely empty or unprocessable."

        chunks = [c['content'] for c in chunks_with_metadata]
        metadatas = [c['metadata'] for c in chunks_with_metadata]

        chunk_embeddings = generate_ollama_embeddings(chunks)
        if not chunk_embeddings:
            return f"Processed {filename}: Error: Failed to generate embeddings for chunks. Skipping ingestion."

        # --- Filter and sanitize metadatas before saving ---
        from rag_llm_json_utils import filter_noisy_metadata
        from rag_utils import sanitize_metadata_for_chromadb
        
        final_chunks, final_embeddings, final_metadatas, final_ids = [], [], [], []
        skipped_count = 0

        for chunk, embedding, meta, chunk_id in zip(chunks, chunk_embeddings, metadatas, ids):
            # 1. First, check for essential fields before any processing
            if not meta.get("product_name") or not meta.get("section_type"):
                logger.warning(f"Skipping chunk for '{filename}': Missing 'product_name' or 'section_type'. Metadata: {meta}")
                skipped_count += 1
                continue

            # 2. Clean noisy placeholder values
            cleaned_meta = filter_noisy_metadata(meta)

            # 3. Sanitize for ChromaDB compatibility (handles None, types, numerics)
            sanitized_meta = sanitize_metadata_for_chromadb(cleaned_meta)

            # 4. Flatten the final, clean metadata
            processed_meta = flatten_metadata(sanitized_meta)

            final_chunks.append(chunk)
            final_embeddings.append(embedding)
            final_metadatas.append(processed_meta)
            final_ids.append(chunk_id)

        logger.info(f"[{filename}] Filtered out {skipped_count} chunks due to missing required metadata. {len(final_chunks)} chunks remain for ChromaDB.")
        if not final_chunks:
            return f"Processed {filename}: All chunks were skipped due to missing/invalid metadata."

        chunks, chunk_embeddings, processed_metadatas, ids = final_chunks, final_embeddings, final_metadatas, final_ids
        # --- END filtering ---
        logger.debug(f"Processed metadatas for '{filename}' (first 2 entries): {processed_metadatas[:2]}", extra={"doc_filename_log": filename, "processed_metadatas_snippet": processed_metadatas[:2]})

        ids = [f"{extracted_metadata.get('source_file_name', filename)}::chunk_{i}" for i in range(len(chunks))]
        logger.info(f"Chunk IDs generated in source_file_name::chunk_i format for '{filename}'.")
        
        @retry(wait=wait_fixed(2), stop=stop_after_attempt(5), before_sleep=before_before_sleep_log(logger, logging.WARNING),
                       retry=retry_if_exception_type(Exception))
        def _add_chunks_to_chromadb(chunks_to_add, embeddings_to_add, metadatas_to_add, ids_to_add):
            if collection: # Ensure collection is available before trying to add
                collection.add(
                    documents=chunks_to_add,
                    embeddings=embeddings_to_add,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
            else:
                raise RuntimeError("ChromaDB collection is not initialized.") # Raise an error to trigger retry if needed
                
        _add_chunks_to_chromadb(chunks, chunk_embeddings, processed_metadatas, ids)

        return f"Processed {filename}: Successfully added {len(chunks)} chunks to ChromaDB. First chunk chars: {chunks[0][:50]}..."

    except Exception as e:
        logger.error(
            f"Processed {filename}: Error: Failed to process due to: {e}",
            extra={"doc_filename_log": filename, "error_detail": str(e)}
        )
        return f"Processed {filename}: Error: Failed to process due to: {e}\n{traceback.format_exc()}"


def ingest_initial_static_documents(static_docs_directory_path: str):
    """
    Performs initial ingestion of static documents from a specified directory.
    This function is called during application startup.
    """
    logger.info(f"Performing initial static document ingestion...")
    logger.info(f"Starting initial static data ingestion from processed directory: '{static_docs_directory_path}'")

    if not os.path.exists(static_docs_directory_path) or not os.listdir(static_docs_directory_path):
        logger.info(f"Static processed documents directory '{static_docs_directory_path}' is empty or does not exist. Skipping initial static ingestion.")
        return

    documents_to_process = []
    for root, _, files in os.walk(static_docs_directory_path): # Fixed: Use static_docs_directory_path here
        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            # Added .pdf and .docx
            if file_extension in [".txt", ".csv", ".json", ".pdf", ".docx"]:
                documents_to_process.append((file_path, filename, file_extension))
    
    if documents_to_process:
        batch_process_documents(documents_to_process, username="company_data", batch_size=5)
    else:
        logger.info("No processed documents found to ingest for initial static ingestion in this subdirectory.")

    logger.info(f"Initial static data ingestion from '{static_docs_directory_path}' complete.")

def batch_process_documents(documents_list: List[Tuple[str, str, str]], username: str, batch_size: int = 5):
    """
    Helper function to process documents in batches using a ThreadPoolExecutor.
    """
    total_docs = len(documents_list)
    for i in range(0, total_docs, batch_size):
        batch = documents_list[i : i + batch_size]
        logger.info(f"Starting batch processing of {len(batch)} documents (batch size: {batch_size})...\n")
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = []
            for doc_path, doc_filename, doc_extension in batch:
                futures.append(executor.submit(_process_single_document_for_ingestion, doc_path, doc_filename, doc_extension, username))
            
            for future in futures:
                try:
                    result_msg = future.result()
                    logger.info(result_msg)
                except Exception as e:
                    logger.error(f"ERROR during batch processing of a document in batching function: {e}", extra={"error_detail": str(e)})
                    traceback.print_exc()
        logger.info("Batch processing complete.")

# Error Handling Improvement - Health Check Endpoint
def system_health_check() -> Dict[str, Any]:
    """
    Provides a health check status of the RAG system components.
    """
    health_status = {
        "chromadb_ready": bool(collection),
        "embedding_model_available": embedding_model_status,
        "primary_reranker_loaded": bool(reranker_model and reranker_tokenizer),
        "secondary_reranker_loaded": bool(secondary_reranker_model and secondary_reranker_tokenizer), # Added secondary reranker status
        "initialization_errors": INITIALIZATION_ERRORS,
        "ollama_api_url": OLLAMA_API_URL,
        "ollama_embed_api_url": OLLAMA_EMBED_API_URL,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "primary_reranker_model_name": RERANKER_MODEL_NAME,
        "secondary_reranker_model_name": SECONDARY_RERANKER_MODEL_NAME, # Added secondary reranker name
        "chroma_db_path": CHROMA_DB_PATH
    }
    
    # Add more specific checks if needed
    if health_status["chromadb_ready"]:
        try:
            health_status["chromadb_collection_count"] = collection.count()
        except Exception as e:
            health_status["chromadb_collection_count"] = f"Error: {e}"
    
    if health_status["embedding_model_available"]:
        try:
            # Attempt a small embedding generation to confirm
            test_embedding = generate_ollama_embeddings(["test health check"])
            health_status["embedding_test_success"] = bool(test_embedding)
            health_status["embedding_test_dimension"] = len(test_embedding[0]) if test_embedding else 0
        except Exception as e:
            health_status["embedding_test_success"] = False
            health_status["embedding_test_error"] = str(e)

    logger.info(f"System Health Check: {health_status}")
    return health_status


# NEW DIAGNOSTIC FUNCTIONS

def get_document_content_by_filename(filename: str) -> Optional[str]:
    """
    Retrieves the raw text content of a document from ChromaDB given its filename.
    Returns None if the document is not found.
    """
    if not collection:
        logger.error("ChromaDB collection not initialized. Cannot retrieve document content.")
        return None
    
    try:
        # Query for documents with the specific filename in metadata
        results = collection.get(
            where={"source_file_name": filename},
            include=['documents']
        )
        
        if results and results['documents']:
            # Concatenate all chunks belonging to this file for the full content
            full_content = "\n\n".join(results['documents'])
            logger.info(f"Retrieved content for document '{filename}'. Length: {len(full_content)} chars.")
            return full_content
        else:
            logger.warning(f"No document found with filename: '{filename}' in ChromaDB.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving document content for '{filename}': {e}", extra={"filename_log": filename, "error_detail": str(e)})
        return None

def debug_retrieval_for_query(query: str, username: str, chat_history: List[Dict]) -> Dict[str, Any]:
    """
    Provides a detailed debug output of the RAG retrieval process for a given query.
    This function is for diagnostic purposes.
    """
    debug_info = {
        "query": query,
        "username": username,
        "chat_history_summary": [{"role": msg["role"], "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]} for msg in chat_history],
        "extracted_query_params": {},
        "extracted_history_params": {},
        "final_where_clause": {},
        "initial_retrieved_docs": [],
        "reranked_docs": [],
        "final_context_string_length": 0,
        "final_context_string_snippet": "",
        "source_filenames": [],
        "confidence_score": 0.0,
        "llm_raw_response": "",
        "llm_final_answer": "",
        "validation_status": False,
        "validation_reason": "",
        "retrieved_metadata": {},
    }

    start_retrieval_time = time.perf_counter()

    try:
        # Step 1: Extract attributes from current query (LLM-powered query analysis)
        query_attribute_extraction_prompt_messages = [
            {"role": "system", "content": LLM_PROMPT_TEMPLATES["query_attribute_extraction_system"] + \
                                          LLM_PROMPT_TEMPLATES["query_attribute_extraction_examples"]},
            {"role": "user", "content": f"User Query: {query}"}
        ]
        llm_query_analysis_response = ask_llm_model_phi3(query_attribute_extraction_prompt_messages)
        debug_info["llm_raw_response_query_analysis"] = llm_query_analysis_response[:500] + "..." if len(llm_query_analysis_response) > 500 else llm_query_analysis_response
        
        query_attribute_schema = {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_]+$": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ]
                }
            },
            "additionalProperties": True
        }
        extracted_query_params = safe_parse_llm_json_response(llm_query_analysis_response, schema=query_attribute_schema)
        
        # FIX for ERROR E: Ensure extracted_query_params is a dictionary.
        if isinstance(extracted_query_params, list) and len(extracted_query_params) == 1 and isinstance(extracted_query_params[0], dict):
            extracted_query_params = extracted_query_params[0]
            logger.info("Unwrapped single dictionary from LLM list response for query analysis.")
        elif not isinstance(extracted_query_params, dict):
            logger.warning(f"LLM returned an unexpected non-dictionary type ({type(extracted_query_params)}) for query attribute extraction. Initializing to empty dict.")
            extracted_query_params = {}

        debug_info["extracted_query_params"] = extracted_query_params

        # Query Analysis Enhancement - Add regex fallback if LLM extraction fails or returns empty
        if not extracted_query_params:
            logger.info("LLM attribute extraction returned empty for query. Attempting regex fallback for query analysis.")
            voltage_match = re.search(r"(\d+)\s*V(?:AC)?", query, re.IGNORECASE)
            if voltage_match:
                extracted_query_params["supply_voltage_vac"] = voltage_match.group(1)
            
            wattage_match = re.search(r"(\d+)\s*W", query, re.IGNORECASE)
            if wattage_match:
                extracted_query_params["total_power_consumption_w"] = wattage_match.group(1) + "W"
            
            ip_match = re.search(r"IP-(\d{2})", query, re.IGNORECASE)
            if ip_match:
                extracted_query_params["ip_rating"] = "IP" + ip_match.group(1)
            
            model_match = re.search(r"\b(PH-\d{2}-[A-Z]-\d{1,2}-[A-Z]{2,4})\b", query, re.IGNORECASE)
            if model_match:
                extracted_query_params["model_number"] = model_match.group(1)

            debug_info["extracted_query_params_regex_fallback"] = extracted_query_params
            logger.debug(f"Regex fallback extracted parameters: {extracted_query_params}")


        # Step 2: Extract persistent entities from chat history
        extracted_history_params = {}
        if chat_history:
            history_string_for_llm = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
            history_analysis_prompt = [
                {"role": "system", "content": LLM_PROMPT_TEMPLATES["chat_history_entity_extraction_system"]},
                {"role": "user", "content": f"Chat History:\n{history_string_for_llm}"}
            ]
            llm_history_analysis_response = ask_llm_model_phi3(history_analysis_prompt)
            debug_info["llm_raw_response_history_analysis"] = llm_history_analysis_response[:500] + "..." if len(llm_history_analysis_response) > 500 else llm_history_analysis_response
            
            extracted_history_params = extract_metadata_with_schema(llm_history_analysis_response, doc_type="history_entities", llm_call_fn=llm_call_fn)
            
            # FIX for ERROR E: Ensure extracted_history_params is a dictionary.
            if isinstance(extracted_history_params, list) and len(extracted_history_params) == 1 and isinstance(extracted_history_params[0], dict):
                extracted_history_params = extracted_history_params[0]
                logger.info("Unwrapped single dictionary from LLM list response for history analysis.")
            elif not isinstance(extracted_history_params, dict):
                logger.warning(f"LLM returned an unexpected non-dictionary type ({type(extracted_history_params)}) for history attribute extraction. Initializing to empty dict.")
                extracted_history_params = {}

            debug_info["extracted_history_params"] = extracted_history_params
            logger.debug(f"LLM-extracted history parameters: {extracted_history_params}")

        # Step 3: Merge query and history parameters for dynamic filtering
        merged_params = extracted_history_params.copy()
        
        # Keys where current query should strictly override history
        override_keys = ["product_name", "model_number", "total_power_consumption_w", "supply_voltage_vac", "ip_rating"]
        for key in override_keys:
            if key in extracted_query_params and extracted_query_params[key]:
                merged_params[key] = extracted_query_params[key]
        
        # For other keys, add from query if not already present or if query value is more specific/different
        for key, value in extracted_query_params.items():
            if key not in merged_params or (key not in override_keys and merged_params[key] != value):
                merged_params[key] = value
        
        logger.info(f"Merged parameters from query and chat history: {merged_params}")

        dynamic_filters = {}
        section_filter_values = []
        if isinstance(merged_params, dict):
            for key, value in merged_params.items():
                clean_key = re.sub(r'[^a-z0-9_]', '', key.lower())
                if clean_key in ["product_name", "model_number", "product_category", "ingress_protection",
                                 "total_power_consumption_w", "supply_voltage_vac", "power_factor", "thd",
                                 "operating_temperature_c", "working_humidity_rh", "surge_protection_kv",
                                 "led_luminous_efficacy_lm_w", "system_luminous_efficacy_lm_w",
                                 "system_luminous_flux_lumen", "cct_k", "cri", "beam_angle_deg",
                                 "ip_rating", "enclosure_material", "led_life_hours", "dimensions_lxwxh_mm",
                                 "frequency_hz", "attribute", "topic", "weight_kg", "color",
                                 "hr_policy_category", "leave_type", "safety_topic", "document_type",
                                 "policy_type", "tags"]:
                    if isinstance(value, list):
                        dynamic_filters[clean_key] = {"$or": [{"$eq": sanitize_text(v)} for v in value]}
                    else:
                        dynamic_filters[clean_key] = {"$eq": sanitize_text(str(value))}
                
                if clean_key == "section_type":
                    query_lower = query.lower()
                    if "features" in query_lower or "specs" in query_lower or "technical_specifications" in query_lower or "technical_highlights" in query_lower:
                        section_filter_values.extend(["features", "technical_specifications", "technical_highlights", "table", "table_row"])
                        section_filter_values = list(set(section_filter_values))
                    else:
                        if isinstance(value, list):
                            section_filter_values.extend([sanitize_text(v) for v in value])
                        else:
                            section_filter_values.append(sanitize_text(str(value)))
                        section_filter_values = list(set(section_filter_values))
        
        if any(kw in query.lower() for kw in ["spec", "technical", "parameter", "model", "voltage", "wattage"]):
            if "section_type" in dynamic_filters:
                current_section_types = dynamic_filters["section_type"].get("$in", [])
                if not isinstance(current_section_types, list):
                    current_section_types = [current_section_types]
                current_section_types.extend(["technical_specifications", "table"])
                dynamic_filters["section_type"] = {"$in": list(set(current_section_types))}
            else:
                dynamic_filters["section_type"] = {"$in": ["technical_specifications", "table"]}
            logger.info(f"Enforced technical_specifications filter due to query keywords: {dynamic_filters['section_type']}")
        elif section_filter_values:
            normalized_section_filter_values = []
            for s_type in section_filter_values:
                if s_type == "technical highlights":
                    normalized_section_filter_values.append("technical_specifications")
                else:
                    normalized_section_filter_values.append(s_type)
            normalized_section_filter_values = list(set(normalized_section_filter_values))

            dynamic_filters["section_type"] = {"$in": normalized_section_filter_values}
            logger.info(f"Applying section filter for retrieval: {dynamic_filters['section_type']}")

        debug_info["dynamic_filters"] = dynamic_filters
        logger.debug(f"LLM-extracted query parameters for dynamic filtering: {dynamic_filters}")

        base_where_clause = {"$or": [{"user": username}, {"user": "company_data"}]}

        final_where_clause_components = [base_where_clause]
        for k, v in dynamic_filters.items():
            final_where_clause_components.append({k: v})

        if len(final_where_clause_components) > 1:
            final_where_clause = {"$and": final_where_clause_components}
        else:
            final_where_clause = base_where_clause

        debug_info["final_where_clause"] = final_where_clause
        logger.info(f"Final ChromaDB query 'where' clause: {final_where_clause}")

        # Step 4: Perform hybrid search or vector search
        effective_num_results = NUM_RAG_RESULTS
        effective_similarity_thresh = SIMILARITY_THRESHOLD

        # Iterative Retrieval: Adjust parameters for fallback pass
        is_fallback_pass = False # This will be set to True if a fallback pass is triggered later
        
        if use_hybrid_search and BM25Okapi:
            initial_retrieved_docs = hybrid_search(query, username, n_results=effective_num_results, query_filters=final_where_clause)
        else:
            query_embedding_list = generate_ollama_embeddings([query])
            if not query_embedding_list:
                logger.error("Could not generate query embedding. Returning empty context.",
                             extra={"query_text_snippet": query[:50]})
                debug_info["retrieval_error"] = "Could not generate query embedding."
                return LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"], [], [], [], {}

            @retry(wait=wait_fixed(2), stop=stop_after_attempt(3), before_sleep=before_before_sleep_log(logger, logging.WARNING),
                   retry=retry_if_exception_type(Exception))
            def _query_chromadb():
                return collection.query(
                    query_embeddings=query_embedding_list,
                    n_results=effective_num_results,
                    where=final_where_clause,
                    include=['documents', 'embeddings', 'metadatas']
                )
            
            query_results = _query_chromadb()
            
            retrieved_doc_texts = query_results.get('documents', [[]])[0]
            retrieved_embeddings = query_results.get('embeddings', [[]])[0]
            retrieved_metadatas = query_results.get('metadatas', [[]])[0]

            if not retrieved_doc_texts:
                logger.info("No documents found in initial ChromaDB query.",
                            extra={"query_text_snippet": query[:50], "where_clause_log": final_where_clause})
                debug_info["initial_retrieved_docs"] = []
                debug_info["confidence_score"] = 0.0
                return LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"], [], [], [], {}

            query_embedding_vector = array(query_embedding_list[0])
            calculated_similarities = []
            for doc_emb_list_from_db, doc_meta in zip(retrieved_embeddings, retrieved_metadatas):
                doc_text = doc_meta.get("content", "N/A") # Ensure content is available if needed for logging
                try:
                    doc_embedding_vector = array(doc_emb_list_from_db)
                    norm_query = norm(query_embedding_vector)
                    norm_doc = norm(doc_embedding_vector)
                    
                    if norm_query == 0 or norm_doc == 0:
                        sim_score = 0.0
                    else:
                        sim_score = dot(query_embedding_vector, doc_embedding_vector) / (norm_query * norm_doc)
                    
                    calculated_similarities.append(sim_score)
                except Exception as e_sim:
                    logger.warning(f"Could not calculate similarity for document '{doc_text[:50]}...'. Error: {e_sim}. Defaulting to 0.",
                                   extra={"doc_snippet_log": doc_text[:50], "error_detail": str(e_sim)})
                    calculated_similarities.append(0.0)

            if calculated_similarities:
                sim_scores_array = np.array(calculated_similarities)
                mean_sim = np.mean(sim_scores_array)
                std_dev_sim = np.std(sim_scores_array)
                min_sim = np.min(sim_scores_array)
                max_sim = np.max(sim_scores_array)
                logger.info(f"Cosine similarity distribution: Mean={mean_sim:.4f}, StdDev={std_dev_sim:.4f}, Min={min_sim:.4f}, Max={max_sim:.4f}",
                            extra={"mean_similarity": mean_sim, "std_dev_similarity": std_dev_sim, "min_similarity": min_sim, "max_similarity": max_sim})
                
                logger.info(f"Using effective similarity threshold: {effective_similarity_thresh:.4f}",
                            extra={"effective_similarity_threshold_value": effective_similarity_thresh})
            else:
                logger.info(f"No similarities calculated, using default threshold: {effective_similarity_thresh:.4f}")


            initial_retrieved_docs = []
            for doc_text, sim_score, doc_meta in zip(retrieved_doc_texts, calculated_similarities, retrieved_metadatas):
                if sim_score >= effective_similarity_thresh:
                    initial_retrieved_docs.append({"content": doc_text, "score": sim_score, "metadata": doc_meta})
            
            if not initial_retrieved_docs:
                logger.warning(f"[RAG] No documents passed similarity threshold ({effective_similarity_thresh}). Retrying with relaxed filter...")
                fallback_retrieved = vector_store.similarity_search_with_score(query_text, k=effective_num_results)
            
                if fallback_retrieved:
                   logger.info(f"[RAG] Fallback retrieval succeeded with {len(fallback_retrieved)} docs.")
                   initial_retrieved_docs = [{"content": doc.page_content, "score": score, "metadata": doc.metadata}
                                  for doc, score in fallback_retrieved]
                else:
                    logger.error("[RAG] Fallback retrieval also failed — no documents found at all.")
                    debug_info["initial_retrieved_docs"] = []
                    debug_info["confidence_score"] = 0.0
                    return LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"], [], [], [], {}
            initial_retrieved_docs.sort(key=lambda item: item["score"], reverse=True)

        debug_info["initial_retrieved_docs"] = initial_retrieved_docs[:5] # Log top 5 for brevity


        # Step 5: Prepare embeddings for reranking and apply MMR (CRITICAL for UnboundLocalError)
        # Initialize all_retrieved_embeddings to an empty list here, guaranteeing it's always defined.
        all_retrieved_embeddings: List[List[float]] = [] 

        # Attempt to generate embeddings for all initial retrieved documents.
        # This list will be empty if initial_retrieved_docs is empty.
        try:
            if initial_retrieved_docs: # Only try to generate if there are documents
                all_retrieved_embeddings = generate_ollama_embeddings([doc["content"] for doc in initial_retrieved_docs])
                
                if not all_retrieved_embeddings:
                    logger.warning("`generate_ollama_embeddings` returned an empty list for initial retrieved documents. Reranking might be impacted.")
                    # If no embeddings generated, `all_retrieved_embeddings` correctly remains an empty list.
                    # No need for dummy embeddings if the list is already empty and consistent.
            else:
                logger.info("`initial_retrieved_docs` is empty. `all_retrieved_embeddings` will remain empty.")

        except Exception as e:
            logger.error(f"Failed to generate embeddings for initial retrieved documents due to: {e}. Reranking will proceed with empty embeddings.", exc_info=True)
            # In case of an exception, `all_retrieved_embeddings` remains an empty list, which is a safe fallback.

        final_docs_for_reranking = initial_retrieved_docs

        # Optional: Apply MMR
        # Ensure `maximal_marginal_relevance` is imported and available.
        if use_mmr and len(initial_retrieved_docs) > rerank_top_n and maximal_marginal_relevance:
            logger.info(f"Applying ChromaDB's built-in MMR for diversity (factor: {MMR_DIVERSITY_FACTOR})...",
                        extra={"mmr_diversity_factor_value": MMR_DIVERSITY_FACTOR})
            
            # `query_embedding` is already generated earlier in the function.
            # Ensure it's converted to a numpy array if `maximal_marginal_relevance` expects it.
            query_embedding_for_mmr = array(query_embedding) 

            # Only apply MMR if there are embeddings to work with
            if all_retrieved_embeddings:
                selected_indices = maximal_marginal_relevance(
                    query_embedding_for_mmr,
                    all_retrieved_embeddings,
                    k=rerank_top_n,
                    lambda_mult=MMR_DIVERSITY_FACTOR
                )
                temp_selected_docs = [initial_retrieved_docs[i] for i in selected_indices]
                
                if len(temp_selected_docs) > 0:
                    existing_types = {doc['metadata'].get('section_type') for doc in temp_selected_docs}
                    for doc in initial_retrieved_docs:
                        if doc not in temp_selected_docs and doc['metadata'].get('section_type') not in existing_types:
                            # Ensure 'score' key exists before modifying
                            if 'score' in doc:
                                doc['score'] *= 1.3 # Boost score for diversity
                                logger.debug(f"Boosted score for new category '{doc['metadata'].get('section_type')}' during MMR diversification.")
                            else:
                                logger.warning(f"Document missing 'score' key, skipping diversity boost for '{doc['metadata'].get('section_type')}'")
                    
                    final_docs_for_reranking = temp_selected_docs
                    logger.info(f"ChromaDB MMR selected {len(final_docs_for_reranking)} documents for reranking.")
                else:
                    logger.warning("MMR did not select any documents. Proceeding with initial retrieved documents for reranking.")
            else:
                logger.warning("`all_retrieved_embeddings` is empty, skipping MMR application.")

        # Always rerank. Pass all_retrieved_embeddings to rerank_documents.
        # Ensure `rerank_documents` function signature is:
        # `def rerank_documents(query: str, documents: List[Dict], all_retrieved_embeddings: List[List[float]]) -> List[Dict]:`
        final_reranked_docs = rerank_documents(query, final_docs_for_reranking, all_retrieved_embeddings)
        debug_info["reranked_docs"] = final_reranked_docs[:5] # Log top 5 for brevity


        # Step 6: Compute Confidence Score
        confidence = compute_confidence_score(final_reranked_docs)
        debug_info["confidence_score"] = confidence
        logger.info(f"Computed confidence score: {confidence:.4f}")
        
        final_docs_for_context = final_reranked_docs[:TOP_N_RERANKED_RESULTS]

        context_parts = []
        source_filenames = sorted(list(set(item["metadata"].get("source_file_name", "Unknown Source") for item in final_docs_for_context)))
        debug_info["source_filenames"] = source_filenames
        
        for d in final_docs_for_context:
            chunk_header = f"--- Document: {d['metadata'].get('source_file_name', 'Unknown Source')}"
            if d['metadata'].get('page'):
                chunk_header += f", Page: {d['metadata']['page']}"
            if d['metadata'].get('document_type'):
                chunk_header += f", Type: {d['metadata']['document_type']}"
            if d['metadata'].get('product_name'):
                chunk_header += f", Product: {d['metadata']['product_name']}"
            if d['metadata'].get('model_number'):
                chunk_header += f", Model: {d['metadata']['model_number']}"
            if d['metadata'].get('hr_policy_category'):
                chunk_header += f", HR Policy Category: {d['metadata']['hr_policy_category']}"
            if d['metadata'].get('section_type'):
                chunk_header += f", Section Type: {d['metadata']['section_type']}"
            if d['metadata'].get('source_doc_type'):
                chunk_header += f", Source Doc Type: {d['metadata']['source_doc_type']}"
            if d['metadata'].get('policy_type'):
                chunk_header += f", Policy Type: {d['metadata']['policy_type']}"
            if d['metadata'].get('tags'):
                chunk_header += f", Tags: {', '.join(d['metadata']['tags'])}"
            chunk_header += " ---\n"
            context_parts.append(chunk_header + d["content"])
        
        context_final_str = "\n\n".join(context_parts)
        debug_info["final_context_string_length"] = len(context_final_str)
        debug_info["final_context_string_snippet"] = context_final_str[:500] + "..." if len(context_final_str) > 500 else context_final_str

        retrieved_unique_metadata = defaultdict(set)
        for item in final_docs_for_context:
            for k, v in item["metadata"].items():
                if k not in ["filename", "user", "source_file_name", "processing_date", "chunk_order", "_embeddings", "_id", "_collection", "chunk_length", "chunk_hash"]:
                    sanitized_value = sanitize_text(str(v)).lower()
                    if sanitized_value:
                        retrieved_unique_metadata[k].add(sanitized_value)
        
        final_retrieved_metadata_for_response = {k: list(v) for k, v in retrieved_unique_metadata.items()}
        debug_info["retrieved_metadata"] = final_retrieved_metadata_for_response

        end_retrieval_time = time.perf_counter()
        retrieval_duration = (end_retrieval_time - start_retrieval_time) * 1000
        debug_info["retrieval_duration_ms"] = retrieval_duration
        logger.info(f"Final RAG context string generated (first 100 chars): '{context_final_str[:100]}...' in {retrieval_duration:.2f} ms",
                    extra={"retrieval_time_ms_log": retrieval_duration, "context_length_chars": len(context_final_str)})
        
        if not context_final_str.strip():
            debug_info["final_answer"] = LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"]
            return debug_info

        # Step 7: LLM Call
        llm_prompt_messages = build_llm_prompt(context_final_str, query, chat_history)
        raw_llm_response = ask_llm_model_phi3(llm_prompt_messages)
        debug_info["llm_raw_response"] = raw_llm_response[:500] + "..." if len(raw_llm_response) > 500 else raw_llm_response
        debug_info["llm_final_answer"] = raw_llm_response

        # Step 8: LLM Self-Correction/Validation
        logger.info("Performing LLM self-correction/validation on generated answer.")
        is_valid, validation_reason = validate_llm_answer(query, raw_llm_response, context_final_str)
        debug_info["validation_status"] = is_valid
        debug_info["validation_reason"] = validation_reason
        
        if not is_valid:
            logger.warning(f"LLM answer failed self-validation: {validation_reason}. Returning 'no information'.")
            debug_info["llm_final_answer"] = LLM_PROMPT_TEMPLATES["incremental_prompt_no_docs"]
            debug_info["final_source_docs_after_validation"] = []
            debug_info["final_source_filenames_after_validation"] = []
            debug_info["final_retrieved_metadata_after_validation"] = {}
        else:
            logger.info(f"LLM answer passed self-validation: {validation_reason}.")
            debug_info["final_source_docs_after_validation"] = final_docs_for_context # Keep the docs if answer is valid
            debug_info["final_source_filenames_after_validation"] = source_filenames
            debug_info["final_retrieved_metadata_after_validation"] = final_retrieved_metadata_for_response

    except ValueError as ve:
        error_msg = f"ValueError in debug_retrieval_for_query: {ve}. This often means a problem with embeddings or data shape."
        logger.error(f"ERROR: {error_msg}", extra={"error_detail": str(ve)})
        traceback.print_exc()
        debug_info["retrieval_error"] = error_msg
        debug_info["llm_final_answer"] = "An error occurred during retrieval. Please try again."
    except Exception as e:
        error_msg = f"An unexpected error occurred during debug_retrieval_for_query: {e}"
        logger.error(f"ERROR: {error_msg}", extra={"error_detail": str(e)})
        traceback.print_exc()
        debug_info["retrieval_error"] = error_msg
        debug_info["llm_final_answer"] = "An unexpected error occurred. Please try again."
    
    return debug_info

# Initialize the RAG system on startup
init_rag_system()

# Ingestion logic has been removed from rag.py. All ingestion must be performed via rag_ingest.py.

def remove_trailing_lists_and_non_kv(json_str: str) -> str:
    """
    Removes lines that are not key-value pairs and attempts to remove trailing lists from a JSON-like string.
    """
    lines = json_str.splitlines()
    filtered_lines = []
    inside_list = False
    for line in lines:
        # Detect start of a list value
        if re.match(r'\s*".*?"\s*:\s*\[', line):
            inside_list = True
            continue  # skip the line that starts a list
        if inside_list:
            if ']' in line:
                inside_list = False
            continue  # skip all lines inside the list
        # Only keep lines that look like key-value pairs or braces
        if ':' in line or line.strip() in ['{', '}']:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

# Production-grade, schema-driven metadata extraction for generalized documents

def extract_and_save_metadata(document_text, doc_type, filename, llm_call_fn):
    """
    Extracts metadata from document_text using a schema-dr
iven prompt and LLM, normalizes and filters it,
    and logs or saves the final metadata. Now robustly handles all LLM response types, including lists of dicts.
    """
    try:
        # Build prompt and extract metadata using LLM
        raw_metadata = extract_metadata_with_schema(document_text, doc_type, llm_call_fn)

        # --- Robust handling for LLM response types ---
        import logging
        logger = logging.getLogger("rag")
        # Unwrap single dict from list, or merge multiple dicts
        if isinstance(raw_metadata, list):
            if all(isinstance(item, dict) for item in raw_metadata):
                if len(raw_metadata) == 1:
                    raw_metadata = raw_metadata[0]
                    logger.info(f"Unwrapped single dictionary from LLM list response for {filename}.")
                else:
                    merged = {}
                    for d in raw_metadata:
                        merged.update(d)
                    raw_metadata = merged
                    logger.info(f"Merged {len(raw_metadata)} dicts from LLM list response for {filename}.")
            else:
                logger.warning(f"LLM metadata response for {filename} was a list but not all elements are dicts: {raw_metadata}")
                raw_metadata = {}
        elif not isinstance(raw_metadata, dict):
            logger.warning(f"LLM metadata response for {filename} was not a dict or list of dicts: {type(raw_metadata)} - {repr(raw_metadata)}")
            raw_metadata = {}

        # Normalize and filter metadata according to schema
        filtered_metadata = normalize_and_filter_metadata(raw_metadata, doc_type)
        logger.info(f"Final metadata for {filename}: {filtered_metadata}")
        # Save or use filtered_metadata only
        # db.save_metadata(filtered_metadata)  # Replace with your DB save logic
        return filtered_me
        tadata if isinstance(filtered_metadata, dict) and filtered_metadata else {}
    except Exception as e:
        logger.error(f"Metadata extraction failed for {filename}: {e}", exc_info=True)
        return {}

class EmbeddingCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite database for the cache."""
        try:
            with self.lock:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
                cursor = self.conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        text_hash TEXT PRIMARY KEY,
                        embedding_vector BLOB,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                self.conn.commit()
                logger.info(f"Initialized embedding cache at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing embedding cache database: {e}")
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            raise
    
    def save(self):
        """Save any pending changes to the database."""
        with self.lock:
            if not self.conn:
                self._init_db()
                return
                
            try:
                self.conn.commit()
                logger.debug("Successfully committed changes to embedding cache.")
            except (sqlite3.ProgrammingError, sqlite3.InterfaceError) as e:
                if "Cannot operate on a closed database" in str(e) or "closed database" in str(e).lower():
                    logger.warning("Attempted to save to closed embedding cache, reinitializing...")
                    try:
                        self.conn.close()
                    except:
                        pass
                    self._init_db()
                    self.conn.commit()
                else:
                    logger.error(f"Error saving embedding cache: {e}")
                    raise
    
    def close(self):
        """Close the database connection safely."""
        with self.lock:
            if self.conn:
                try:
                    self.save()  # Ensure all changes are committed
                    self.conn.close()
                    logger.info("Closed embedding cache database connection.")
                except Exception as e:
                    logger.error(f"Error closing embedding cache: {e}")
                finally:
                    self.conn = None


class RerankerCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database for the reranker cache."""
        try:
            with self.lock:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
                cursor = self.conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS reranker_scores (
                        query_hash TEXT NOT NULL,
                        document_hash TEXT NOT NULL,
                        score REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (query_hash, document_hash)
                    )
                ''')
                self.conn.commit()
                logger.info(f"Initialized reranker cache at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing reranker cache database: {e}")
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            raise
    
    def save(self):
        """Save any pending changes to the database."""
        with self.lock:
            if not self.conn:
                self._init_db()
                return
                
            try:
                self.conn.commit()
                logger.debug("Successfully committed changes to reranker cache.")
            except (sqlite3.ProgrammingError, sqlite3.InterfaceError) as e:
                if "Cannot operate on a closed database" in str(e) or "closed database" in str(e).lower():
                    logger.warning("Attempted to save to closed reranker cache, reinitializing...")
                    try:
                        self.conn.close()
                    except:
                        pass
                    self._init_db()
                    self.conn.commit()
                else:
                    logger.error(f"Error saving reranker cache: {e}")
                    raise
    
    def close(self):
        """Close the database connection safely."""
        with self.lock:
            if self.conn:
                try:
                    self.save()  # Ensure all changes are committed
                    self.conn.close()
                    logger.info("Closed reranker cache database connection.")
                except Exception as e:
                    logger.error(f"Error closing reranker cache: {e}")
                finally:
                    self.conn = None

def save_caches():
    try:
        if 'embedding_cache' in globals() and hasattr(embedding_cache, 'save'):
            embedding_cache.save()
            logger.info("Successfully saved embedding cache.")
        if 'reranker_cache' in globals() and hasattr(reranker_cache, 'save'):
            reranker_cache.save()
            logger.info("Successfully saved reranker cache.")
    except Exception as e:
        logger.error(f"Error saving caches: {e}", exc_info=True)

def close_caches():
    """Safely close all cache connections with proper error handling."""
    # Declare globals at the beginning
    global embedding_cache, reranker_cache
    
    # Close embedding cache if it exists
    if 'embedding_cache' in globals() and embedding_cache is not None:
        try:
            if hasattr(embedding_cache, 'close'):
                embedding_cache.close()
                logger.info("Successfully closed embedding cache.")
        except Exception as e:
            logger.error(f"Error closing embedding cache: {e}", exc_info=True)
    
    # Close reranker cache if it exists
    if 'reranker_cache' in globals() and reranker_cache is not None:
        try:
            if hasattr(reranker_cache, 'close'):
                reranker_cache.close()
                logger.info("Successfully closed reranker cache.")
        except Exception as e:
            logger.error(f"Error closing reranker cache: {e}", exc_info=True)
    
    # Clear global references
    embedding_cache = None
    reranker_cache = None

# Register the save function to run at exit
# atexit.register(save_caches)  # Commented out: handled explicitly in rag_ingest.py

# Import JSON utilities after cache setup to avoid circular imports
from rag_llm_json_utils import _clean_json_string, extract_kv_pairs, flatten_json, safe_parse_llm_json_response, filter_noisy_metadata, extract_metadata_with_schema, normalize_and_filter_metadata
