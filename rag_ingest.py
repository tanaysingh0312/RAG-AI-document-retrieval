# In rag_ingest.py

import os
import re
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import uuid
from uuid import uuid4
from typing import List, Dict, Optional, Any, Callable
import sqlite3
import logging
import requests
from requests.exceptions import RequestException
from tenacity import retry, wait_fixed, stop_after_attempt, before_sleep_log, retry_if_exception_type
import threading # Import threading for thread-local storage

# Setup basic logging at the very beginning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONSTANTS ---
MIN_ALPHANUM_RATIO = 0.05

# Ensure these imports are correct and available from rag.py
# Import necessary components
from rag import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    _process_json_for_html_tables,
    close_caches # Keep close_caches for atexit
)

# --- Define EmbeddingCache and RerankerCache locally ---
class EmbeddingCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local() # Thread-local storage for connection and cursor
        self._initialize_db_schema()
        logger.info(f"EmbeddingCache initialized (schema checked) at {db_path}")

    def _initialize_db_schema(self):
        # Create table if it doesn't exist using a temporary connection
        # This ensures the schema is set up, but doesn't hold a global connection.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # IMPORTANT: Drop table if it exists to ensure fresh schema
        cursor.execute('''DROP TABLE IF EXISTS embeddings''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                text_content TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"EmbeddingCache schema ensured for {self.db_path}")


    def _get_db_connection(self):
        # Get or create a thread-local connection and cursor
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.cursor = self._local.connection.cursor()
            logger.debug(f"Created new SQLite connection for thread {threading.get_ident()}")
        return self._local.connection, self._local.cursor

    def get(self, text: str) -> Optional[List[float]]:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        conn, cursor = self._get_db_connection()
        cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
        result = cursor.fetchone()
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode cached embedding for hash {text_hash}: {e}")
                return None
        return None

    def set(self, text: str, embedding: List[float]):
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        conn, cursor = self._get_db_connection()
        embedding_json = json.dumps(embedding)
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, text_content, embedding) VALUES (?, ?, ?)",
                (text_hash, text, embedding_json)
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save embedding for hash {text_hash} to cache: {e}")

    def save(self):
        # This method will commit the current thread's connection.
        # It's primarily for ensuring writes are flushed before a thread exits.
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.commit()
            logger.debug(f"Committed thread-local EmbeddingCache connection for thread {threading.get_ident()}")

    # Removed the 'close' method as thread-local connections are typically managed by thread lifecycle.
    # Global cleanup should be handled by atexit.

class RerankerCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local() # Thread-local storage for connection and cursor
        self._initialize_db_schema()
        logger.info(f"RerankerCache initialized (schema checked) at {db_path}")

    def _initialize_db_schema(self):
        # Create table if it doesn't exist using a temporary connection
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # IMPORTANT: Drop table if it exists to ensure fresh schema
        cursor.execute('''DROP TABLE IF EXISTS rerankings''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rerankings (
                query_doc_hash TEXT PRIMARY KEY,
                reranked_score REAL
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"RerankerCache schema ensured for {self.db_path}")

    def _get_db_connection(self):
        # Get or create a thread-local connection and cursor
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.cursor = self._local.connection.cursor()
            logger.debug(f"Created new SQLite connection for thread {threading.get_ident()}")
        return self._local.connection, self._local.cursor

    def get(self, query_doc_hash: str) -> Optional[float]:
        conn, cursor = self._get_db_connection()
        cursor.execute("SELECT reranked_score FROM rerankings WHERE query_doc_hash = ?", (query_doc_hash,))
        result = cursor.fetchone()
        return result[0] if result else None

    def set(self, query_doc_hash: str, reranked_score: float):
        conn, cursor = self._get_db_connection()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO rerankings (query_doc_hash, reranked_score) VALUES (?, ?)",
                (query_doc_hash, reranked_score)
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save reranking for hash {query_doc_hash} to cache: {e}")

    def save(self):
        # This method will commit the current thread's connection.
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.commit()
            logger.debug(f"Committed thread-local RerankerCache connection for thread {threading.get_ident()}")

    # Removed the 'close' method as thread-local connections are typically managed by thread lifecycle.


# Initialize the embedding cache with a path in the same directory as the ChromaDB
chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(chroma_dir, exist_ok=True)

# Initialize caches
embedding_cache = EmbeddingCache(os.path.join(chroma_dir, "embeddings_cache.db"))
reranker_cache = RerankerCache(os.path.join(chroma_dir, "reranker_cache.db"))

# Import extract_metadata_with_schema, flatten_json, filter_noisy_metadata, and safe_parse_llm_json_response
from rag_llm_json_utils import extract_metadata_with_schema, flatten_json, filter_noisy_metadata, safe_parse_llm_json_response


def is_markdown_table_doc(file_name: str) -> bool:
    """Checks if the filename indicates a structured markdown table document."""
    return "markdown_tables" in file_name.lower()


# --- Deterministic Metadata Extraction Utilities ---

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # graceful fallback if bs4 not installed

def extract_product_name_and_power_from_filename(filename: str):
    """
    Extracts product name and power consumption (wattage) from a filename such as '1. 7-20W HAND LAMP.txt'.
    Returns (product_name, power_consumption) as strings.
    """
    # Remove extension and hash tail
    name = os.path.splitext(filename)[0]
    name = name.split('_1_')[0]
    # Remove leading index (e.g., '1. ')
    name = re.sub(r'^\s*\d+\.\s*', '', name)
    # Extract power consumption (e.g., '7-20W', '12W', etc.)
    power_match = re.search(r'(\d+[\-–]?\d*)\s*[wW]', name)
    power_consumption = power_match.group(0).strip() if power_match else None
    # Remove power part from the name
    name_wo_power = re.sub(r'(\d+[\-–]?\d*)\s*[wW]', '', name)
    # Remove extra spaces and numbers at start
    product_name = re.sub(r'^\s*[-–]*\s*', '', name_wo_power).strip()
    # Remove any remaining leading numbers/dots
    product_name = re.sub(r'^\s*\d+\s*', '', product_name).strip()
    # Remove extra spaces and punctuation
    product_name = re.sub(r'[_\-]+', ' ', product_name).strip()
    # Capitalize nicely for metadata
    product_name = product_name.title()
    return product_name, power_consumption

# Backward compatibility
extract_product_name_from_filename = lambda filename: extract_product_name_and_power_from_filename(filename)[0]


def extract_table_fields(chunk_text: str) -> dict:
    """
    Extract key-value pairs from markdown and HTML tables.
    Normalizes keys (lowercase, underscores, strips parens/hyphens), and applies alias mapping.
    """
    fields = {}
    # Key normalization and alias mapping
    def normalize_key(key):
        key = key.lower()
        key = re.sub(r'[\s\-]+', '_', key)
        key = re.sub(r'[()\[\]]', '', key)
        key = key.strip('_')
        return key
    # Aliases for schema consistency
    alias_map = {
        'product_category': 'product_name',
        'product': 'product_name',
        'model': 'model_number',
        'power': 'power_consumption',
        'power_consumption_w': 'power_consumption',
        'wattage': 'power_consumption',
        'input_voltage': 'input_voltage',
        'ip_rating': 'ip_rating',
        'section': 'section_type',
        'section_type': 'section_type',
    }
    # Markdown parsing
    for line in chunk_text.splitlines():
        match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if match:
            raw_key = match.group(1).strip()
            val = match.group(2).strip()
            norm_key = normalize_key(raw_key)
            aliased_key = alias_map.get(norm_key, norm_key)
            fields[aliased_key] = val
    # HTML parsing fallback
    if BeautifulSoup and "<table" in chunk_text.lower():
        try:
            soup = BeautifulSoup(chunk_text, "html.parser")
            rows = soup.find_all("tr")
            for row in rows:
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    raw_key = cols[0].get_text(strip=True)
                    val = cols[1].get_text(strip=True)
                    norm_key = normalize_key(raw_key)
                    aliased_key = alias_map.get(norm_key, norm_key)
                    if aliased_key and val:
                        fields[aliased_key] = val
        except Exception:
            pass  # skip HTML parsing if bs4 fails
    return fields

def enrich_metadata(chunk_text: str, source_file_name: str) -> dict:
    """Generate structured metadata for a chunk."""
    product_name, power_consumption = extract_product_name_and_power_from_filename(source_file_name)
    fields = extract_table_fields(chunk_text)
    metadata = {
        "product_name": product_name,
        "power_consumption": power_consumption,
        "section_type": "table" if fields else "general",
        "table_fields": fields
    }
    # Always include keys, even if missing
    if "product_name" not in fields:
        fields["product_name"] = product_name
    if "power_consumption" not in fields:
        fields["power_consumption"] = power_consumption
    if "ordering_code" not in fields:
        fields["ordering_code"] = None
    return metadata


# For self-containment in rag_ingest.py, let's define it here.
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain.embeddings import OllamaEmbeddings

class ChromaCompatibleOllamaEmbeddings(OllamaEmbeddings):
    def name(self) -> str:
        return "ollama_embeddings"

    def __call__(self, input):
        return self.embed_documents(input)
    
    def embed_documents(self, texts):
        global embedding_cache  # Access the global cache
        embeddings = []
        for text in texts:
            if embedding_cache:
                # Check if embedding exists in cache
                cached_embedding = embedding_cache.get(text)
                if cached_embedding is not None:
                    embedding = cached_embedding
                    logger.debug(f"Cache used for embedding: {hashlib.md5(text.encode('utf-8')).hexdigest()}")
                else:
                    # If not in cache, compute and store
                    embedding = super(ChromaCompatibleOllamaEmbeddings, self).embed_documents([text])[0]
                    embedding_cache.set(text, embedding)
                    logger.debug(f"Cache computed and stored for: {hashlib.md5(text.encode('utf-8')).hexdigest()}")
            else:
                embedding = super().embed_documents([text])[0]
                logger.debug(f"No cache available, embedding computed for: {hashlib.md5(text.encode('utf-8')).hexdigest()}")
            embeddings.append(embedding)
        return embeddings


# Unstructured imports (still useful for element types in metadata inference, but not for parsing PDFs directly here)
try:
    from unstructured.documents.elements import Element, Text, NarrativeText, Title, ListItem, Table
    # Import partition_auto directly for file paths, and partition_text for raw text
    from unstructured.partition.auto import partition as partition_auto_file
    from unstructured.partition.text import partition_text # Correct import for text content
    from unstructured.partition.csv import partition_csv # Assuming this is used for CSVs
    UNSTRUCTURED_AVAILABLE = True
except (ImportError, RuntimeError, AttributeError) as e:
    logging.warning(f"Unstructured library not fully available: {e}. Some partitioning features may be limited.")
    UNSTRUCTURED_AVAILABLE = False
    class Element: pass
    class Text:
        def __init__(self, text): self.text = text
        def __str__(self): return self.text
    class NarrativeText(Text): pass
    class Title(Text): pass
    class ListItem(Text): pass
    class Table(Text): pass
    # Fallback dummy functions if Unstructured is not available
    def partition_auto_file(filename): return [Text("Dummy content from " + filename)]
    def partition_text(text): return [Text(text)] # Fallback for partition_text
    def partition_csv(filename): return [Text("Dummy CSV content from " + filename)]


# Semantic Chunker imports
try:
    from semantic_chunker import semantic_chunk_unstructured, semantic_chunk_plain_text
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    logging.error("Error: semantic_chunker.py not found or functions not defined. Please ensure it's in the same directory.")
    SEMANTIC_CHUNKER_AVAILABLE = False
    # Fallback dummy functions if semantic_chunker is not available
    def semantic_chunk_unstructured(elements, max_chars, overlap): return [Text(text=str(e)) for e in elements]
    def semantic_chunk_plain_text(text, max_chars, overlap): return [Text(text=text)]


# --- Configuration ---
# IMPORTANT: These are now placeholders. The actual paths will be passed from __main__
TEXT_FILES_FOLDER = ""
TABLE_FILES_FOLDER = ""

MAX_CHUNK_CHARS = 4000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100

# OLLAMA API URL for LLM calls (e.g., for metadata extraction)
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', "http://localhost:11434/v1/chat/completions")
# Use a potentially different, smaller model for metadata extraction if needed
OLLAMA_METADATA_MODEL_NAME = os.getenv('OLLAMA_METADATA_MODEL_NAME', "phi3") # Changed default to phi3
logger.info(f"Ollama Metadata Extraction Model: {OLLAMA_METADATA_MODEL_NAME}")


# --- Cache Availability Flag ---
class EmbeddingCache:
    # ... (existing methods)
    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
            logger.info("EmbeddingCache closed.")

class RerankerCache:
    # ... (existing methods)
    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
            logger.info("RerankerCache closed.")

# This flag is now derived from whether embedding_cache and reranker_cache were successfully imported
CACHE_AVAILABLE = 'embedding_cache' in globals() and embedding_cache is not None

# --- Noise Filtering Function (moved directly into this file for self-containment) ---
def calculate_alphanum_ratio(text):
    if not text:
        return 0.0
    alphanum_count = sum(c.isalnum() for c in text)
    return alphanum_count / len(text)

def contains_critical_phrase(text: str) -> bool:
    """
    Checks if a chunk contains phrases that indicate it's likely important
    even if short or has low alphanum ratio (e.g., tables, key-value pairs).
    """
    critical_patterns = [
        re.compile(r'\b(voltage|wattage|power|current|dimensions|weight|ip rating|lumens|cct|cri|model|series|features|specifications|technical|guarantee|warranty|application|ordering code)\b', re.IGNORECASE),
        re.compile(r'\|.*\|', re.MULTILINE), # Markdown table row
        re.compile(r':\s*\d'), # Key-value pair with a number
        re.compile(r'^\s*[\-\*\•]\s*\w+', re.MULTILINE), # List item
        re.compile(r'\d{2,}\s*(mm|cm|m|kg|g|W|V|A|lm|K|hrs)\b', re.IGNORECASE) # Numbers with units
    ]
    for pattern in critical_patterns:
        if pattern.search(text): # Use .search() and the compiled pattern
            return True
    return False

# filter_noisy_metadata is now imported from rag_llm_json_utils


# --- ChromaDB Setup ---
client = None
collection = None 
embedding_function = None # Initialize to None

try:
    # Use the imported ChromaCompatibleOllamaEmbeddings
    embedding_function = ChromaCompatibleOllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    logger.info(f"Initialized OllamaEmbeddings (via wrapper) with model: {EMBEDDING_MODEL_NAME}")
    
    from chromadb import PersistentClient # Import here to ensure it's available
    client = PersistentClient(path=CHROMA_DB_PATH)
    
    # Attempt to delete the collection if it exists to ensure a clean slate
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Existing collection '{COLLECTION_NAME}' deleted successfully.")
    except Exception as e:
        logger.warning(f"Could not delete existing collection '{COLLECTION_NAME}' (might not exist or permissions issue): {e}")
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    logger.info(f"Connected to ChromaDB at '{CHROMA_DB_PATH}', collection: '{COLLECTION_NAME}'")
except ImportError as ie:
    logger.error(f"Failed to import ChromaDB or OllamaEmbeddings: {ie}. Ensure 'chromadb' and 'langchain-community' are installed.")
    collection = None
    embedding_function = None
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB or OllamaEmbeddings: {e}. Please ensure Ollama is running and model '{EMBEDDING_MODEL_NAME}' is pulled. Ingestion to ChromaDB will be skipped.", exc_info=True)
    collection = None
    embedding_function = None


# --- Metadata Extraction Helper Functions (Retained from previous version, adjusted for text/json input) ---
# (Patched: deterministic enrich_metadata now used for every chunk)

def normalize_metadata_fields(metadata: dict) -> dict:
    """
    Normalizes key metadata fields (e.g., product_name, document_type, section_type) for consistent retrieval.
    Lowercases and trims whitespace for key string fields.
    """
    for key in ["product_name", "document_type", "section_type"]:
        if key in metadata and isinstance(metadata[key], str):
            metadata[key] = metadata[key].strip().lower()
    return metadata

def infer_product_name(chunk_text: str, original_file_name: str, elements: Optional[List[Element]] = None) -> Optional[str]:
    """
    Infers the product name from chunk text, original file name.
    The 'elements' parameter is kept for consistency if you ever pass Unstructured elements.
    """
    lower_chunk = chunk_text.lower()
    base_name = original_file_name.lower() # Use original_file_name for better inference

    # 1. From original filename (common pattern: "Product_Name_Version.pdf" -> "Product_Name_Version.txt")
    # Expanded filename_product_map based on provided HTML files and common product names
    filename_product_map = {
        "led_flatpanel_downlight": "LED Flat Panel Down Light",
        "led_flood_light": "LED Flood Light",
        "led_highbay_light": "LED Highbay Light",
        "led_non_flameproof_wellglass": "LED Non Flameproof Wellglass",
        "led_solar_street_light": "LED Solar Street Light",
        "led_street_light_earth_series": "LED Street Light Earth Series",
        "led_tubelights": "LED Tubelight",
        "led_surface_mounted_downlight": "LED Surface Mounted Downlight",
        "led_cfl_retrofit": "LED CFL Retrofit",
        "recessed_mounted_clean_room": "Recessed Mounted Clean Room Light",
        "led_emergency_light": "LED Emergency Light",
        "sensor_based_tubelight": "Sensor Based Tube Light",
        "recessed_mounted_downlight": "Recessed Mounted Downlight",
        "led_garden_pathway_bollard": "LED Garden & Pathway Bollard",
        "led_inground_burial": "LED Inground Burial Light",
        "led_underwater_nozzle_light": "LED Underwater/Nozzle Light",
        "led_spot_light": "LED Spot Light",
        "led_garden_light": "LED Garden Light",
        "led_canopy_light": "LED Canopy Light",
        "post_top_lantern": "Post Top Lantern",
        "vertical_solar_pole": "Vertical Solar Pole",
        "led_street_lights_venus_series": "LED Street Light Venus Series",
        "led_flood_light_eco_series": "LED Flood Light Eco Series",
        "led_solar_light_all_in_one": "LED Solar Street Light All in One",
        "led_decorative_post_of_lantern": "LED Decorative Post of Lantern",
        "led_traffic_light": "LED Traffic Light",
        "led_aviation_light": "LED Aviation Light",
        "bulk_head_light": "Bulk Head Light",
        "led_flameproof_wellglass": "LED Flameproof Wellglass",
        "led_hand_lamp": "LED Hand Lamp",
        "led_pit_light": "LED Pit Light",
        "led_helmet_light": "LED Helmet Light",
        "led_24vac_products": "24V AC LED Product",
        "led_triproof_light": "LED Triproof Light",
        "led_flameproof_tube_light": "LED Flameproof Tube Light",
        "blood_bag_tube_sealer": "Blood Bag Tube Sealer",
        "deep_freezer_-40°c": "Deep Freezer -40°C",
        "blood_collection_monitor": "Blood Collection Monitor",
        "endoscope": "Endoscope",
        "ultra_deep_freezer_-80°c": "Ultra Deep Freezer -80°C",
        "double_pan_balance": "Double Pan Balance",
        "x_ray_view_box": "X-Ray View Box",
        "operation_theater_light": "Operation Theater Light",
        "blood_bank_referigeator": "Blood Bank Refrigerator",
        "bldc_ceiling_fan": "BLDC Ceiling Fan",
        "hvls_fan": "HVLS Fan",
        "bldc_table_fan": "BLDC Table Fan",
        "bldc_pedestal_fan": "BLDC Pedestal Fan",
        "ccms": "Centrally Controlled Monitoring System",
        "internet_of_things": "Internet of Things",
        "constant_current_ac-dc_led_driver": "Constant Current AC-DC LED Driver",
        "dimmable_solutions": "Dimmable Solution",
        "wifi_smart_power_socket_plug": "WiFi Smart Power Socket/Plug",
        "pir_occupancy_sensors": "PIR Occupancy Sensor",
        "hand_held_sanitizer_cable_based": "Hand Held Sanitizer (Cable Based)",
        "hand_held_sanitizer": "Hand Held Sanitizer",
        "shoe_sanitizer": "Shoe Sanitizer",
        "self_check_kiosk": "Self Check Kiosk",
        "smart_facial_body_temperature_display": "Smart Facial Body Temperature Display",
        "infrared_forehead_thermometer": "Infrared Forehead Thermometer",
        "laminar_air_flow": "Laminar Air Flow",
        "pcr_workstation": "PCR Workstation",
        "corona_disinfection_box": "Corona Disinfection Box",
        "uv_tower": "DRDO Approved UV Tower",
        "sanitizer_bottle_dispenser": "Automatic Outdoor Sanitizer Bottle Dispenser",
        "currency_sanitizer": "Currency Sanitizer",
        "blood_component_extractor_machine": "Automated Blood Component Extractor Machine",
        # Existing generic entries, keep them for broader matching
        "pyro-hb": "high bay light",
        "pyro-fl": "flood light",
        "pyro-cdv": "corona disinfection box",
        "pyrotech x300": "pyrotech x300 robot vacuum cleaner",
        "srs-xb41": "srs-xb41 portable bluetooth speaker",
    }
    for key, product in filename_product_map.items():
        if key in base_name:
            return product

    # 2. From chunk content (using regex or keyword search)
    # Refined product patterns to better capture names from H4 tags and general text
    product_patterns = [
        r"<h4>(.*?)</h4>", # Capture content within h4 tags, common for product titles
        r"product name[:\s]*([^\n,;]+)",
        r"model number[:\s]*([^\n,;]+)",
        r"product[:\s]*([^\n,;]+)",
        r"series[:\s]*([^\n,;]+)",
        r"pyrotech\s+([a-zA-Z0-9\s-]+?)(?:\s+series|\s+model|\s+light|\s+lamp|\s+box|\s+cleaner|\s+socket|\s+pole|\s+unit|\s+fan|\s+driver|\s+sensor|\s+kiosk|\s+thermometer|\s+workstation|\s+tower|\s+dispenser|\s+machine|)", # Expanded product types
    ]
    for pattern in product_patterns:
        match = re.search(pattern, lower_chunk, re.IGNORECASE)
        if match:
            extracted_name = match.group(1).strip()
            if extracted_name and extracted_name.lower() not in ["product", "model", "series", "unit", "light", "lamp", "box", "cleaner", "socket", "pole", "fan", "driver", "sensor", "kiosk", "thermometer", "workstation", "tower", "dispenser", "machine"]:
                return extracted_name
    logger.debug(f"Could not infer specific product_name for chunk from {original_file_name}: '{lower_chunk[:100]}...'")
    
    # Fallback to generic product type if common product keywords are found
    if "light" in lower_chunk or "lamp" in lower_chunk or "box" in lower_chunk or "socket" in lower_chunk or \
       "pole" in lower_chunk or "cleaner" in lower_chunk or "fan" in lower_chunk or "driver" in lower_chunk or \
       "sensor" in lower_chunk or "kiosk" in lower_chunk or "thermometer" in lower_chunk or \
       "workstation" in lower_chunk or "dispenser" in lower_chunk or "machine" in lower_chunk:
        return "generic product" # More general fallback
    return "unknown_product"


def infer_document_type(chunk_text: str, original_file_name: str, elements: Optional[List[Element]] = None) -> str:
    """
    Infers the document type (e.g., 'Product Datasheet', 'User Manual', 'HR Policy', 'Report').
    Uses original_file_name for inference.
    """
    lower_chunk = chunk_text.lower()
    base_name = original_file_name.lower()

    # From original filename
    if "datasheet" in base_name or "specifications" in base_name or "tech_spec" in base_name or "data_sheet" in base_name:
        return "Product Datasheet"
    if "manual" in base_name or "guide" in base_name or "instructions" in base_name:
        return "User Manual"
    if "policy" in base_name or "handbook" in base_name or "hr" in base_name:
        return "HR Policy"
    if "report" in base_name or "analysis" in base_name or "summary" in base_name:
        return "Report"
    if "brochure" in base_name:
        return "Brochure"
    if "html" in base_name and any(p in lower_chunk for p in ["product", "light", "fan", "medical", "covid"]):
        return "Product Page" # New document type for HTML product pages

    # From chunk content keywords
    if re.search(r"\b(technical\s+specifications|data\s+sheet|product\s+overview|performance\s+data)\b", lower_chunk):
        return "Product Datasheet"
    if re.search(r"\b(user\s+manual|installation\s+guide|troubleshooting)\b", lower_chunk):
        return "User Manual"
    if re.search(r"\b(hr\s+policy|leave\s+policy|code\s+of\s+conduct|employee\s+handbook)\b", lower_chunk):
        return "HR Policy"
    if re.search(r"\b(financial\s+report|annual\s+report|market\s+analysis)\b", lower_chunk):
        return "Report"

    return "General Document" # Default if not classified


def extract_product_name(text: str, file_name: str = "") -> str:
    match = re.search(r'((?:[A-Z][a-zA-Z]+\s){1,5})(Light|Lamp|LED|Fixture)', text)
    if match:
        return (match.group(1) + match.group(2)).strip().lower()
    # Fallback: try file name
    file_match = re.search(r'([a-zA-Z\s]+)(light|lamp|led|fixture)', file_name, re.I)
    if file_match:
        return (file_match.group(1) + file_match.group(2)).strip().lower()
    return "unknown"

def detect_section_type(text: str) -> str:
    if re.search(r"\|\s*Parameter\s*\|\s*Value\s*\|", text) or re.findall(r"\|.*\|", text):
        return "table"
    elif re.search(r"Voltage|Power|Lumens|Mounting|Input Voltage|Ordering Code", text, re.I):
        return "technical_specifications"
    else:
         return "product description"

def extract_table_fields(chunk_text: str) -> dict:
    """
    Extracts key-value pairs from markdown or HTML table text.
    Returns a dictionary of parameters and their values. Always includes 'product_name' and 'ordering_code' (set to None if missing).
    No hallucinated values: only what is present in the table or chunk.
    """
    fields = {}
    # Markdown table rows
    for line in chunk_text.splitlines():
        match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if match:
            key = match.group(1).strip().lower().replace(' ', '_')
            value = match.group(2).strip()
            if key:
                fields[key] = value if value else None
    # HTML table parsing
    if '<table' in chunk_text.lower():
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(chunk_text, 'html.parser')
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True).lower().replace(' ', '_')
                        value = cells[1].get_text(strip=True)
                        if key:
                            fields[key] = value if value else None
        except Exception as e:
            # If bs4 is not installed or HTML is malformed, skip HTML parsing
            pass
    # Always include product_name and ordering_code (None if missing)
    if 'product_name' not in fields:
        fields['product_name'] = None
    if 'ordering_code' not in fields:
        fields['ordering_code'] = None
    return fields

def generate_metadata(chunk_text: str, source_file_name: str) -> dict:
    """
    Generates metadata for a chunk. Ensures unique chunk_id, robust chunk_hash, and normalized source_file_name.
    """
    import hashlib
    import os
    product = extract_product_name(chunk_text, source_file_name)
    section = detect_section_type(chunk_text)
    table_fields = extract_table_fields(chunk_text)
    # Always ensure product_name and ordering_code in table_fields
    if not table_fields.get('product_name'):
        table_fields['product_name'] = product or "unknown"
    if 'ordering_code' not in table_fields:
        table_fields['ordering_code'] = None
    normalized_source = os.path.basename(source_file_name).replace(" ", "_")
    metadata = {
        "chunk_id": str(uuid.uuid4()),
        "chunk_hash": hashlib.md5(chunk_text.encode("utf-8")).hexdigest(),
        "section_type": section,
        "product_name": table_fields['product_name'],
        "ordering_code": table_fields['ordering_code'],
        "source_file_name": normalized_source,
        "user": "company_data",
        "original_element_type": "Text",
        "chunk_length": len(chunk_text),
        "table_fields": table_fields  # Always present
    }
    logger.debug(f"Chunk metadata: {json.dumps(metadata, indent=2)}")
    return metadata

    product = extract_product_name(chunk_text, source_file_name)
    section = detect_section_type(chunk_text)
    table_fields = extract_table_fields(chunk_text)
    # Always ensure product_name and ordering_code in table_fields
    if not table_fields.get('product_name'):
        table_fields['product_name'] = product or "unknown"
    if 'ordering_code' not in table_fields:
        table_fields['ordering_code'] = None
    metadata = {
        "chunk_id": str(uuid.uuid4()),
        "chunk_hash": str(hash(chunk_text)),
        "section_type": section,
        "product_name": table_fields['product_name'],
        "ordering_code": table_fields['ordering_code'],
        "source_file_name": source_file_name,
        "user": "company_data",
        "original_element_type": "Text",
        "chunk_length": len(chunk_text),
        "table_fields": table_fields  # Always present
    }
    logger.debug(f"Chunk metadata: {json.dumps(metadata, indent=2)}")
    return metadata

def chunk_document_text(
    text_content: str,
    filename: str,
    elements: List[Any],
    min_chars_per_chunk: int = 30,
    max_chars_per_chunk: int = 800,
    overlap_chars: int = 50,
    metadata: Dict = None
) -> List[Dict]:
    """
    Splits a large text document into smaller, overlapping chunks using semantic chunking.
    Each returned chunk is a dictionary containing 'content' and 'metadata',
    with all metadata normalized and validated for robust retrieval.
    """
    if not text_content or not isinstance(text_content, str):
        return []

    # --- Preprocess text_content ---
    text_content = text_content.strip()
    text_content = re.sub(r'<[^>]+>', '', text_content) # Remove all HTML tags
    text_content = re.sub(r'\s*\n\s*\n\s*', '\n\n', text_content) # Normalize multiple newlines
    text_content = re.sub(r'(?i)(product category|product category)\s*', '', text_content)
    text_content = re.sub(r'\s*\(W\)\s*\+/\-5W\s*', ' (W)+/-5W ', text_content)
    text_content = re.sub(r'\"([^\"]+)\"\s*,\s*\"([^\"]+)\"', r'\1: \2', text_content)
    text_content = re.sub(r'^\s*[\-\*\•]\s*', '- ', text_content, flags=re.MULTILINE)
    text_content = re.sub(r'^\s*\d+\.\s*', 'Numbered List Item: ', text_content, flags=re.MULTILINE)
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    if not text_content:
        logger.warning("chunk_document_text: Text content is empty after preprocessing. Returning empty list.")
        return []

    # --- Pydantic models for validation ---
    try:
        from pydantic import BaseModel, ValidationError, Field
    except ImportError:
        logger.warning("Pydantic not installed. Skipping validation.")
        BaseModel = object
        ValidationError = Exception
        Field = lambda *a, **k: None

    class ChunkMetadataModel(BaseModel):
        user: str = Field(..., min_length=1)
        section_type: Optional[str]
        product_name: Optional[str]
        document_type: Optional[str]
        # Extend as needed

    class ChunkModel(BaseModel):
        content: str = Field(..., min_length=1)
        metadata: ChunkMetadataModel

    # --- User whitelist enforcement ---
    ALLOWED_USERS = {"company_data", "admin", "superuser"}
    def normalize_user(user_val):
        if not isinstance(user_val, str) or user_val.strip().lower() not in ALLOWED_USERS:
            return "company_data"
        return user_val.strip().lower()

    # --- Semantic chunking ---
    try:
        from semantic_chunker import semantic_chunk_unstructured, semantic_chunk_plain_text
    except ImportError:
        logger.error("semantic_chunker.py not found. Fallback to single chunk.")
        semantic_chunk_unstructured = lambda elements, max_chars, overlap: [" ".join(str(e) for e in elements)]
        semantic_chunk_plain_text = lambda text, max_chars, overlap: [text]

    processed_chunks_with_metadata = []

    # --- Smart Chunking Logic ---
    # Prioritize unstructured elements if available and semantic chunker is enabled
    if UNSTRUCTURED_AVAILABLE and SEMANTIC_CHUNKER_AVAILABLE and elements:
        logger.info(f"Using semantic_chunk_unstructured for '{filename}' based on elements.")
        semantically_chunked_elements = semantic_chunk_unstructured(
            elements,
            max_chars=max_chars_per_chunk,
            overlap=overlap_chars
        )
        chunks_to_process = [{'content': str(el), 'original_element_type': type(el).__name__} for el in semantically_chunked_elements]
    elif SEMANTIC_CHUNKER_AVAILABLE:
        logger.info(f"Using semantic_chunk_plain_text for '{filename}'.")
        semantically_chunked_elements = semantic_chunk_plain_text(
            text_content,
            max_chars=max_chars_per_chunk,
            overlap=overlap_chars
        )
        chunks_to_process = [{'content': str(el), 'original_element_type': 'Text'} for el in semantically_chunked_elements]
    else:
        logger.warning(f"Semantic chunker not available. Falling back to basic chunking for '{filename}'.")
        paragraphs = text_content.split('\n\n')
        temp_chunks = []
        current_chunk_text_fallback = ""
        for para in paragraphs:
            if len(current_chunk_text_fallback) + len(para) + 2 <= max_chars_per_chunk:
                current_chunk_text_fallback += ("\n\n" if current_chunk_text_fallback else "") + para
            else:
                if current_chunk_text_fallback:
                    temp_chunks.append(current_chunk_text_fallback)
                current_chunk_text_fallback = para
        if current_chunk_text_fallback:
            temp_chunks.append(current_chunk_text_fallback)
        chunks_to_process = [{'content': chunk, 'original_element_type': 'Text'} for chunk in temp_chunks]

    # --- Per-chunk metadata extraction ---
    from rag_utils import extract_metadata_with_llm
    for chunk in chunks_to_process:
        chunk_text = chunk['content']
        try:
            chunk_metadata = extract_metadata_with_llm(chunk_text)
        except Exception as e:
            logger.error(f"Metadata extraction failed for chunk: {e}")
            chunk_metadata = {}
        processed_chunks_with_metadata.append({'content': chunk_text, 'metadata': chunk_metadata})

    # Patterns for section headers and markdown tables (used for metadata tagging, not splitting here)
    section_headers_pattern = r'(?i)^(TECHNICAL\s+SPECIFICATIONS?|PRODUCT\s+SPECIFICATIONS?|FEATURES|APPLICATION|ELECTRICAL\s+CHARACTERISTICS|LAMP\s+CHARACTERISTICS|DIMENSIONS|INSTALLATION|MAINTENANCE|WORKING\s+HOURS|LEAVE/HOLIDAYS|STATUTORY\s+BENEFITS|EMPLOYEE\s+BENEFITS|HEALTH\s+&\s+SAFETY|DISCIPLINE\s+POLICY|GENERAL\s+RULES|RETIREMENT|TERMINATION|ABOUT\s+US|OUR\s+QUALITY|PRODUCT\s+OVERVIEW|TECHNICAL\s+HIGHLIGHTS)$'
    markdown_table_pattern = r'(?m)(^\|.+\|$\n^\|[ :\-|=]+\|$\n(?:^\|.+\|$\n?)+)' # Matches full markdown table structure, added (?m) for multiline
    
    # Noise Filtering patterns (used for filtering chunks, not splitting)
    known_noise_patterns = [
        r"(?i)^more$", r"(?i)^home$", r"(?i)^services$", r"(?i)^portfolio$", r"(?i)^team$", r"(?i)^blog$", r"(?i)^gallery$", r"(?i)career$", r"(?i)contact us$", r"(?i)subscribe to our newsletter.*", r"(?i)certificats$", r"(?i)achievements$", r"(?i)useful\s+links", r"(?i)our\s+infrastructure", r"(?i)our\s+exhibitions", r"(?i)company\s+background", r"(?i)about\s+us", r"(?i)industrial\s*\|\s*lighting\s*\|\s*medical\s*\|\s*products\s*\|\s*covid\s*\|\s*products\s*\|\s*our\s*\|\s*newsletter", # Specific observed noise
        r"(?i)plug\s*sensors\s*pir\s*occupancy\s*sensor\s*covid\s*products\s*hand\s*held\s*sanitizer.*", # Specific observed noise
        r"(?i)led\s*decorative\s*post\s*of\s*lantern\s*industrial\s*lighting.*", # Specific observed noise
        r"(?i)solar\s*street\s*light\s*solar\s*garden\s*light\s*solar\s*flood\s*light.*", # Another observed noise pattern
        r"(?i)led\s*street\s*light\s*led\s*flood\s*light\s*led\s*high\s*bay\s*light.*", # Another observed noise pattern
        r"(?i)product\s*category\s*.*", # General product category headers that are often just noise
        r"(?i)parameter\s*\|\s*value\s*\|\s*---\s*\|\s*---", # Common table header noise
        r"(?i)more\s*details\s*here", r"(?i)click\s*to\s*view", r"(?i)learn\s*more", # Call to action phrases
        r"(?i)all\s*rights\s*reserved", r"(?i)privacy\s*policy", r"(?i)terms\s*of\s*service" # Footer/legal text
    ]

    # Set chunking thresholds based on file type
    is_md_table_doc = is_markdown_table_doc(filename)
    current_min_chars = min_chars_per_chunk
    current_min_alphanum_ratio = MIN_ALPHANUM_RATIO
    if is_md_table_doc:
        current_min_chars = 20
        current_min_alphanum_ratio = 0.2

    for chunk_info in chunks_to_process:
        chunk_content = chunk_info['content'].strip()
        original_element_type = chunk_info.get('original_element_type', 'Text')

        if not chunk_content:
            continue

        alphanum_ratio = calculate_alphanum_ratio(chunk_content)
        is_noise = False
        for pattern in known_noise_patterns:
            if re.search(pattern, chunk_content, re.IGNORECASE):
                is_noise = True
                break

        # Apply filtering criteria: Don't filter out crucial element types or chunks with critical phrases
        if (is_noise or len(chunk_content) < current_min_chars or alphanum_ratio < current_min_alphanum_ratio) and \
           not contains_critical_phrase(chunk_content) and \
           original_element_type not in ["Title", "Table", "ListItem", "NarrativeText"]: # Added NarrativeText for general content
            logger.debug(f"Rejected chunk due to noise/length/alphanum (noise: {is_noise}, len: {len(chunk_content)}, ratio: {alphanum_ratio:.2f}, type: {original_element_type}): {chunk_content[:100]}...")
            continue

        chunk_meta = dict(metadata) if metadata else {}
        chunk_meta["original_element_type"] = original_element_type

        # --- Smarter Section Type Assignment for Individual Chunks ---
        # This logic is applied *per chunk* after semantic chunking.
        chunk_lower = chunk_content.lower()
        section_type = "general"

        # Check for section headers (if the chunk itself is a header)
        header_match = re.fullmatch(section_headers_pattern, chunk_content, re.IGNORECASE)
        if header_match:
            detected_header = header_match.group(1).lower().replace(' ', '_').replace('/', '_and_')
            if detected_header in ["technical_specifications", "product_specifications", "technical_highlights"]:
                section_type = "technical_specifications"
            elif detected_header in ["features"]:
                section_type = "features"
            elif detected_header in ["application"]:
                section_type = "application"
            elif detected_header in ["leave_holidays", "leave_and_holidays"]:
                section_type = "hr_leaves_holidays"
            elif detected_header in ["health_safety", "health_and_safety"]:
                section_type = "hr_health_safety"
            else:
                section_type = detected_header # Use the detected header as section type
            logger.debug(f"Chunk identified as section header: {section_type}")

        # Check for markdown tables
        elif re.match(markdown_table_pattern, chunk_content, flags=re.DOTALL | re.MULTILINE):
            section_type = 'table'
            # Further refine table section type based on content
            if any(phrase in chunk_lower for phrase in ["voltage", "wattage", "power consumption", "ip rating", "cct", "cri", "luminous efficacy", "dimensions", "technical", "specifications"]):
                section_type = 'technical_specifications'
            elif "features" in chunk_lower:
                section_type = 'features'
            elif "application" in chunk_lower:
                section_type = 'application'
            elif "ordering code" in chunk_lower:
                section_type = 'ordering'
            
            # Extract column names for table chunks if applicable
            table_rows = chunk_content.strip().split('\n')
            if len(table_rows) >= 2 and table_rows[0].strip().startswith('|') and re.match(r'^\|[ :\-|=]+\|$', table_rows[1].strip()):
                headers = [h.strip() for h in table_rows[0].strip('|').split('|')]
                if headers:
                    chunk_meta['table_column_names'] = headers
            logger.debug(f"Chunk identified as markdown table: {section_type}")

        elif original_element_type == "Table": # For Unstructured Table elements
            section_type = 'table'
            # Attempt to infer more specific table type from content
            if any(phrase in chunk_lower for phrase in ["voltage", "wattage", "power consumption", "ip rating", "cct", "cri", "luminous efficacy", "dimensions", "technical", "specifications"]):
                section_type = 'technical_specifications'
            elif "features" in chunk_lower:
                section_type = 'features'
            elif "application" in chunk_lower:
                section_type = 'application'
            elif "ordering code" in chunk_lower:
                section_type = 'ordering'
            logger.debug(f"Chunk identified as Unstructured Table element: {section_type}")

        elif original_element_type == "ListItem":
            section_type = "list_item"
            if any(phrase in chunk_lower for phrase in ["feature", "highlight", "spec", "parameter"]):
                section_type = "features" # Tag list items that look like features
            logger.debug(f"Chunk identified as ListItem: {section_type}")

        elif original_element_type == "Title":
            section_type = "title"
            # If title looks like a product name, add it to metadata
            inferred_product = infer_product_name(chunk_content, filename)
            if inferred_product and inferred_product != "unknown_product":
                chunk_meta["product_name"] = inferred_product
                chunk_meta["document_title"] = chunk_content # Store the actual title
            logger.debug(f"Chunk identified as Title: {section_type}")
        
        # Add the determined section_type to the chunk's metadata
        chunk_meta['section_type'] = section_type

        # --- ENFORCE product_name and power_consumption for every chunk (from filename) ---
        product_name, power_consumption = extract_product_name_and_power_from_filename(filename)
        # Robust normalization
        if product_name:
            product_name = product_name.strip().title()
        else:
            product_name = "unknown_product"
        if power_consumption:
            power_consumption = power_consumption.strip().upper()
        else:
            power_consumption = None
        chunk_meta['product_name'] = product_name
        chunk_meta['power_consumption'] = power_consumption
        if product_name == "unknown_product" or not power_consumption:
            logger.warning(f"[Metadata Enforcement] Could not reliably extract product_name or power_consumption from filename '{filename}'. Parsed values: product_name='{product_name}', power_consumption='{power_consumption}'. Chunk content snippet: '{chunk_content[:60]}...'")

# --- Per-chunk metadata extraction ---
from rag_utils import extract_metadata_with_llm
existing_hashes = set()
for chunk in chunks_to_process:
    chunk_text = chunk['content']
    try:
        chunk_metadata = extract_metadata_with_llm(chunk_text)
    except Exception as e:
        logger.error(f"Metadata extraction failed for chunk: {e}")
        chunk_metadata = {}
    chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
    if chunk_hash in existing_hashes:
        logger.info(f"Duplicate chunk skipped: {chunk_hash}")
        continue
    existing_hashes.add(chunk_hash)
    processed_chunks_with_metadata.append({'content': chunk_text, 'metadata': chunk_metadata})

# Patterns for section headers and markdown tables (used for metadata tagging, not splitting here)
section_headers_pattern = r'(?i)^(TECHNICAL\s+SPECIFICATIONS?|PRODUCT\s+SPECIFICATIONS?|FEATURES|APPLICATION|ELECTRICAL\s+CHARACTERISTICS|LAMP\s+CHARACTERISTICS|DIMENSIONS|INSTALLATION|MAINTENANCE|WORKING\s+HOURS|LEAVE/HOLIDAYS|STATUTORY\s+BENEFITS|EMPLOYEE\s+BENEFITS|HEALTH\s+&\s+SAFETY|DISCIPLINE\s+POLICY|GENERAL\s+RULES|RETIREMENT|TERMINATION|ABOUT\s+US|OUR\s+QUALITY|PRODUCT\s+OVERVIEW|TECHNICAL\s+HIGHLIGHTS)$'
markdown_table_pattern = r'(?m)(^\|.+\|$\n^\|[ :\-|=]+\|$\n(?:^\|.+\|$\n?)+)' # Matches full markdown table structure, added (?m) for multiline

# Noise Filtering patterns (used for filtering chunks, not splitting)
known_noise_patterns = [
    r"(?i)^more$", r"(?i)^home$", r"(?i)^services$", r"(?i)^portfolio$", r"(?i)^team$", r"(?i)^blog$", r"(?i)^gallery$", r"(?i)career$", r"(?i)contact us$", r"(?i)subscribe to our newsletter.*", r"(?i)certificats$", r"(?i)achievements$", r"(?i)useful\s+links", r"(?i)our\s+infrastructure", r"(?i)our\s+exhibitions", r"(?i)company\s+background", r"(?i)about\s+us", r"(?i)industrial\s*\|\s*lighting\s*\|\s*medical\s*\|\s*products\s*\|\s*covid\s*\|\s*products\s*\|\s*our\s*\|\s*newsletter", # Specific observed noise
    r"(?i)plug\s*sensors\s*pir\s*occupancy\s*sensor\s*covid\s*products\s*hand\s*held\s*sanitizer.*", # Specific observed noise
    r"(?i)led\s*decorative\s*post\s*of\s*lantern\s*industrial\s*lighting.*", # Specific observed noise
    r"(?i)solar\s*street\s*light\s*solar\s*garden\s*light\s*solar\s*flood\s*light.*", # Another observed noise pattern
    r"(?i)led\s*street\s*light\s*led\s*flood\s*light\s*led\s*high\s*bay\s*light.*", # Another observed noise pattern
    r"(?i)product\s*category\s*.*", # General product category headers that are often just noise
    r"(?i)parameter\s*\|\s*value\s*\|\s*---\s*\|\s*---", # Common table header noise
    r"(?i)more\s*details\s*here", r"(?i)click\s*to\s*view", r"(?i)learn\s*more", # Call to action phrases
    r"(?i)all\s*rights\s*reserved", r"(?i)privacy\s*policy", r"(?i)terms\s*of\s*service" # Footer/legal text
]

# Set chunking thresholds based on file type
is_md_table_doc = is_markdown_table_doc(filename)
current_min_chars = min_chars_per_chunk
current_min_alphanum_ratio = MIN_ALPHANUM_RATIO
if is_md_table_doc:
    current_min_chars = 20
    current_min_alphanum_ratio = 0.2

for chunk_info in processed_chunks_with_metadata:
    chunk_content = chunk_info['content'].strip()
    original_element_type = chunk_info.get('original_element_type', 'Text')

    if not chunk_content:
        continue

    alphanum_ratio = calculate_alphanum_ratio(chunk_content)
    is_noise = False
    for pattern in known_noise_patterns:
        if re.search(pattern, chunk_content, re.IGNORECASE):
            is_noise = True
            break

    # Apply filtering criteria: Don't filter out crucial element types or chunks with critical phrases
    if (is_noise or len(chunk_content) < current_min_chars or alphanum_ratio < current_min_alphanum_ratio) and \
       not contains_critical_phrase(chunk_content) and \
       original_element_type not in ["Title", "Table", "ListItem", "NarrativeText"]: # Added NarrativeText for general content
        logger.debug(f"Rejected chunk due to noise/length/alphanum (noise: {is_noise}, len: {len(chunk_content)}, ratio: {alphanum_ratio:.2f}, type: {original_element_type}): {chunk_content[:100]}...")
        continue

    chunk_meta = dict(metadata) if metadata else {}
    chunk_meta["original_element_type"] = original_element_type

    # --- Smarter Section Type Assignment for Individual Chunks ---
    # This logic is applied *per chunk* after semantic chunking.
    chunk_lower = chunk_content.lower()
    section_type = "general"

    # Check for section headers (if the chunk itself is a header)
    header_match = re.fullmatch(section_headers_pattern, chunk_content, re.IGNORECASE)
    if header_match:
        detected_header = header_match.group(1).lower().replace(' ', '_').replace('/', '_and_')
        if detected_header in ["technical_specifications", "product_specifications", "technical_highlights"]:
            section_type = "technical_specifications"
        elif detected_header in ["features"]:
            section_type = "features"
        elif detected_header in ["application"]:
            section_type = "application"
        elif detected_header in ["leave_holidays", "leave_and_holidays"]:
            section_type = "hr_leaves_holidays"
        elif detected_header in ["health_safety", "health_and_safety"]:
            section_type = "hr_health_safety"
        else:
            section_type = detected_header # Use the detected header as section type
        logger.debug(f"Chunk identified as section header: {section_type}")

    # Check for markdown tables
    elif re.match(markdown_table_pattern, chunk_content, flags=re.DOTALL | re.MULTILINE):
        section_type = 'table'
        # Further refine table section type based on content
        if any(phrase in chunk_lower for phrase in ["voltage", "wattage", "power consumption", "ip rating", "cct", "cri", "luminous efficacy", "dimensions", "technical", "specifications"]):
            section_type = 'technical_specifications'
        elif "features" in chunk_lower:
            section_type = 'features'
        elif "application" in chunk_lower:
            section_type = 'application'
        elif "ordering code" in chunk_lower:
            section_type = 'ordering'
        
        # Extract column names for table chunks if applicable
        table_rows = chunk_content.strip().split('\n')
        if len(table_rows) >= 2 and table_rows[0].strip().startswith('|') and re.match(r'^\|[ :\-|=]+\|$', table_rows[1].strip()):
            headers = [h.strip() for h in table_rows[0].strip('|').split('|')]
            if headers:
                chunk_meta['table_column_names'] = headers
        logger.debug(f"Chunk identified as markdown table: {section_type}")

    elif original_element_type == "Table": # For Unstructured Table elements
        section_type = 'table'
        # Attempt to infer more specific table type from content
        if any(phrase in chunk_lower for phrase in ["voltage", "wattage", "power consumption", "ip rating", "cct", "cri", "luminous efficacy", "dimensions", "technical", "specifications"]):
            section_type = 'technical_specifications'
        elif "features" in chunk_lower:
            section_type = 'features'
        elif "application" in chunk_lower:
            section_type = 'application'
        elif "ordering code" in chunk_lower:
            section_type = 'ordering'
        logger.debug(f"Chunk identified as Unstructured Table element: {section_type}")

    elif original_element_type == "ListItem":
        section_type = "list_item"
        if any(phrase in chunk_lower for phrase in ["feature", "highlight", "spec", "parameter"]):
            section_type = "features" # Tag list items that look like features
        logger.debug(f"Chunk identified as ListItem: {section_type}")

    elif original_element_type == "Title":
        section_type = "title"
        # If title looks like a product name, add it to metadata
        inferred_product = infer_product_name(chunk_content, filename)
        if inferred_product and inferred_product != "unknown_product":
            chunk_meta["product_name"] = inferred_product
            chunk_meta["document_title"] = chunk_content # Store the actual title
        logger.debug(f"Chunk identified as Title: {section_type}")
    
    # Add the determined section_type to the chunk's metadata
    chunk_meta['section_type'] = section_type

    # --- ENFORCE product_name and power_consumption for every chunk (from filename) ---
    product_name, power_consumption = extract_product_name_and_power_from_filename(filename)
    # Robust normalization
    if product_name:
        product_name = product_name.strip().title()
    else:
        product_name = "unknown_product"
    if power_consumption:
        power_consumption = power_consumption.strip().upper()
    else:
        power_consumption = None
    chunk_meta['product_name'] = product_name
    chunk_meta['power_consumption'] = power_consumption
    if product_name == "unknown_product" or not power_consumption:
        logger.warning(f"[Metadata Enforcement] Could not reliably extract product_name or power_consumption from filename '{filename}'. Parsed values: product_name='{product_name}', power_consumption='{power_consumption}'. Chunk content snippet: '{chunk_content[:60]}...'")
    # This guarantees these fields are always present and correct, regardless of LLM or upstream metadata.

    # Add chunk_id, chunk_length, chunk_hash
    chunk_meta["chunk_id"] = str(uuid.uuid4())
    chunk_meta["chunk_length"] = len(chunk_content)
    chunk_meta["chunk_hash"] = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()

    processed_chunks_with_metadata.append({'content': chunk_content, 'metadata': chunk_meta})
    logger.debug(f"Added chunk (len: {len(chunk_content)}, type: {original_element_type}, section: {section_type}): {chunk_content[:100]}...")

return processed_chunks_with_metadata


def _process_single_document_for_ingestion(file_path: str, filename: str, file_extension: str, username: str) -> str:
    """
    Processes a single document (TXT, CSV, JSON, PDF, DOCX) for ingestion into ChromaDB.
    Handles text extraction, chunking, embedding, and metadata association.
    Includes idempotency check.
    """
    logger.info(f"Processing document for storage: '{filename}' by user: '{username}'", extra={"doc_filename_log": filename, "ingestion_user": username})
    
    text_content = ""
    elements = []
    
    # Initialize extracted_metadata here, it will be updated by extract_document_metadata
    # and then passed to chunk_document_text
    extracted_metadata = {"source_file_name": filename, "user": username} 

    # --- Begin: Clean and set product_name and source_file_name in document_level_metadata ---

    original_filename = os.path.basename(file_path)
    document_level_metadata = extracted_metadata  # Alias for clarity
    if "product_name" not in document_level_metadata or not document_level_metadata.get("product_name", "").strip():
        product_name_match = re.sub(r"^\d+\.\s*", "", original_filename)  # Remove leading index like '1. '
        product_name_match = re.sub(r"\d{2,3}-\d{2,3}W\s*", "", product_name_match)  # Remove wattage like '30-45W'
        product_name_match = re.sub(r"_[a-f0-9]{6,}\.txt$", "", product_name_match, flags=re.IGNORECASE)  # Remove hash
        product_name_match = product_name_match.replace(".txt", "").strip()
        document_level_metadata["product_name"] = product_name_match
    clean_source_name = document_level_metadata["product_name"].replace(" ", "_").replace("/", "_") + ".txt"
    document_level_metadata["source_file_name"] = clean_source_name
    # --- End: Clean and set product_name and source_file_name ---

    raw_user = extracted_metadata.get("user", username)
    extracted_metadata["user"] = str(raw_user).strip().lower() if isinstance(raw_user, str) else "company_data"

    try:
        # --- Text Extraction based on file type ---
        if file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text_content = f.read()
            text_content = raw_text_content.strip()
            logger.info(f"Read TXT file: {file_path}", extra={"file_path_log": file_path})
            # Correctly use partition_text for raw string content
            if UNSTRUCTURED_AVAILABLE:
                elements = partition_text(text=text_content) 
            else:
                elements = [Text(text_content)] # Fallback if Unstructured not available

        elif file_extension == ".csv":
            if UNSTRUCTURED_AVAILABLE:
                elements = partition_csv(filename=file_path)
            else:
                # Basic fallback for CSV if Unstructured is not available
                with open(file_path, "r", encoding="utf-8") as f:
                    csv_content = f.read()
                elements = [Text(csv_content)]
            text_content_raw = "\n\n".join([ str(el) for el in elements if isinstance(el, (Text, NarrativeText, Title, Table, ListItem)) and str(el).strip() ])
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
                        text_content_parts.append(item)
                        if re.match(r'^\|.*\|$\n^\|[-:|=]+\|$\n(?:^\|.*\|$\n?)+', item, re.DOTALL | re.MULTILINE):
                            markdown_table_found = True
                    elif isinstance(item, dict):
                        if 'type' in item and item['type'] == 'table' and 'content' in item and \
                           isinstance(item['content'], str):
                            text_content_parts.append(item['content'])
                            if re.match(r'^\|.*\|$\n^\|[-:|=]+\|$\n(?:^\|.*\|$\n?)+', item['content'], re.DOTALL | re.MULTILINE):
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
                        text_content_parts.append(str(item))
                text_content = "\n\n".join(text_content_parts)
            elif isinstance(processed_json_data, dict):
                dict_content_parts = []
                for key, value in processed_json_data.items():
                    if isinstance(value, str):
                        dict_content_parts.append(f"{key}: {value}")
                    elif isinstance(value, dict):
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
                text_content = "\n\n".join(dict_content_parts)
            elif isinstance(processed_json_data, str):
                text_content = processed_json_data
            else:
                text_content = json.dumps(processed_json_data, indent=2)
                logger.warning(f"JSON file '{filename}' did not contain a list, dict, or string at top level after processing. Treating as raw JSON string. Type: {type(processed_json_data).__name__}", extra={"doc_filename_log": filename, "json_data_type": type(processed_json_data).__name__})
            logger.info(f"Read and processed JSON file: {file_path}", extra={"file_path_log": file_path})

        elif file_extension in [".pdf", ".docx"]:
            logger.info(f"Partitioning {file_extension} file using unstructured.partition.auto: {file_path}", extra={"file_path_log": file_path})
            elements = partition_auto_file(filename=file_path)
            text_content = "\n\n".join([str(el) for el in elements if str(el).strip()])
            text_content = text_content.strip()

        else:
            return f"WARNING: Skipping unsupported file type '{file_extension}' for ingestion: {filename}"

        # --- CENTRALIZED TEXT PREPROCESSING AFTER text_content IS ASSEMBLED ---
        text_content = re.sub(r'<[^>]+>', '', text_content) # Remove all HTML tags
        text_content = re.sub(r'\s*\n\s*\n\s*', '\n\n', text_content) # Normalize multiple newlines
        text_content = re.sub(r'(?i)(product category|product category)\s*', '', text_content) # Specific removals
        text_content = re.sub(r'\s*\(W\)\s*\+/\-5W\s*', ' (W)+/-5W ', text_content) # Specific replacements
        text_content = re.sub(r'\"([^\"]+)\"\s*,\s*\"([^\"]+)\"', r'\1: \2', text_content) # Normalize quoted pairs
        text_content = re.sub(r'^\s*[\-\*\•]\s*', '- ', text_content, flags=re.MULTILINE) # Normalize list bullets
        text_content = re.sub(r'^\s*\d+\.\s*', 'Numbered List Item: ', text_content, flags=re.MULTILINE) # Normalize numbered lists
        text_content = re.sub(r'\s+', ' ', text_content).strip() # Normalize all whitespace to single space
        logger.debug(f"Text content from '{filename}' after preprocessing (first 500 chars):\n{text_content[:500]}...", extra={"doc_filename_log": filename, "content_snippet_log": text_content[:500]})

        if not text_content.strip():
            return f"Processed {filename}: Warning: Document contains no usable text after parsing and preprocessing. Skipping."

        # --- CRITICAL FIX: Extract metadata using LLM and update extracted_metadata ---
        @retry(wait=wait_fixed(5), stop=stop_after_attempt(3), retry=retry_if_exception_type(RequestException), before_sleep=before_sleep_log(logger, logging.WARNING))
        def _local_llm_call_for_ingestion(prompt_text: str) -> str:
            try:
                logger.info(f"Starting LLM call for metadata extraction for '{filename}'...")
                # Enforce JSON-only output from LLM
                json_only_prompt = f"{prompt_text}\n\nReturn only a valid JSON object containing extracted fields. Do not include any explanation or additional text."
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": OLLAMA_METADATA_MODEL_NAME, # Use the specific metadata model
                        "messages": [{"role": "user", "content": json_only_prompt}]
                    },
                    timeout=(30, 1200) # Connect timeout 30s, Read timeout 600s
                )
                response.raise_for_status()
                logger.info(f"LLM call for metadata extraction for '{filename}' completed successfully.")
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                logger.error(f"Error in _local_llm_call_for_ingestion (RequestException): {e}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Error in _local_llm_call_for_ingestion (General Exception): {e}", exc_info=True)
                return "{}"

        # Use safe_parse_llm_json_response to robustly parse the LLM's raw string output
        llm_raw_response_str = _local_llm_call_for_ingestion(text_content)
        # Early validator for non-JSON response
        if not llm_raw_response_str.strip().startswith("{") and not llm_raw_response_str.strip().startswith("["):
            raise ValueError("Non-JSON response received from LLM")
        parsed_llm_json = safe_parse_llm_json_response(llm_raw_response_str)

        # Now pass the already parsed and potentially corrected JSON to extract_metadata_with_schema
        llm_extracted_meta = extract_metadata_with_schema(
            llm_json=parsed_llm_json,
            doc_type="default", # Or infer a more specific doc_type here
            llm_call_fn=_local_llm_call_for_ingestion # Pass for potential recursive calls if schema validation needs it
        )
        extracted_metadata.update(llm_extracted_meta)
        logger.info(f"LLM-extracted and processed metadata for '{filename}': {extracted_metadata}", extra={"doc_filename_log": filename, "extracted_metadata_final": extracted_metadata})


        logger.info(f"Chunking text from '{filename}' with content length {len(text_content)}...", extra={"doc_filename_log": filename, "content_length_chars": len(text_content)})
        
        # Pass the rich `extracted_metadata` and elements to the chunking function
        chunks_with_metadata = chunk_document_text(
            text_content,
            filename,
            elements, # Pass elements for chunking logic to use
            min_chars_per_chunk=MIN_CHUNK_LENGTH,
            max_chars_per_chunk=MAX_CHUNK_CHARS,
            overlap_chars=CHUNK_OVERLAP,
            metadata=extracted_metadata # CRITICAL: Pass the updated metadata here
        )

        logger.debug(f"[{filename}] Chunking {len(text_content)} chars -> {len(chunks_with_metadata)} chunks.", extra={"doc_filename_log": filename, "original_chars_count": len(text_content), "num_chunks_created": len(chunks_with_metadata)})

        if not chunks_with_metadata:
            return f"Processed {filename}: Error: No usable text chunks generated from {filename} after all attempts. Document might be genuinely empty or unprocessable."

        chunks = [c['content'] for c in chunks_with_metadata]
        metadatas = [c['metadata'] for c in chunks_with_metadata]
        
        # Normalize all metadata before storing and ensure no None values
        normalized_metadatas = []
        for meta in metadatas:
            # Normalize source_file_name and product_name
            meta["source_file_name"] = os.path.basename(file_path).replace(" ", "_")
            meta["product_name"] = meta.get("product_name", "").title()
            # Sanitize and validate power_consumption
            import re
            raw_power = meta.get("power_consumption", "").strip().lower()
            if raw_power in ["", "unknown", None]:
                meta["power_consumption"] = "unknown"
            elif re.match(r"^\d+[-/]?\d*\s*w$", raw_power):
                meta["power_consumption"] = raw_power.upper()
            else:
                meta["power_consumption"] = "unknown"
            chromadb_compatible_meta = {}
            for k, v in meta.items():
                # Skip None values entirely as ChromaDB doesn't accept them
                if v is None:
                    logger.debug(f"Skipping None value for metadata key '{k}' in document {filename}")
                    continue
                
                # Convert non-scalar types to strings
                if not isinstance(v, (str, int, float, bool)):
                    logger.warning(f"Metadata value for key '{k}' has unsupported type '{type(v)}' in document {filename}. Converting to string. Value: {v}")
                    v = str(v)
                
                # Add to metadata if value is not None after conversion
                if v is not None:
                    chromadb_compatible_meta[k] = v
            
            # Ensure required fields have default values if missing
            if not chromadb_compatible_meta.get('section_type'):
                chromadb_compatible_meta["section_type"] = "product technical datasheet"
                
            # Only add non-empty metadata
            if chromadb_compatible_meta:
                normalized_metadatas.append(chromadb_compatible_meta)
            else:
                logger.warning(f"All metadata was filtered out for a chunk in {filename}. Adding minimal metadata.")
                normalized_metadatas.append({"section_type": "product technical datasheet", "source_file": filename})
                
        # Ensure we have the same number of metadatas as chunks
        if len(normalized_metadatas) != len(chunks):
            logger.warning(f"Mismatch between number of chunks ({len(chunks)}) and metadatas ({len(normalized_metadatas)}) in {filename}. Adjusting...")
            # Ensure we have at least one metadata object per chunk
            normalized_metadatas = normalized_metadatas[:len(chunks)]
            while len(normalized_metadatas) < len(chunks):
                normalized_metadatas.append({"section_type": "product technical datasheet", "source_file": filename})
        
        logger.debug(f"Normalized metadatas for '{filename}' (first 2 entries): {normalized_metadatas[:2]}", extra={"doc_filename_log": filename, "normalized_metadatas_snippet": normalized_metadatas[:2]})

        # Generate embeddings for the chunks
        chunk_embeddings = embedding_function.embed_documents(chunks)
        
        if not chunk_embeddings:
            return f"Processed {filename}: Error: Failed to generate embeddings for chunks. Skipping ingestion."

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}-{chunk_idx}-{hashlib.md5(chunk.encode('utf-8')).hexdigest()}")) for chunk_idx, chunk in enumerate(chunks)]

        try:
            # Add validation before attempting to add to ChromaDB
            if not chunks or not chunk_embeddings or not normalized_metadatas or not ids:
                error_msg = f"Cannot add empty data to ChromaDB for {filename}. Chunks: {len(chunks)}, Embeddings: {len(chunk_embeddings)}, Metadatas: {len(normalized_metadatas)}, IDs: {len(ids)}"
                logger.error(error_msg)
                return f"Processed {filename}: {error_msg}"
                
            # Log first metadata for debugging
            logger.debug(f"First metadata being added: {json.dumps(normalized_metadatas[0], indent=2, default=str)}")
            
            # Add documents in smaller batches to avoid timeouts
            batch_size = 50
            success_count = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = chunk_embeddings[i:i+batch_size]
                batch_metadatas = normalized_metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                try:
                    collection.add(
                        documents=batch_chunks,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    success_count += len(batch_chunks)
                    logger.info(f"Successfully added batch {i//batch_size + 1} with {len(batch_chunks)} chunks from '{filename}' to ChromaDB.")
                except Exception as batch_e:
                    logger.error(f"Failed to add batch {i//batch_size + 1} of {filename} to ChromaDB: {batch_e}", exc_info=True)
                    # Try to continue with next batch instead of failing completely
                    continue
            
            if success_count == 0:
                error_msg = f"Failed to add any chunks from {filename} to ChromaDB after multiple attempts."
                logger.error(error_msg)
                return f"Processed {filename}: {error_msg}"
                
            logger.info(f"Successfully added {success_count}/{len(chunks)} chunks from '{filename}' to ChromaDB.")
            return f"Processed {filename}: Successfully added {success_count} chunks to ChromaDB. First chunk chars: {chunks[0][:50]}..."
            
        except Exception as e:
            error_detail = f"Failed to add chunks from '{filename}' to ChromaDB: {str(e)}"
            logger.error(error_detail, exc_info=True)
            
            # Log problematic metadata for debugging
            if 'metadatas' in locals():
                logger.error(f"Problematic metadata sample (first 2): {json.dumps(normalized_metadatas[:2], indent=2, default=str)}")
            
            return f"Processed {filename}: Error during ChromaDB ingestion: {error_detail}"

    except Exception as e:
        logger.error(f"Processed {filename}: Error: Failed to process due to: {e}", exc_info=True, extra={"doc_filename_log": filename, "error_detail": str(e)})
        return f"Processed {filename}: Error: Failed to process due to: {e}"


# --- Main Ingestion Logic ---
def ingest_documents_to_chromadb(text_files_folder: str, table_files_folder: str):
    global collection, embedding_function
    
    if not collection or not embedding_function:
        logger.error("ChromaDB collection or embedding function not initialized. Cannot proceed with ingestion.")
        return

    # --- Clear ChromaDB collection before ingestion ---
    try:
        logger.info(f"Clearing ChromaDB collection '{COLLECTION_NAME}' before ingestion...")
        # Use a more specific delete to avoid issues with empty 'where' clause if ChromaDB version is strict
        # For a full clear, you might need to re-instantiate the collection or use client.delete_collection
        # Given the global client/collection, re-instantiating might be safer.
        # Let's try deleting all documents by providing a dummy filter that matches all.
        collection.delete(where={"source_file_name": {"$ne": "non_existent_file"}}) # Delete all documents
        logger.info(f"Collection '{COLLECTION_NAME}' cleared.")
    except Exception as e:
        logger.warning(f"Could not clear collection '{COLLECTION_NAME}' (might not exist or permissions issue): {e}")

    logger.info("Starting initial static data ingestion from processed directory.")

    text_documents_to_process = []
    logger.info(f"Scanning text files in: {text_files_folder}")
    for root, _, files in os.walk(text_files_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == ".txt": # Explicitly process only .txt here
                text_documents_to_process.append(file_path)
    logger.info(f"Found {len(text_documents_to_process)} text documents to process.")

    # Process text files in batches of 5
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if text_documents_to_process:
        for i in range(0, len(text_documents_to_process), 5):
            batch = text_documents_to_process[i:i+5]
            logger.info(f"Processing text file batch {i//5 + 1}: {batch}")
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_process_single_document_for_ingestion, file_path, os.path.basename(file_path), ".txt", "company_data"): file_path for file_path in batch}
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        logger.info(result)
                    except Exception as e:
                        logger.error(f"Error processing text document '{file_path}': {e}", exc_info=True)
    else:
        logger.info("No text documents found to process.")


    table_documents_to_process = []
    logger.info(f"Scanning table files in: {table_files_folder}")
    for root, _, files in os.walk(table_files_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == ".json": # Explicitly process only .json here
                table_documents_to_process.append(file_path)
    logger.info(f"Found {len(table_documents_to_process)} table documents to process.")

    # Process table files in batches of 5
    if table_documents_to_process:
        for i in range(0, len(table_documents_to_process), 5):
            batch = table_documents_to_process[i:i+5]
            logger.info(f"Processing table file batch {i//5 + 1}: {batch}")
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_process_single_document_for_ingestion, file_path, os.path.basename(file_path), ".json", "company_data"): file_path for file_path in batch}
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        logger.info(result)
                    except Exception as e:
                        logger.error(f"Error processing table JSON document: {e}", exc_info=True)
    else:
        logger.info("No table documents found to process.")

    # Removed: Redundant "Document ingestion complete." message here
    # It will be printed once in the __main__ block


    # --- Log unique users and section_types after ingestion ---
    try:
        meta = collection.get(include=["metadatas"])
        unique_users = set()
        unique_section_types = set()
        for m in meta["metadatas"]:
            if isinstance(m, dict):
                if "user" in m:
                    unique_users.add(str(m["user"]))
                if "section_type" in m:
                    unique_section_types.add(str(m["section_type"]))
        logger.info(f"[POST-INGEST] Unique users in collection: {sorted(unique_users)}")
        logger.info(f"[POST-INGEST] Unique section_types in collection: {sorted(unique_section_types)}")
    except Exception as e:
        logger.warning(f"Could not enumerate unique users/section_types after ingestion: {e}")

    # Moved saving caches to main block for consistent handling
    # if CACHE_AVAILABLE:
    #     try:
    #         if embedding_cache:
    #             embedding_cache.save()
    #             logger.info("Successfully saved embedding cache.")
    #         if reranker_cache:
    #             reranker_cache.save()
    #             logger.info("Successfully saved reranker cache.")
    #     except Exception as e:
    #         logger.error(f"Error saving caches after ingestion: {e}", exc_info=True)


import atexit
# from rag import close_caches # Kept the import for atexit if close_caches is external
atexit.register(close_caches)

if __name__ == "__main__":
    _local_base_dir = os.path.dirname(os.path.abspath(__file__))
    _local_text_files_folder = os.path.join(_local_base_dir, "docs_processed", "text")
    _local_table_files_folder = os.path.join(_local_base_dir, "docs_processed", "tables")
    _local_db_dir = os.path.join(_local_base_dir, "db")

    os.makedirs(_local_text_files_folder, exist_ok=True)
    os.makedirs(_local_table_files_folder, exist_ok=True)
    os.makedirs(_local_db_dir, exist_ok=True)

    logger.info(f"Text files directory: {_local_text_files_folder}")
    logger.info(f"Tables files directory: {_local_table_files_folder}")

    dummy_text_path = os.path.join(_local_text_files_folder, "dummy_product_datasheet.pdf.txt")
    if not os.path.exists(dummy_text_path):
        with open(dummy_text_path, "w", encoding='utf-8') as f:
            f.write("This is a dummy PDF content for LED Hand Lamp. It has some technical specifications.\n")
            f.write("Product Name: LED Hand Lamp Pro\n")
            f.write("Model: PH-11\n")
            f.write("Voltage: 24V DC\n")
            f.write("Wattage: 10W\n")
            f.write("IP Rating: IP67\n")
            f.write("Features: Durable, long-lasting.\n")
            f.write("Applications: Industrial use, emergency lighting.\n")
        logger.info(f"Created dummy text file: {dummy_text_path}")

    dummy_json_path = os.path.join(_local_table_files_folder, "dummy_table_data.pdf.json")
    if not os.path.exists(dummy_json_path):
        dummy_table_content = {
            "table_id": "1",
            "text_as_markdown": "| Feature | Value |\n|---------|-------|\n| Lumens  | 800lm |\n| Battery | Li-ion |\n| Weight | 345g |"
        }
        with open(dummy_json_path, "w", encoding='utf-8') as f:
            json.dump(dummy_table_content, f, indent=2)
        logger.info(f"Created dummy JSON file: {dummy_json_path}")

    try:
        logger.info(f"Starting document ingestion.")
        _local_client = PersistentClient(path=CHROMA_DB_PATH)
        _local_embedding_function = ChromaCompatibleOllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

        try:
            _local_client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Existing collection '{COLLECTION_NAME}' deleted for standalone ingestion.")
        except Exception as e:
            logger.warning(f"Could not delete existing collection '{COLLECTION_NAME}' for standalone ingestion: {e}")
        
        collection = _local_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_local_embedding_function)
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' initialized for standalone ingestion.")

        ingest_documents_to_chromadb(_local_text_files_folder, _local_table_files_folder)
        logger.info("Document ingestion complete.")
        
        if CACHE_AVAILABLE:
            try:
                if embedding_cache:
                    embedding_cache.save()
                    logger.info("Successfully saved embedding cache.")
                if reranker_cache:
                    reranker_cache.save()
                    logger.info("Successfully saved reranker cache.")
            except Exception as e:
                logger.error(f"Error saving caches after ingestion: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"Standalone ingestion failed: {e}", exc_info=True)
