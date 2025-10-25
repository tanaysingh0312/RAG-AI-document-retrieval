import logging
import os
from rag import ask_llm_model_phi3
import uvicorn
import shutil
import uuid
import asyncio
import json
import time
import sqlite3
import base64 # For encoding image bytes to base64
from typing import List, Optional, Dict, ClassVar, Union, Any, Tuple
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, Path, Query, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
import httpx # For making asynchronous HTTP requests to microservices
from PIL import Image # For image manipulation (cropping)
import io # For handling image bytes
import mimetypes # For guessing MIME types
import traceback # For detailed error logging

from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type, before_sleep_log # Added tenacity

# Set up logging for the main FastAPI application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Assuming auth.py and rag.py are in the same directory as main.py
import auth
import rag
from auth import User, SessionSummary, SessionMessages, AdminChatLogEntry
from rag import EmbeddingFunctionWrapper

# --- Service URLs ---
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:8001/process_document")
LAYOUT_DETECTOR_SERVICE_URL = os.getenv("LAYOUT_DETECTOR_SERVICE_URL", "http://localhost:8002/detect_layout")
# Ollama should already be localhost, but double-check:
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")
# Explicitly define embedding URL for clarity, though rag.py constructs it
OLLAMA_EMBED_API_URL = os.getenv("OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings")


# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
INPUT_RAW_DOCS_FOLDER = os.path.join(BASE_DIR, "docs_raw")
OUTPUT_PROCESSED_TEXT_FOLDER = os.path.join(BASE_DIR, 'docs_processed', 'text')
OUTPUT_PROCESSED_TABLES_FOLDER = os.path.join(BASE_DIR, 'docs_processed', 'tables') # New folder for structured tables
PROCESSED_DOCS_FOLDER = os.path.join(BASE_DIR, 'docs_processed') # Parent of text/tables
STATIC_DOCS_SERVE_DIR = os.path.join(PROCESSED_DOCS_FOLDER, 'static_serve')
RAW_DOCS_ARCHIVE_FOLDER = os.path.join(INPUT_RAW_DOCS_FOLDER, 'processed_archive')

# --- Frontend Integration Paths ---
FRONTEND_PROJECT_ROOT = os.path.join(BASE_DIR, "pyrotech-ai-ui")
FRONTEND_DIST_DIR = os.path.join(FRONTEND_PROJECT_ROOT, "dist")

# Ensure directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(INPUT_RAW_DOCS_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_DOCS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_PROCESSED_TEXT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_PROCESSED_TABLES_FOLDER, exist_ok=True) # Ensure tables folder exists
os.makedirs(STATIC_DOCS_SERVE_DIR, exist_ok=True)
os.makedirs(RAW_DOCS_ARCHIVE_FOLDER, exist_ok=True)
os.makedirs(FRONTEND_DIST_DIR, exist_ok=True) 

# --- Redis Cache Configuration ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

import redis
redis_client = None
try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info("Redis cache client initialized successfully.")
except redis.exceptions.ConnectionError as e:
    redis_client = None
    logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.", extra={"service": "Redis", "error": str(e)})

# --- JWT Secret Key ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Initialize Embedding Function ---
# Create embedding function instance that will be passed to RAG functions
embedding_function_instance = None
try:
    embedding_function_instance = EmbeddingFunctionWrapper()
    logger.info("Initialized EmbeddingFunctionWrapper successfully.")
except Exception as e:
    logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
    embedding_function_instance = None

# --- Poppler path for PDF processing (used by pdf2image in OCR service) ---
POPPLER_BIN_PATH_DETECTED = None
poppler_exec = shutil.which("pdftocairo")
if poppler_exec:
    POPPLER_BIN_PATH_DETECTED = os.path.dirname(poppler_exec)
    logger.info(f"Poppler binary path detected in system PATH: {POPPLER_BIN_PATH_DETECTED}")
else:
    POPPLER_BIN_PATH_DETECTED = "/usr/bin" # Default for WSL/Linux common path
    logger.info(f"Poppler binary path not detected, defaulting to common path: {POPPLER_BIN_PATH_DETECTED}")

os.environ["POPPLER_BIN_PATH"] = POPPLER_BIN_PATH_DETECTED
if POPPLER_BIN_PATH_DETECTED and POPPLER_BIN_PATH_DETECTED not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + POPPLER_BIN_PATH_DETECTED

# --- Constants ---
MIN_CHARS_FOR_OCR_RECHECK = 50 # Minimum characters in a text block to avoid re-OCR

# --- Service Health Tracking ---
service_health_status = {
    "ollama_llm": {"status": "unknown", "last_checked": None, "error_count": 0},
    "ollama_embeddings": {"status": "unknown", "last_checked": None, "error_count": 0},
    "chroma_db": {"status": "unknown", "last_checked": None, "error_count": 0},
    "redis_cache": {"status": "unknown", "last_checked": None, "error_count": 0},
    "layout_detector_service": {"status": "unknown", "last_checked": None, "error_count": 0},
    "ocr_service": {"status": "unknown", "last_checked": None, "error_count": 0},
}

# ==============================================================================
# ====== FastAPI APP SETUP ======
# ==============================================================================
import os
from fastapi.responses import FileResponse

app = FastAPI(
    title="Pyrotech FastAPI RAG API",
    description="Backend API for Pyrotech RAG chatbot with authentication, document management, and AI capabilities.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static_docs", StaticFiles(directory=STATIC_DOCS_SERVE_DIR), name="static_docs")


# =============================================================================
# ====== AUTHENTICATION AND DEPENDENCIES ======
# =============================================================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")
security = HTTPBearer()

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    user = auth.verify_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation forbidden: Admin privileges required."
        )
    return current_user

# =============================================================================
# ====== Pydantic Models for API Request/Response Bodies ======
# =============================================================================

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_query: str
    session_id: str
    chat_history_json: str = "[]"
    session_title: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    context_used: str
    source_filenames: List[str]
    retrieved_metadata: Dict
    session_id: str
    response_time_ms: Optional[int] = None
    response_char_length: Optional[int] = None
    status: str = "success"
    message: ClassVar[str] = 'Operation completed successfully.'

class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict] = None

class DocumentStatus(BaseModel):
    file_path: str
    status: bool
    message: str

# --- MODIFIED: Document Processing with Orchestration ---
async def process_document_with_services(file_bytes: bytes, original_filename: str, file_content_type: Optional[str]) -> Tuple[bool, str]:
    """
    Orchestrates document processing using Layout Detector and OCR services.
    Returns (success_status, message).
    Includes circuit breakers for microservice calls.
    """
    logger.info(f"Starting orchestration for '{original_filename}' (Type: {file_content_type})",
                extra={"original_filename_log": original_filename, "file_type": file_content_type}) 
    
    all_extracted_text_parts = []
    extracted_markdown_tables = []
    extracted_camelot_tables = [] # To store structured table data from Camelot

    # Generate a unique filename base for saving processed outputs
    filename_base = os.path.splitext(original_filename)[0]
    unique_id_part = uuid.uuid4().hex[:8]
    processed_text_filename = f"{filename_base}_{unique_id_part}.txt"
    processed_markdown_tables_filename = f"{filename_base}_{unique_id_part}_markdown_tables.json"
    processed_camelot_tables_filename = f"{filename_base}_{unique_id_part}_structured_tables.json"

    try:
        # Step 1: Call Layout Detector Service for all supported document types
        logger.info(f"Calling Layout Detector Service for '{original_filename}' at {LAYOUT_DETECTOR_SERVICE_URL}",
                    extra={"service": "LayoutDetector", "url": LAYOUT_DETECTOR_SERVICE_URL, "original_filename_log": original_filename}) 
        files = {'file': (original_filename, file_bytes, file_content_type if file_content_type else 'application/octet-stream')}
        
        @retry(wait=wait_fixed(2), stop=stop_after_attempt(3), before_sleep=before_sleep_log(logger, logging.WARNING),
               retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)))
        async def call_layout_detector():
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(LAYOUT_DETECTOR_SERVICE_URL, files=files)
                response.raise_for_status()
                return response.json()

        layout_result = await call_layout_detector()
        
        if layout_result.get("status") == "success":
            detected_blocks = layout_result.get("detected_blocks_raw", [])
            extracted_markdown_tables = layout_result.get("extracted_markdown_tables", [])
            extracted_camelot_tables = layout_result.get("extracted_camelot_tables", []) # Get structured tables
            
            logger.info(f"Layout Detector found {len(detected_blocks)} blocks for '{original_filename}'.",
                        extra={"original_filename_log": original_filename, "num_blocks": len(detected_blocks)}) 
            logger.info(f"Layout Detector extracted {len(extracted_markdown_tables)} markdown tables and {len(extracted_camelot_tables)} structured tables.",
                        extra={"original_filename_log": original_filename, "num_markdown_tables": len(extracted_markdown_tables), "num_structured_tables": len(extracted_camelot_tables)}) 

            # Determine if OCR should be skipped for this file type (e.g., HTML)
            skip_ocr_for_file = original_filename.lower().endswith(".html") or \
                                (file_content_type and "text/html" in file_content_type.lower())

            # --- Iterate through detected blocks and process for text/OCR ---
            for i, block in enumerate(detected_blocks):
                block_type = block.get("type")
                block_content = block.get("content", "").strip()
                block_coords = block.get("coordinates") # [x1, y1, x2, y2]
                block_page_number = block.get("page")

                # Always add LayoutParser's text content first if it's a text/title/list block
                if block_type in ["narrative_text", "title", "list_item", "composite", "other_element"] and block_content: 
                    all_extracted_text_parts.append(block_content)
                    logger.debug(f"Added LayoutParser text block {i} (page {block_page_number}). Content snippet: {block_content[:50]}...",
                                 extra={"block_idx": i, "page": block_page_number, "block_type": block_type, "content_snippet": block_content[:50]})
                elif block_type == "table" and block_content:
                    pass # Handled by full_extracted_text from layout service and separate table files

                # Step 2: Conditional OCR for image blocks or sparse text blocks, UNLESS skipping OCR for the file
                should_ocr_block = False
                if not skip_ocr_for_file: # Only consider OCR if not skipping for the entire file
                    if block_type == "image":
                        logger.info(f"Image block {i} detected on page {block_page_number}. Triggering OCR service for this region.",
                                    extra={"block_idx": i, "page": block_page_number, "block_type": block_type})
                        should_ocr_block = True
                    elif block_type in ["narrative_text", "title", "list_item", "composite", "other_element"] and len(block_content) < MIN_CHARS_FOR_OCR_RECHECK: 
                        logger.info(f"Text block {i} (page {block_page_number}) has low character count ({len(block_content)}). Re-checking with OCR service.",
                                    extra={"block_idx": i, "page": block_page_number, "char_count": len(block_content)})
                        should_ocr_block = True

                if should_ocr_block and block_coords:
                    try:
                        # Send the original file bytes and the block coordinates to the OCR service
                        ocr_files = {'file': (original_filename, file_bytes, file_content_type)}
                        # Convert coordinates object to a comma-separated string if it's an object from layout_detector_service
                        if isinstance(block_coords, dict) and 'points' in block_coords:
                            coords_str = ','.join(map(str, [p for sublist in block_coords['points'] for p in sublist]))
                        elif isinstance(block_coords, list) and len(block_coords) == 4: # Already [x1, y1, x2, y2]
                            coords_str = ','.join(map(str, block_coords))
                        else:
                            logger.warning(f"Invalid or unexpected block_coords format for block {i}: {block_coords}. Skipping OCR for this block.",
                                           extra={"block_idx": i, "block_coords_log": block_coords}) # Changed 'block_coords' to 'block_coords_log'
                            continue # Skip OCR if coordinates are malformed

                        ocr_data = {'coordinates': coords_str}
                        
                        @retry(wait=wait_fixed(2), stop=stop_after_attempt(3), before_sleep=before_sleep_log(logger, logging.WARNING),
                               retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)))
                        async def call_ocr_service():
                            async with httpx.AsyncClient(timeout=600.0) as client_ocr:
                                response = await client_ocr.post(OCR_SERVICE_URL, files=ocr_files, data=ocr_data)
                                response.raise_for_status()
                                return response.json()

                        ocr_result = await call_ocr_service()

                        if ocr_result.get("status") == "success" and ocr_result.get("extracted_text"):
                            ocr_text = ocr_result["extracted_text"].strip()
                            if ocr_text:
                                all_extracted_text_parts.append(f"\n[OCR EXTRACTED - Page {block_page_number if block_page_number else 'Unknown'}]\n{ocr_text}\n[/OCR EXTRACTED]\n")
                                logger.info(f"Successfully OCR'd block {i} (page {block_page_number}). Extracted {len(ocr_text)} chars.",
                                            extra={"block_idx": i, "page": block_page_number, "extracted_char_count": len(ocr_text)})
                        else:
                            logger.warning(f"OCR service returned no text or failed for block {i} (page {block_page_number}): {ocr_result.get('message', 'Unknown error')}",
                                           extra={"block_idx": i, "page": block_page_number, "ocr_response_message": ocr_result.get('message', 'Unknown error')}) # Changed 'ocr_message' to 'ocr_response_message'
                    except httpx.HTTPStatusError as e:
                        logger.error(f"HTTP error from OCR service for block {i}: {e.response.status_code} - {e.response.text}",
                                     extra={"block_idx": i, "status_code": e.response.status_code, "response_text": e.response.text})
                    except Exception as e:
                        logger.error(f"Error calling OCR service for block {i}: {e}",
                                     extra={"block_idx": i, "error_detail": str(e)}) # Changed 'error' to 'error_detail'
                        traceback.print_exc()
                elif should_ocr_block and not block_coords:
                    logger.warning(f"Skipping OCR for block {i} (type: {block_type}) because no coordinates were provided.",
                                   extra={"block_idx": i, "block_type": block_type, "reason": "no_coordinates"})

            # Combine all text parts (from LayoutParser and conditional OCR)
            final_extracted_full_text = "\n\n".join(all_extracted_text_parts).strip()

            # Save extracted text
            text_output_path = os.path.join(OUTPUT_PROCESSED_TEXT_FOLDER, processed_text_filename)
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(final_extracted_full_text)
            logger.info(f"Final extracted text saved to: {text_output_path}", extra={"output_path": text_output_path})

            # Save extracted markdown tables (if any)
            if extracted_markdown_tables:
                markdown_tables_output_path = os.path.join(OUTPUT_PROCESSED_TABLES_FOLDER, processed_markdown_tables_filename)
                with open(markdown_tables_output_path, "w", encoding="utf-8") as tables_file:
                    json.dump(extracted_markdown_tables, tables_file, ensure_ascii=False, indent=4)
                logger.info(f"Extracted markdown tables saved to: {markdown_tables_output_path}", extra={"output_path": markdown_tables_output_path})
            
            # Save extracted structured tables (if any, from Camelot)
            if extracted_camelot_tables:
                camelot_tables_output_path = os.path.join(OUTPUT_PROCESSED_TABLES_FOLDER, processed_camelot_tables_filename)
                with open(camelot_tables_output_path, "w", encoding="utf-8") as tables_file:
                    json.dump(extracted_camelot_tables, tables_file, ensure_ascii=False, indent=4)
                logger.info(f"Extracted structured (Camelot) tables saved to: {camelot_tables_output_path}", extra={"output_path": camelot_tables_output_path})
            
            return True, f"Document '{original_filename}' processed successfully by layout and OCR services. Outputs: {processed_text_filename}, {processed_markdown_tables_filename} (if any), {processed_camelot_tables_filename} (if any)."

        else:
            message = layout_result.get('message', 'Unknown error from layout detector service')
            logger.error(f"Layout detector service returned an error for '{original_filename}': {message}",
                         extra={"original_filename_log": original_filename, "layout_service_message": message}) 
            return False, f"Layout detection failed for '{original_filename}': {message}"

    except httpx.TimeoutException:
        message = f"Layout Detector or OCR service request timed out for '{original_filename}'."
        logger.error(message, extra={"original_filename_log": original_filename, "error_type": "Timeout"}) 
        return False, message
    except httpx.ConnectError as e:
        message = f"Could not connect to layout detector or OCR service: {e}. Are they running on http://localhost:8001 and http://localhost:8002?"
        logger.error(message, extra={"original_filename_log": original_filename, "error_type": "ConnectionError", "error_detail": str(e)}) 
        return False, message
    except httpx.HTTPStatusError as e:
        message = f"HTTP error from layout detector or OCR service for '{original_filename}': {e.response.status_code} - {e.response.text}"
        logger.error(message, extra={"original_filename_log": original_filename, "error_type": "HTTPStatusError", "status_code": e.response.status_code, "response_text": e.response.text}) 
        return False, message
    except Exception as e:
        message = f"Unexpected error during orchestrated document processing for '{original_filename}': {e}"
        logger.error(message, extra={"original_filename_log": original_filename, "error_type": "UnexpectedError", "error_detail": str(e)}) 
        traceback.print_exc()
        return False, message

# =============================================================================
# ====== API ENDPOINTS ======
# =============================================================================

@app.get("/api/health", response_model=StatusResponse, summary="Get health status of the API and RAG system")
async def get_health_status():
    """Returns the health status of the FastAPI application and RAG system components."""
    status_message = "API is running."
    rag_status = "Not Ready"
    details = {}

    # Check RAG system status
    if rag.SYSTEM_READY:
        rag_status = "Ready"
        details["chroma_db_path"] = rag.CHROMA_DB_PATH
        details["ollama_api_url"] = rag.OLLAMA_API_URL
        details["ollama_embedding_model"] = rag.EMBEDDING_MODEL_NAME
        details["reranker_model"] = rag.RERANKER_MODEL_NAME
        try:
            details["chroma_document_count"] = rag.collection.count() if rag.collection else 0
            service_health_status["chroma_db"]["status"] = "healthy"
            service_health_status["chroma_db"]["error_count"] = 0
        except Exception as e:
            details["chroma_document_count"] = "Error"
            service_health_status["chroma_db"]["status"] = "unhealthy"
            service_health_status["chroma_db"]["error_count"] += 1
            logger.error(f"ChromaDB health check failed: {e}", extra={"service_name": "ChromaDB", "error_detail": str(e)})
    else:
        status_message += " RAG system not fully initialized. Check logs for details."
        details["initialization_errors"] = rag.INITIALIZATION_ERRORS
        service_health_status["chroma_db"]["status"] = "unhealthy" # If RAG not ready, Chroma is likely part of it

    # Check Redis status
    if redis_client:
        try:
            redis_client.ping()
            details["redis_status"] = "Connected"
            service_health_status["redis_cache"]["status"] = "healthy"
            service_health_status["redis_cache"]["error_count"] = 0
        except Exception as e:
            details["redis_status"] = f"Disconnected: {e}"
            service_health_status["redis_cache"]["status"] = "unhealthy"
            service_health_status["redis_cache"]["error_count"] += 1
            logger.error(f"Redis health check failed: {e}", extra={"service_name": "Redis", "error_detail": str(e)})
    else:
        details["redis_status"] = "Not configured or failed to connect"
        service_health_status["redis_cache"]["status"] = "disabled"

    # Check Ollama LLM service status
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            ollama_llm_response = await client.post(rag.OLLAMA_API_URL, json={"model": rag.OLLAMA_MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]}, timeout=5)
            ollama_llm_response.raise_for_status()
            details["ollama_llm_status"] = "Connected"
            service_health_status["ollama_llm"]["status"] = "healthy"
            service_health_status["ollama_llm"]["error_count"] = 0
    except Exception as e:
        details["ollama_llm_status"] = f"Disconnected: {e}"
        service_health_status["ollama_llm"]["status"] = "unhealthy"
        service_health_status["ollama_llm"]["error_count"] += 1
        logger.error(f"Ollama LLM health check failed: {e}", extra={"service_name": "Ollama_LLM", "error_detail": str(e)})

    # Check Ollama Embeddings service status
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            ollama_embed_response = await client.post(rag.OLLAMA_EMBED_API_URL, json={"model": rag.EMBEDDING_MODEL_NAME, "prompt": "test"}, timeout=5)
            ollama_embed_response.raise_for_status()
            details["ollama_embeddings_status"] = "Connected"
            service_health_status["ollama_embeddings"]["status"] = "healthy"
            service_health_status["ollama_embeddings"]["error_count"] = 0
    except Exception as e:
        details["ollama_embeddings_status"] = f"Disconnected: {e}"
        service_health_status["ollama_embeddings"]["status"] = "unhealthy"
        service_health_status["ollama_embeddings"]["error_count"] += 1
        logger.error(f"Ollama Embeddings health check failed: {e}", extra={"service_name": "Ollama_Embeddings", "error_detail": str(e)})

    # Check Layout Detector Service status
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            layout_response = await client.get(LAYOUT_DETECTOR_SERVICE_URL.replace("/detect_layout", "/health"), timeout=5) # Assuming health endpoint
            layout_response.raise_for_status()
            details["layout_detector_service_status"] = "Connected"
            service_health_status["layout_detector_service"]["status"] = "healthy"
            service_health_status["layout_detector_service"]["error_count"] = 0
    except Exception as e:
        details["layout_detector_service_status"] = f"Disconnected: {e}"
        service_health_status["layout_detector_service"]["status"] = "unhealthy"
        service_health_status["layout_detector_service"]["error_count"] += 1
        logger.error(f"Layout Detector Service health check failed: {e}", extra={"service_name": "LayoutDetector", "error_detail": str(e)})

    # Check OCR Service status
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            ocr_response = await client.get(OCR_SERVICE_URL.replace("/process_document", "/health"), timeout=5) # Assuming health endpoint
            ocr_response.raise_for_status()
            details["ocr_service_status"] = "Connected"
            service_health_status["ocr_service"]["status"] = "healthy"
            service_health_status["ocr_service"]["error_count"] = 0
    except Exception as e:
        details["ocr_service_status"] = f"Disconnected: {e}"
        service_health_status["ocr_service"]["status"] = "unhealthy"
        service_health_status["ocr_service"]["error_count"] += 1
        logger.error(f"OCR Service health check failed: {e}", extra={"service_name": "OCR", "error_detail": str(e)})

    # Update last checked timestamp for all services
    current_time = datetime.now(timezone.utc).isoformat()
    for service in service_health_status:
        service_health_status[service]["last_checked"] = current_time
        # Add to details for the response
        details[f"{service}_status"] = service_health_status[service]["status"]
        details[f"{service}_last_checked"] = service_health_status[service]["last_checked"]
        details[f"{service}_error_count"] = service_health_status[service]["error_count"]


    return StatusResponse(status="success", message=status_message, details={"rag_system": rag_status, **details})


@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/register", status_code=status.HTTP_201_CREATED, response_model=StatusResponse)
async def register_user(user_create: UserCreate, current_user: User = Depends(get_current_admin_user)):
    try:
        success, message = auth.register_user(user_create.username, user_create.password, user_create.role)
        if success:
            return StatusResponse(status="success", message=message)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    except Exception as e:
        logger.error(f"User registration failed: {e}", extra={"username_log": user_create.username, "error_detail": str(e)}) # Changed 'username' to 'username_log' and 'error' to 'error_detail'
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"User registration failed: {e}")

@app.get("/api/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/api/uploadfile/", response_model=StatusResponse)
async def upload_file(files: List[UploadFile], current_user: User = Depends(get_current_admin_user)):
    """
    Uploads raw PDF, TXT, Image, or HTML files.
    PDFs, Images, and HTML are sent to the Layout Detector service for processing.
    TXT/CSV files are saved directly.
    """
    uploaded_files_info = []
    for file_obj in files:
        # Generate a unique filename for the raw storage, preserving original extension
        unique_raw_filename = f"{uuid.uuid4()}{os.path.splitext(file_obj.filename)[1].lower()}"
        raw_file_location = os.path.join(INPUT_RAW_DOCS_FOLDER, unique_raw_filename)
        
        try:
            file_bytes = await file_obj.read() # Read file content once

            # Save the raw file for record-keeping
            with open(raw_file_location, "wb") as f_raw:
                f_raw.write(file_bytes)
            logger.info(f"Raw file '{file_obj.filename}' (saved as '{unique_raw_filename}') saved to {INPUT_RAW_DOCS_FOLDER}",
                        extra={"original_filename_log": file_obj.filename, "saved_as": unique_raw_filename, "path": INPUT_RAW_DOCS_FOLDER}) 

            file_extension = os.path.splitext(file_obj.filename)[1].lower()
            
            # MODIFIED: Include .html in the list of files processed via services
            if file_extension in [".pdf", ".png", ".jpg", ".jpeg", ".html"]:
                # Process PDFs, images, and HTML through the new orchestrated service call
                success, message = await process_document_with_services(file_bytes, file_obj.filename, file_obj.content_type)
                uploaded_files_info.append(DocumentStatus(file_path=file_obj.filename, status=success, message=message))
                
                # Move the raw file to archive if processed successfully
                if success:
                    shutil.move(raw_file_location, os.path.join(RAW_DOCS_ARCHIVE_FOLDER, unique_raw_filename))
                    logger.info(f"Raw file '{unique_raw_filename}' moved to processed_archive.",
                                extra={"original_filename_log": unique_raw_filename, "destination": RAW_DOCS_ARCHIVE_FOLDER}) 
            elif file_extension in [".txt", ".csv"]:
                # For TXT/CSV, save directly to processed/text or processed/tables
                if file_extension == ".txt":
                    processed_output_path = os.path.join(OUTPUT_PROCESSED_TEXT_FOLDER, f"{os.path.splitext(file_obj.filename)[0]}_{uuid.uuid4().hex[:8]}.txt")
                else: # .csv
                    processed_output_path = os.path.join(OUTPUT_PROCESSED_TABLES_FOLDER, f"{os.path.splitext(file_obj.filename)[0]}_{uuid.uuid4().hex[:8]}.csv")

                with open(processed_output_path, "wb") as f_processed:
                    f_processed.write(file_bytes)
                
                message = f"File '{file_obj.filename}' saved directly to processed folder."
                uploaded_files_info.append(DocumentStatus(file_path=file_obj.filename, status=True, message=message))
                logger.info(message, extra={"original_filename_log": file_obj.filename, "destination": processed_output_path}) 
                # Move raw file to archive after direct saving
                shutil.move(raw_file_location, os.path.join(RAW_DOCS_ARCHIVE_FOLDER, unique_raw_filename))
                logger.info(f"Raw file '{unique_raw_filename}' moved to processed_archive.",
                            extra={"original_filename_log": unique_raw_filename, "destination": RAW_DOCS_ARCHIVE_FOLDER}) 
            else:
                # MODIFIED: Updated error message to include HTML
                error_msg = f"Unsupported file type: {file_obj.filename}. Only PDF, TXT, CSV, PNG, JPG, JPEG, HTML are allowed."
                uploaded_files_info.append(DocumentStatus(file_path=file_obj.filename, status=False, message=error_msg))
                logger.warning(error_msg, extra={"original_filename_log": file_obj.filename, "reason": "unsupported_type"}) 

        except Exception as e:
            error_msg = f"Failed to upload or process {file_obj.filename}: {e}"
            logger.error(error_msg, extra={"original_filename_log": file_obj.filename, "error_detail": str(e)}) 
            traceback.print_exc()
            uploaded_files_info.append(DocumentStatus(file_path=file_obj.filename, status=False, message=error_msg))
    
    successful_uploads = [res for res in uploaded_files_info if res.status]
    failed_uploads = [res for res in uploaded_files_info if not res.status]

    if successful_uploads or failed_uploads:
        overall_status = "success" if not failed_uploads else "warning"
        overall_message = f"Upload complete. {len(successful_uploads)} succeeded, {len(failed_uploads)} failed."
        logger.info(overall_message, extra={"successful_uploads_count": len(successful_uploads), "failed_uploads_count": len(failed_uploads)}) # Changed keys
        return StatusResponse(status=overall_status, message=overall_message, details={"results": [res.dict() for res in uploaded_files_info]})
    else:
        logger.warning("No files provided for upload.", extra={"event_type": "no_files_uploaded"}) # Changed key
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided for upload.")

@app.post("/api/ingest_processed_documents/", response_model=StatusResponse)
async def ingest_processed_documents_to_rag(current_user: User = Depends(get_current_admin_user)):
    """
    Triggers the ingestion of all processed documents (TXT/CSV/JSON) from the
    'docs_processed' folder (including its 'text' and 'tables' subfolders)
    into the RAG knowledge base.
    Only callable by 'admin' users.
    """
    logger.info(f"Admin user '{current_user.username}' triggering ingestion of processed documents.",
                extra={"triggering_user": current_user.username}) # Changed key
    if not rag.SYSTEM_READY:
        logger.error("RAG system is not ready for ingestion.", extra={"rag_system_status": "RAG_not_ready"}) # Changed key
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG system is not ready for ingestion.")
    try:
        rag.ingest_txt_documents_from_folder(PROCESSED_DOCS_FOLDER, username="company_data")
        return StatusResponse(status="success", message="Ingestion of processed documents triggered successfully. Check server logs for details.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", extra={"error_detail": str(e)}) # Changed key
        raise HTTPException(status_code=500, detail=f"Failed to trigger ingestion: {e}")

@app.delete("/api/clear_documents/", response_model=StatusResponse)
async def clear_documents(current_user: User = Depends(get_current_admin_user)):
    """Clears all documents from the RAG knowledge base. Admin only."""
    try:
        if rag.collection:
            # Only delete documents associated with 'company_data' or the current user
            rag.collection.delete(where={"$or": [{"user": "company_data"}, {"user": current_user.username}]})
            logger.info(f"Documents cleared for user 'company_data' and '{current_user.username}'.",
                        extra={"clearing_user": current_user.username, "action": "clear_documents"}) # Changed key
        else:
            logger.error("Vector store collection not initialized.", extra={"chroma_status": "Chroma_not_initialized"}) # Changed key
            raise HTTPException(status_code=500, detail="Vector store collection not initialized.")
        return StatusResponse(status="success", message="All relevant documents cleared from the knowledge base.")
    except Exception as e:
        logger.error(f"Error clearing documents: {e}", extra={"clearing_user": current_user.username, "error_detail": str(e)}) # Changed key
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {e}")

from fastapi.responses import StreamingResponse
import asyncio

@app.post("/api/chat")
async def chat_with_rag(chat_request: ChatRequest, current_user: User = Depends(get_current_user)):
    """Handles user chat queries, retrieves context from RAG, and streams AI responses."""
    try:
        start_time = time.time()
        user_query = chat_request.user_query
        session_title = chat_request.session_title
        current_username = current_user.username

        session_id = chat_request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session_id: {session_id} for user '{current_username}'",
                        extra={"chat_user": current_username, "session_id_log": session_id, "event_type": "new_session"}) # Changed keys

        if rag.is_rate_limited(current_username):
            logger.warning(f"Rate limit exceeded for user '{current_username}'.", extra={"rate_limited_user": current_username, "event_type": "rate_limited"}) # Changed keys
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Please try again later.")

        cache_key = f"rag:{user_query.lower()}:{current_username}:{session_id}"
        if redis_client:
            try:
                if redis_client.exists(cache_key):
                    cached = redis_client.get(cache_key)
                    logger.info(f"Cache hit for query: '{user_query[:50]}'",
                                extra={"cache_key_log": cache_key, "query_snippet_log": user_query[:50], "event_type": "cache_hit"}) # Changed keys
                    cached_response_data = json.loads(cached)
                    return ChatResponse(
                        response=cached_response_data.get("response", ""),
                        context_used=cached_response_data.get("context_used", ""),
                        source_filenames=cached_response_data.get("source_filenames", []),
                        retrieved_metadata=cached_response_data.get("retrieved_metadata", {}),
                        session_id=session_id,
                        response_time_ms=cached_response_data.get("response_time_ms"),
                        response_char_length=cached_response_data.get("response_char_length"),
                        status=cached_response_data.get("status", "success"),
                        message=cached_response_data.get("message", "Response from cache.")
                    )
            except redis.exceptions.ConnectionError as e:
                logger.warning(f"Redis connection error during cache check: {e}. Proceeding without cache.",
                               extra={"service_name": "Redis", "error_detail": str(e), "event_type": "cache_error"}) # Changed keys
            except Exception as e:
                logger.warning(f"Error retrieving from cache: {e}. Proceeding without cache.",
                               extra={"cache_key_log": cache_key, "error_detail": str(e), "event_type": "cache_retrieval_error"}) # Changed keys

        auth.log_chat_message_to_db(current_username, session_id, "user", user_query, session_title)

        chat_history_list = []
        try:
            if chat_request.chat_history_json:
                parsed_history = json.loads(chat_request.chat_history_json)
                if not isinstance(parsed_history, list):
                    raise ValueError("chat_history_json must be a JSON string representing a list.")
                for item in parsed_history:
                    role = item.get("role")
                    content = item.get("content") or item.get("message")
                    if role and content:
                        chat_history_list.append({"role": role, "content": content})
                    else:
                        logger.warning(f"Malformed history item (missing role or content/message): {item}. Skipping.",
                                       extra={"history_item_data": item, "event_type": "malformed_chat_history"}) # Changed key
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON for chat_history_json: {chat_request.chat_history_json}. Error: {e}. Proceeding with empty history.",
                           extra={"chat_history_json_snippet": chat_request.chat_history_json[:100], "error_detail": str(e), "event_type": "json_decode_error"}) # Changed key
        except ValueError as e:
            logger.warning(f"Error parsing chat_history_json: {e}. Proceeding with empty history.",
                           extra={"error_detail": str(e), "event_type": "chat_history_parsing_error"}) # Changed key

        import re
        generic_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        greeting_pattern = r"^(?:" + "|".join([re.escape(g) for g in generic_greetings]) + r")[\s!.,?]*$"
        is_greeting = re.match(greeting_pattern, user_query.strip(), re.IGNORECASE) is not None

        # --- Accumulate all streamed chunks and return a single JSON object ---
        # --- Normal synchronous RAG response block ---
        source_filenames = []
        retrieved_metadata_dict = {}

        if is_greeting:
            llm_response = "Hello! I am your Pyrotech AI Assistant, specialized in providing information from your documents. How can I help you today?"
            auth.log_chat_message_to_db(current_username, session_id, "assistant", llm_response, session_title)

            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            chat_response = ChatResponse(
                response=llm_response,
                context_used="",
                source_filenames=[],
                session_id=session_id,
                retrieved_metadata={},
                response_time_ms=response_time_ms,
                response_char_length=len(llm_response),
                status="success",
                message="Greeting response generated."
            )

            if redis_client:
                try:
                    cache_content = chat_response.model_dump_json() if hasattr(chat_response, 'model_dump_json') else chat_response.json()
                    redis_client.setex(cache_key, 3600, cache_content.encode('utf-8'))
                    logger.info(f"Cached greeting response for key: '{cache_key}'",
                                extra={"cache_key_log": cache_key, "event_type": "cache_set_greeting"}) # Changed key
                except redis.exceptions.ConnectionError as e:
                    logger.warning(f"Redis connection error during cache set: {e}. Response not cached.",
                                   extra={"service_name": "Redis", "error_detail": str(e), "event_type": "cache_error"}) # Changed key
                except Exception as e:
                    logger.warning(f"Error setting cache for greeting: {e}.",
                                   extra={"cache_key_log": cache_key, "error_detail": str(e), "event_type": "cache_set_error"}) # Changed key
            return chat_response

        # --- Attribute Extraction (LLM) ---
        from rag import _extract_and_process_attributes_improved
        attribute_start_time = time.time()
        llm_call_fn = timed_cached_llm_call_fn
        query_attributes = _extract_and_process_attributes_improved(
            prompt_template="Extract relevant query attributes as JSON:",
            input_text=user_query,
            source_description="user_query",
            doc_type="default",
            llm_call_fn=llm_call_fn
        )
        import ast
        if isinstance(query_attributes, str):
            try:
                query_attributes = ast.literal_eval(query_attributes)
            except Exception as e:
                logger.warning(f"Failed to convert stringified query_attributes to dict: {e}")
                query_attributes = {}
        attribute_end_time = time.time()
        logger.info(f"Query attribute extraction took {int((attribute_end_time-attribute_start_time)*1000)} ms. Extracted: {query_attributes}")
        def rerank_fn(query, docs, embeddings=None):
            return rag.rerank_documents(query, docs)
        # --- Retrieval ---
        if not embedding_function_instance:
            raise HTTPException(status_code=503, detail="Embedding function not available")
            
        context, _, _, source_filenames, retrieved_metadata_dict = rag.retrieve_context_from_vector_db(
            query=user_query,
            query_attributes=query_attributes,
            collection=rag.collection,
            embedding_fn=embedding_function_instance,
            rerank_fn=rerank_fn,
            llm_call_fn=llm_call_fn,
            current_username=current_username,
        )

        final_messages_for_llm = rag.build_llm_prompt(context, user_query, chat_history_list)
        llm_prompt = final_messages_for_llm
        if isinstance(llm_prompt, list):
            llm_prompt = "\n".join(str(x) for x in llm_prompt)
        llm_response = llm_call_fn(llm_prompt)

        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        auth.log_chat_message_to_db(current_username, session_id, "assistant", llm_response, session_title)

        chat_response = ChatResponse(
            response=llm_response,
            context_used=context,
            source_filenames=source_filenames,
            session_id=session_id,
            retrieved_metadata=retrieved_metadata_dict,
            response_time_ms=response_time_ms,
            response_char_length=len(llm_response),
            status="success",
            message="Response generated."
        )

        if redis_client:
            try:
                cache_content = chat_response.model_dump_json() if hasattr(chat_response, 'model_dump_json') else chat_response.json()
                redis_client.setex(cache_key, 3600, cache_content.encode('utf-8'))
                logger.info(f"Cached response for key: '{cache_key}'",
                            extra={"cache_key_log": cache_key, "event_type": "cache_set_chat_response"}) # Changed key
            except redis.exceptions.ConnectionError as e:
                logger.warning(f"Redis connection error during cache set: {e}. Response not cached.",
                               extra={"service_name": "Redis", "error_detail": str(e), "event_type": "cache_error"}) # Changed key
            except Exception as e:
                logger.warning(f"Error setting cache: {e}.",
                               extra={"cache_key_log": cache_key, "error_detail": str(e), "event_type": "cache_set_error"}) # Changed key
        return chat_response

    except Exception as e:
        logger.error(f"Chat endpoint failed for user '{current_username}', session '{session_id}': {e}",
                     extra={"chat_user": current_username, "session_id_log": session_id, "error_detail": str(e), "event_type": "chat_failure"}) # Changed keys
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred during chat processing. Check backend logs.")

@app.get("/api/sessions", response_model=List[SessionSummary], summary="List all chat sessions for the current user")
async def list_sessions(current_user: User = Depends(get_current_user)):
    return auth.get_user_chat_sessions(current_user.username)

@app.get("/api/sessions/{session_id}", response_model=SessionMessages, summary="Get full chat history for a specific session")
async def get_session_history(session_id: str, current_user: User = Depends(get_current_user)):
    messages_from_db = auth.get_chat_messages_for_session(current_user.username, session_id)
    if not messages_from_db:
        logger.warning(f"Session '{session_id}' not found or no messages for user '{current_user.username}'.",
                       extra={"session_id_log": session_id, "user_log": current_user.username, "event_type": "session_not_found"}) # Changed keys
        raise HTTPException(status_code=404, detail="Session not found or no messages for this session.")
    
    session_title = None
    for msg in messages_from_db:
        if msg.get("session_title"):
            session_title = msg["session_title"]
            break
    if not session_title:
        session_title = f"Session {session_id[:8]}" 
    return SessionMessages(session_id=session_id, session_title=session_title, messages=messages_from_db)

@app.delete("/api/sessions/{session_id}", response_model=StatusResponse, summary="Delete a specific chat session for the current user")
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    is_admin = current_user.role == "admin"
    success = auth.delete_user_session(
        session_id=session_id, 
        username=current_user.username,
        is_admin=is_admin
    )
    if success:
        logger.info(f"Session '{session_id}' deleted by user '{current_user.username}'.",
                    extra={"session_id_log": session_id, "user_log": current_user.username, "action_type": "delete_session"}) # Changed keys
        return StatusResponse(status="success", message=f"Session '{session_id}' deleted.")
    else:
        logger.warning(f"Failed to delete session '{session_id}' for user '{current_user.username}'. Unauthorized or session not found.",
                       extra={"session_id_log": session_id, "user_log": current_user.username, "action_type": "delete_session_failed"}) # Changed keys
        raise HTTPException(status_code=403, detail="Unauthorized or session not found. You can only delete your own sessions, or contact admin for others.")

@app.get("/api/admin/all_chat_logs", response_model=Dict[str, List[AdminChatLogEntry]], summary="List all chat logs in the system (Admin Only)")
async def list_all_chat_logs(admin_user: User = Depends(get_current_admin_user)):
    logs = auth.get_all_chat_message_logs()
    return {"logs": logs}

@app.delete("/api/admin/clear_old_chat_logs", response_model=StatusResponse, summary="Delete chat logs older than N days (Admin Only)")
async def clear_old_chat_logs_api(days: int = Query(30, ge=1, description="Number of days to keep chat logs (older logs will be deleted)"),
                               current_user: User = Depends(get_current_admin_user)):
    if days < 0:
        logger.warning(f"Attempted to clear chat logs with negative days: {days}.", extra={"days_param": days}) # Changed key
        raise HTTPException(status_code=400, detail="Days must be a non-negative number.")
    success = auth.delete_old_chat_logs(days)
    if success:
        logger.info(f"Successfully deleted chat logs older than {days} days.", extra={"days_deleted": days, "action_type": "clear_old_logs"}) # Changed keys
        return StatusResponse(status="success", message=f"Successfully deleted chat logs older than {days} days.")
    else:
        logger.error(f"Failed to delete old chat logs older than {days} days.", extra={"days_failed": days, "action_type": "clear_old_logs_failed"}) # Changed keys
        raise HTTPException(status_code=500, detail=f"Failed to delete old chat logs.")

@app.delete("/api/admin/delete_user_session_admin", response_model=StatusResponse, summary="Admin: Delete a specific user's session")
async def delete_user_session_admin_endpoint(
    session_id: str = Query(..., description="Session ID to delete"),
    username: str = Query(..., description="Username of the user whose session is to be deleted"),
    current_user: User = Depends(get_current_admin_user)
):
    success = auth.delete_user_session(session_id=session_id, username=username, is_admin=True)
    if success:
        logger.info(f"Admin '{current_user.username}' deleted session '{session_id}' for user '{username}'.",
                    extra={"admin_user_log": current_user.username, "target_user_log": username, "session_id_log": session_id, "action_type": "admin_delete_session"}) # Changed keys
        return StatusResponse(status="success", message=f"Session '{session_id}' for user '{username}' deleted successfully by admin.")
    else:
        logger.warning(f"Admin '{current_user.username}' failed to delete session '{session_id}' for user '{username}'. Session not found.",
                       extra={"admin_user_log": current_user.username, "target_user_log": username, "session_id_log": session_id, "action_type": "admin_delete_session_failed"}) # Changed keys
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found for user '{username}' or deletion failed.")

@app.get("/api/users", response_model=Dict[str, List[Dict]], summary="List all users in the system (Admin Only)")
async def list_all_users(admin_user: User = Depends(get_current_admin_user)):
    users = []
    try:
        with sqlite3.connect(auth.USERS_DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT username, role FROM users ORDER BY username ASC")
            for row in cursor.fetchall():
                users.append({"username": row["username"], "role": row["role"]})
    except Exception as e:
        logger.error(f"Error retrieving all users: {e}", extra={"error_detail": str(e)}) # Changed key
        traceback.print_exc()
    return {"users": users}

# New API endpoints for admin panel stats
@app.get("/api/users/count", response_model=Dict[str, int], summary="Get total user count (Admin Only)")
async def get_user_count(admin_user: User = Depends(get_current_admin_user)):
    try:
        with sqlite3.connect(auth.USERS_DB) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting user count: {e}", extra={"error_detail": str(e)}) # Changed key
        raise HTTPException(status_code=500, detail=f"Failed to get user count: {e}")

@app.get("/api/documents/count", response_model=Dict[str, int], summary="Get total documents processed count (Admin Only)")
async def get_document_count(admin_user: User = Depends(get_current_admin_user)):
    try:
        if rag.collection:
            count = rag.collection.count()
            return {"count": count}
        else:
            return {"count": 0}
    except Exception as e:
        logger.error(f"Error getting document count from ChromaDB: {e}", extra={"error_detail": str(e)}) # Changed key
        raise HTTPException(status_code=500, detail=f"Failed to get document count: {e}")

@app.get("/api/analytics/api_calls_today", response_model=Dict[str, int], summary="Get count of API calls today (Admin Only)")
async def get_api_calls_today(admin_user: User = Depends(get_current_admin_user)):
    try:
        today = datetime.now().date()
        count = 0
        with sqlite3.connect(auth.CHAT_LOGS_DB) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM chat_message_logs WHERE role = 'user' AND DATE(timestamp) = ?",
                (today.isoformat(),)
            )
            count = cursor.fetchone()[0]
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting API calls today: {e}", extra={"error_detail": str(e)}) # Changed key
        raise HTTPException(status_code=500, detail=f"Failed to get API calls today: {e}")

@app.get("/api/sessions/active", response_model=Dict[str, int], summary="Get count of active sessions (Admin Only)")
async def get_active_sessions_count(admin_user: User = Depends(get_current_admin_user)):
    try:
        # Define "active" as sessions with messages in the last 24 hours
        time_threshold = datetime.now(timezone.utc) - timedelta(hours=24)
        count = 0
        with sqlite3.connect(auth.CHAT_LOGS_DB) as conn:
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM chat_message_logs WHERE timestamp >= ?",
                (time_threshold.isoformat(),)
            )
            count = cursor.fetchone()[0]
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting active sessions count: {e}", extra={"error_detail": str(e)}) # Changed key
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions count: {e}")


# ==============================================================================
# ====== APPLICATION LIFECYCLE EVENTS (Startup/Shutdown) ======
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*30 + " Initializing Pyrotech FastAPI RAG Backend " + "="*30)
    logger.info("Initializing Pyrotech FastAPI RAG Backend startup event.")
    
    os.makedirs(auth.DB_DIR, exist_ok=True)

    try:
        users_table_schema = """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user'
            )
        """
        auth.init_sqlite_db(auth.USERS_DB, users_table_schema)

        if not auth.get_user("admin"):
            auth.register_user("admin", "jahnavi@123", "admin")
            logger.info("Default admin user 'admin' created with password 'jahnavi@123'.")
        else:
            logger.info("Default admin user 'admin' already exists.")

        chat_logs_table_schema = """
            CREATE TABLE IF NOT EXISTS chat_message_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_id TEXT NOT NULL,
                session_title TEXT,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        auth.init_sqlite_db(auth.CHAT_LOGS_DB, chat_logs_table_schema)

    except Exception as e:
        logger.critical(f"FATAL: One or more database initializations failed: {e}", extra={"error_detail": str(e)}) # Changed key
        traceback.print_exc()

    logger.info("Initializing RAG components...")
    rag.PROCESSED_DOCS_FOLDER = PROCESSED_DOCS_FOLDER
    rag.CHROMA_DB_PATH = os.path.join(DB_DIR, "chroma_db_final")
    rag.OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")
    rag.OLLAMA_EMBED_API_URL = os.getenv("OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings") # Pass embedding URL
    rag.OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', "mistral")
    rag.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
    
    rag.init_rag_system()
    # rag.init_rerank_cache_db() # This is now called inside rag.init_rag_system()

    logger.info("Checking for raw documents to process in docs_raw directory...")
    raw_files_found = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # MODIFIED: Added .html to supported_raw_extensions
    supported_raw_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.txt', '.csv', '.html']

    for filename in os.listdir(INPUT_RAW_DOCS_FOLDER):
        if filename == 'processed_archive':
            continue
        
        file_path = os.path.join(INPUT_RAW_DOCS_FOLDER, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1].lower()
            raw_files_found += 1
            
            # MODIFIED: Include .html in the list of files processed via services
            if file_extension in ['.pdf', '.png', '.jpg', '.jpeg', '.html']:
                logger.info(f"Found raw file for processing via services: {filename}", extra={"original_filename_log": filename, "processing_type": "service_processing"}) # Changed key
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    
                    mime_type, _ = mimetypes.guess_type(filename)
                    
                    success, message = await process_document_with_services(file_bytes, filename, mime_type)
                    
                    if success:
                        processed_count += 1
                        logger.info(f"Successfully processed raw file '{filename}'. Moving to archive.",
                                    extra={"original_filename_log": filename, "action_status": "archive_success"}) # Changed key
                        shutil.move(file_path, os.path.join(RAW_DOCS_ARCHIVE_FOLDER, filename))
                    else:
                        failed_count += 1
                        logger.error(f"Failed to process raw file '{filename}': {message}",
                                     extra={"original_filename_log": filename, "action_status": "process_failed", "error_message_log": message}) # Changed key
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error reading or processing raw file '{filename}': {e}",
                                 extra={"original_filename_log": filename, "action_status": "read_process_error", "error_detail": str(e)}) # Changed key
                    traceback.print_exc()
            elif file_extension in ['.txt', '.csv']:
                # For TXT/CSV, directly move to processed folders
                target_folder = OUTPUT_PROCESSED_TEXT_FOLDER if file_extension == ".txt" else OUTPUT_PROCESSED_TABLES_FOLDER
                target_path = os.path.join(target_folder, filename)
                try:
                    shutil.move(file_path, target_path)
                    processed_count += 1
                    logger.info(f"Moved raw TXT/CSV file '{filename}' directly to '{target_folder}'.",
                                extra={"original_filename_log": filename, "action_type": "direct_move", "destination_path": target_folder}) # Changed key
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error moving raw TXT/CSV file '{filename}': {e}",
                                 extra={"original_filename_log": filename, "action_type": "direct_move_failed", "error_detail": str(e)}) # Changed key
                    traceback.print_exc()
            else:
                skipped_count += 1
                logger.warning(f"Skipping unsupported raw file '{filename}' (extension: {file_extension}).",
                               extra={"original_filename_log": filename, "reason_skipped": "unsupported_extension"}) # Changed key

    if raw_files_found > 0:
        logger.info(f"Raw document processing summary: Total found: {raw_files_found}, Processed: {processed_count}, Failed: {failed_count}, Skipped: {skipped_count}.",
                    extra={"total_raw_files_found": raw_files_found, "processed_count_summary": processed_count, "failed_count_summary": failed_count, "skipped_count_summary": skipped_count}) # Changed keys
    else:
        logger.info("No raw PDF/image/text documents found in docs_raw for initial processing.")

    # Ingestion logic has been removed from main.py. All ingestion must be performed via rag_ingest.py.
    print("="*30 + " Pyrotech FastAPI RAG Backend Initialized " + "="*30 + "\n")


@app.on_event("shutdown")
def shutdown_event():
    print("\n" + "="*30 + " Shutting down Pyrotech FastAPI RAG Backend " + "="*30)
    logger.info("Pyrotech FastAPI RAG Backend shutting down.")
    # Close embedding cache connection
    if rag.embedding_cache_conn:
        rag.embedding_cache_conn.close()
        logger.info("Embedding cache database connection closed.")
    # Close rerank cache connection
    if rag.rerank_cache_conn:
        rag.rerank_cache_conn.close()
        logger.info("Rerank cache database connection closed.")
    print("="*30 + " Backend Shutdown Complete " + "="*30 + "\n")


app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="static_frontend")

from rag import ask_llm_model_phi3
from functools import lru_cache
import time

def make_cached_llm_call_fn(llm_call_fn, maxsize=128):
    @lru_cache(maxsize=maxsize)
    def cached_call(prompt):
        # lru_cache requires hashable arguments; ensure prompt is a string
        if isinstance(prompt, (list, dict)):
            import json
            prompt_str = json.dumps(prompt, sort_keys=True)
        else:
            prompt_str = str(prompt)
        return llm_call_fn(prompt)
    return cached_call

def make_timed_llm_call_fn(llm_call_fn):
    def timed_call(prompt):
        start = time.perf_counter()
        result = llm_call_fn(prompt)
        duration = (time.perf_counter() - start) * 1000
        print(f"LLM call took {duration:.2f} ms for prompt: {str(prompt)[:80]}")
        return result
    return timed_call

# Compose both wrappers for maximum effect
cached_llm_call_fn = make_cached_llm_call_fn(ask_llm_model_phi3)
timed_cached_llm_call_fn = make_timed_llm_call_fn(cached_llm_call_fn)

# Pass timed_cached_llm_call_fn as llm_call_fn to all RAG/LLM functions for caching and timing
# Example usage:
# rag.retrieve_context_from_vector_db(..., timed_cached_llm_call_fn)

import atexit
from rag import close_caches
atexit.register(close_caches)

if __name__ == "__main__":
    print("\n" + "="*30 + " LAUNCHING PYROTECH FASTAPI RAG API " + "="*30)
    logger.info(f"Application base directory: {BASE_DIR}")
    logger.info(f"Raw documents upload directory: {INPUT_RAW_DOCS_FOLDER}")
    logger.info(f"Processed documents ingestion directory: {PROCESSED_DOCS_FOLDER}")
    logger.info(f"ChromaDB path: {rag.CHROMA_DB_PATH}")
    logger.info(f"Ollama API URL: {rag.OLLAMA_API_URL}")
    logger.info(f"Ollama Embedding API URL: {OLLAMA_EMBED_API_URL}") # Log embedding URL

    if POPPLER_BIN_PATH_DETECTED:
        logger.info(f"Poppler binary path being used: {POPPLER_BIN_PATH_DETECTED}")
    else:
        logger.warning("Poppler binary path not configured. PDF processing will likely fail.")

    required_env_vars = ["JWT_SECRET_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            logger.critical(f"CRITICAL ERROR: Environment variable '{var}' is not set.")
            print("Please set it (e.g., 'export JWT_SECRET_KEY=\"your_secret\"') before running.")
            exit(1)
            
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)