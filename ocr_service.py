<<<<<<< HEAD
# ocr_service.py
import os
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import logging
import traceback
from typing import List, Dict, Optional, Tuple, Any
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import re
import yaml # Already imported, but noting its use for potential config
import json # For parsing layout_data
import time # For performance logging
from functools import lru_cache # IMPROVEMENT 1: For lightweight caching
import hashlib # For file hashing for caching

# Import PaddlePaddle inference library directly
try:
    import paddle.inference as paddle_infer 
except ImportError:
    # Ensure logger is defined before use in this block as well
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)
    logger.critical("Failed to import paddle.inference. Please ensure PaddlePaddle is installed correctly.",
                    extra={"event": "paddle_inference_import_failed"})
    paddle_infer = None

# Set up logging for the OCR service (ensure this is only called once, at the very top)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models") 

POPPLER_PATH = os.getenv("POPPLER_BIN_PATH", "/usr/bin") 
if not os.path.isdir(POPPLER_PATH) or not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm")):
    logger.critical(f"Poppler binary path '{POPPLER_PATH}' not found or does not contain pdftoppm. PDF processing WILL FAIL.",
                   extra={"poppler_path": POPPLER_PATH, "event": "poppler_path_critical"})
else:
    logger.info(f"Poppler binary path found at '{POPPLER_PATH}'.", extra={"poppler_path": POPPLER_PATH})

app = FastAPI(
    title="Pyrotech OCR Microservice",
    description="Dedicated service for high-performance OCR using PaddleOCR, with enhanced image pre-processing and text post-processing.",
    version="1.5.0", # Incrementing version due to OCR quality improvements and caching
)

ocr_predictor = None # Initialize globally

# IMPROVEMENT 1: Add lightweight caching to preprocess_image
# Use a wrapper function to allow caching of PIL Image (which is not hashable)
@lru_cache(maxsize=64) # Cache up to 64 preprocessed images
def _preprocess_image_cached_wrapper(image_bytes: bytes) -> np.ndarray:
    """Wrapper to allow caching of preprocess_image using image bytes hash."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess_image_internal(pil_image)

def preprocess_image_internal(pil_image: Image.Image) -> np.ndarray:
    """
    Applies pre-processing steps to a PIL Image for improved OCR accuracy.
    Steps include: Convert to OpenCV, Deskew, Binarize + Denoise, Optional Upscale.
    Args:
        pil_image: The input image as a PIL Image object.
    Returns:
        A preprocessed OpenCV image (NumPy array).
    """
    logger.debug("Starting image pre-processing...", extra={"event": "image_preprocessing_start"})
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if img_np is None or img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
        logger.warning("Input image for preprocessing is empty or invalid.", extra={"event": "empty_image_input"})
        return img_np

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    try:
        if np.max(gray) - np.min(gray) > 10:
            coords = np.column_stack(np.where(gray < 200))
            if coords.size > 0:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle
                
                (h, w) = img_np.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_np = cv2.warpAffine(img_np, M, (w, h),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                logger.debug(f"Image deskewed by {angle:.2f} degrees.", extra={"event": "image_deskewed", "angle": angle})
            else:
                logger.debug("No significant foreground pixels found for deskewing.", extra={"event": "no_foreground_pixels_for_deskew"})
        else:
            logger.debug("Insufficient contrast for deskewing. Skipping.", extra={"event": "insufficient_contrast_for_deskew"})

    except Exception as e:
        logger.warning(f"Deskewing failed: {e}. Skipping deskew for this image.", extra={"event": "deskew_failed", "error": str(e)})

    gray_after_deskew = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    binarized = cv2.adaptiveThreshold(gray_after_deskew, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(binarized, 3)

    h, w = denoised.shape
    if min(h, w) < 1200:
        denoised = cv2.resize(denoised, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Image upscaled from {w}x{h} to {denoised.shape[1]}x{denoised.shape[0]}.",
                     extra={"event": "image_upscaled", "original_dims": f"{w}x{h}", "new_dims": f"{denoised.shape[1]}x{denoised.shape[0]}"})
    
    logger.debug("Image pre-processing complete.", extra={"event": "image_preprocessing_complete"})
    return denoised

def initialize_ocr_models():
    """Initializes only the PaddleOCR predictor."""
    global ocr_predictor

    if ocr_predictor is None:
        try:
            logger.info("Initializing PaddleOCR predictor (CPU mode)...", extra={"event": "paddleocr_init_start"})
            ocr_predictor = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, enable_mkldnn=True) 
            logger.info("PaddleOCR predictor initialized.", extra={"event": "paddleocr_init_success"})
        except Exception as e:
            logger.critical(f"Failed to initialize PaddleOCR predictor: {e}", extra={"event": "paddleocr_init_failed", "error": str(e)})
            logger.critical(traceback.format_exc())
            ocr_predictor = None
            raise

@app.on_event("startup")
async def startup_event():
    logger.info("OCR Microservice starting up...", extra={"event": "service_startup"})
    try:
        initialize_ocr_models()
        logger.info("OCR Microservice models initialized successfully.", extra={"event": "models_init_success"})
    except Exception as e:
        logger.critical(f"FATAL: OCR Microservice startup failed: {e}", extra={"event": "service_startup_failed", "error": str(e)})
        pass

@app.on_event("shutdown")
def shutdown_event():
    logger.info("OCR Microservice shutting down.", extra={"event": "service_shutdown"})

# IMPROVEMENT 1: Add lightweight caching to get_text_from_paddle_ocr_results
@lru_cache(maxsize=128) # Cache up to 128 OCR results
def _get_text_from_paddle_ocr_results_cached_wrapper(ocr_results_tuple: Tuple[Tuple]) -> str:
    """Wrapper to allow caching of get_text_from_paddle_ocr_results."""
    # Convert tuple back to list for internal processing
    ocr_results_list = [list(item) for item in ocr_results_tuple]
    return get_text_from_paddle_ocr_results_internal(ocr_results_list)

def get_text_from_paddle_ocr_results_internal(ocr_results: list) -> str:
    """
    Extracts concatenated text from PaddleOCR results, applying post-processing
    to improve structure and reduce noise.
    """
    full_text_lines = []
    if not ocr_results or not isinstance(ocr_results, list) or not ocr_results[0]:
        logger.debug("PaddleOCR returned empty or invalid results.", extra={"event": "empty_ocr_results"})
        return ""

    # Sort results by Y-coordinate then X-coordinate to improve reading order
    sorted_lines = sorted(ocr_results[0], key=lambda x: (x[0][0][1] + x[0][2][1]) / 2 + (x[0][0][0] + x[0][1][0]) / 10000.0)

    previous_line_bbox = None
    paragraph_buffer = []

    for line_info in sorted_lines:
        if not isinstance(line_info, (list, tuple)) or len(line_info) < 2:
            logger.debug(f"Skipping malformed OCR line_info: {line_info}", extra={"event": "malformed_ocr_line_info", "line_info": line_info})
            continue

        if not isinstance(line_info[1], tuple) or len(line_info[1]) < 1:
            logger.debug(f"Skipping malformed OCR text_info: {line_info[1]}", extra={"event": "malformed_ocr_text_info", "text_info": line_info[1]})
            continue
            
        text = line_info[1][0]
        current_line_bbox = line_info[0]

        # Aggressive Sanitization and Noise Reduction
        cleaned_text = text.strip()
        
        # Remove specific noise patterns like standalone 'l' or 'I' that are likely bullet points
        cleaned_text = re.sub(r'^\s*[lI]\s*$', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^\s*[\-\*\•]\s*', '', cleaned_text)
        
        # Remove non-alphanumeric characters that are not common punctuation or spaces
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-@_+#%&/()\[\]\{\}]+$', '', cleaned_text).strip()
        
        # Remove multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        if not cleaned_text:
            continue

        # IMPROVEMENT 3: Heuristic to handle fragmented lines/paragraphs and logical grouping
        should_join = False
        if previous_line_bbox:
            prev_x1, prev_y1 = previous_line_bbox[0][0], previous_line_bbox[0][1]
            prev_x2, prev_y2 = previous_line_bbox[2][0], previous_line_bbox[2][1]
            
            curr_x1, curr_y1 = current_line_bbox[0][0], current_line_bbox[0][1]
            curr_x2, curr_y2 = current_line_bbox[2][0], current_line_bbox[2][1]

            avg_line_height = (prev_y2 - prev_y1 + curr_y2 - curr_y1) / 2.0
            
            # Condition for joining:
            # 1. Vertical proximity: Lines are on roughly the same horizontal level (for multi-column)
            # 2. Horizontal proximity: The gap between them is small
            # 3. Semantic clues: Previous line doesn't end with strong punctuation, current line starts lowercase

            # Heuristic for multi-column or wrap-around text
            # Check if lines are horizontally close enough and vertically aligned
            if abs(curr_y1 - prev_y1) < avg_line_height * 0.8 and (curr_x1 - prev_x2) < avg_line_height * 3:
                if paragraph_buffer and not re.search(r'[.!?]$', paragraph_buffer[-1]):
                    should_join = True
            
            # Heuristic for simple line continuation (e.g., word broken by newline)
            # Check if the current line starts just below the previous one, and previous line is not a sentence end
            if not should_join and (curr_y1 - prev_y2) < avg_line_height * 0.8:
                if paragraph_buffer and not re.search(r'[.!?]$', paragraph_buffer[-1]):
                    should_join = True

        if should_join and paragraph_buffer:
            paragraph_buffer[-1] += " " + cleaned_text
        else:
            if paragraph_buffer:
                full_text_lines.append(" ".join(paragraph_buffer))
            paragraph_buffer = [cleaned_text]
        
        previous_line_bbox = current_line_bbox

    if paragraph_buffer:
        full_text_lines.append(" ".join(paragraph_buffer))

    final_text = "\n\n".join(full_text_lines)
    
    final_text = re.sub(r'\n\s*\n', '\n\n', final_text) 
    final_text = re.sub(r'(\w)\n(\w)', r'\1 \2', final_text) 
    final_text = "\n".join([line.strip() for line in final_text.split('\n')])
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    # Remove specific repetitive noise patterns that might remain
    # This should be done carefully to avoid removing actual content.
    # Example: if "All-in-One LED Solar Street-light" appears multiple times due to OCR errors,
    # try to normalize it.
    final_text = re.sub(r'(?i)(all-in-one\s+led\s+solar\s+street-light)\s+(ph-11-a-l-wxoa)', r'\1 \2', final_text)
    final_text = re.sub(r'(?i)(technical\s+specifications?)\s*(\n|$)', r'TECHNICAL SPECIFICATIONS\n', final_text)
    final_text = re.sub(r'^\s*[lI]\s*', '', final_text, flags=re.MULTILINE) # Ensure these are gone

    return final_text.strip()


@app.get("/health")
async def health_check():
    """
    Health check endpoint for the OCR microservice.
    Returns 200 OK if the service is running and OCR predictor is loaded.
    """
    if ocr_predictor is None:
        logger.error("OCR predictor is not loaded. Health check failed.", extra={"status": "unhealthy", "reason": "ocr_predictor_not_loaded"})
        raise HTTPException(status_code=503, detail="OCR predictor not loaded. Service is not ready.")
    logger.info("OCR microservice health check successful.", extra={"status": "healthy"})
    return JSONResponse(content={"status": "healthy", "message": "OCR service is running and models are loaded."})

# File-level caching using a simple hash check
# This dictionary will store hashes of processed files and their OCR results
# In a production system, this would be a more persistent cache (e.g., Redis, diskcache)
FILE_OCR_CACHE = {}

def compute_file_hash(file_bytes: bytes) -> str:
    """Computes SHA256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


@app.post("/process_document")
async def process_document_endpoint(
    file: UploadFile = File(...),
    coordinates: Optional[str] = Form(None) 
):
    """
    Processes an uploaded document (PDF, PNG, JPG, JPEG, TXT) using direct PaddleOCR.
    Can optionally crop a specific region of an image/PDF page if coordinates are provided.
    Returns extracted text. (Table extraction is not supported in this mode).
    Includes file-level caching.
    """
=======
# ocr_service.py
import os
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import logging
import traceback
from typing import List, Dict, Optional, Tuple, Any
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import re
import yaml # Already imported, but noting its use for potential config
import json # For parsing layout_data
import time # For performance logging
from functools import lru_cache # IMPROVEMENT 1: For lightweight caching
import hashlib # For file hashing for caching

# Import PaddlePaddle inference library directly
try:
    import paddle.inference as paddle_infer 
except ImportError:
    # Ensure logger is defined before use in this block as well
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)
    logger.critical("Failed to import paddle.inference. Please ensure PaddlePaddle is installed correctly.",
                    extra={"event": "paddle_inference_import_failed"})
    paddle_infer = None

# Set up logging for the OCR service (ensure this is only called once, at the very top)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models") 

POPPLER_PATH = os.getenv("POPPLER_BIN_PATH", "/usr/bin") 
if not os.path.isdir(POPPLER_PATH) or not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm")):
    logger.critical(f"Poppler binary path '{POPPLER_PATH}' not found or does not contain pdftoppm. PDF processing WILL FAIL.",
                   extra={"poppler_path": POPPLER_PATH, "event": "poppler_path_critical"})
else:
    logger.info(f"Poppler binary path found at '{POPPLER_PATH}'.", extra={"poppler_path": POPPLER_PATH})

app = FastAPI(
    title="Pyrotech OCR Microservice",
    description="Dedicated service for high-performance OCR using PaddleOCR, with enhanced image pre-processing and text post-processing.",
    version="1.5.0", # Incrementing version due to OCR quality improvements and caching
)

ocr_predictor = None # Initialize globally

# IMPROVEMENT 1: Add lightweight caching to preprocess_image
# Use a wrapper function to allow caching of PIL Image (which is not hashable)
@lru_cache(maxsize=64) # Cache up to 64 preprocessed images
def _preprocess_image_cached_wrapper(image_bytes: bytes) -> np.ndarray:
    """Wrapper to allow caching of preprocess_image using image bytes hash."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess_image_internal(pil_image)

def preprocess_image_internal(pil_image: Image.Image) -> np.ndarray:
    """
    Applies pre-processing steps to a PIL Image for improved OCR accuracy.
    Steps include: Convert to OpenCV, Deskew, Binarize + Denoise, Optional Upscale.
    Args:
        pil_image: The input image as a PIL Image object.
    Returns:
        A preprocessed OpenCV image (NumPy array).
    """
    logger.debug("Starting image pre-processing...", extra={"event": "image_preprocessing_start"})
    img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if img_np is None or img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
        logger.warning("Input image for preprocessing is empty or invalid.", extra={"event": "empty_image_input"})
        return img_np

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    try:
        if np.max(gray) - np.min(gray) > 10:
            coords = np.column_stack(np.where(gray < 200))
            if coords.size > 0:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle
                
                (h, w) = img_np.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_np = cv2.warpAffine(img_np, M, (w, h),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                logger.debug(f"Image deskewed by {angle:.2f} degrees.", extra={"event": "image_deskewed", "angle": angle})
            else:
                logger.debug("No significant foreground pixels found for deskewing.", extra={"event": "no_foreground_pixels_for_deskew"})
        else:
            logger.debug("Insufficient contrast for deskewing. Skipping.", extra={"event": "insufficient_contrast_for_deskew"})

    except Exception as e:
        logger.warning(f"Deskewing failed: {e}. Skipping deskew for this image.", extra={"event": "deskew_failed", "error": str(e)})

    gray_after_deskew = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    binarized = cv2.adaptiveThreshold(gray_after_deskew, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(binarized, 3)

    h, w = denoised.shape
    if min(h, w) < 1200:
        denoised = cv2.resize(denoised, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Image upscaled from {w}x{h} to {denoised.shape[1]}x{denoised.shape[0]}.",
                     extra={"event": "image_upscaled", "original_dims": f"{w}x{h}", "new_dims": f"{denoised.shape[1]}x{denoised.shape[0]}"})
    
    logger.debug("Image pre-processing complete.", extra={"event": "image_preprocessing_complete"})
    return denoised

def initialize_ocr_models():
    """Initializes only the PaddleOCR predictor."""
    global ocr_predictor

    if ocr_predictor is None:
        try:
            logger.info("Initializing PaddleOCR predictor (CPU mode)...", extra={"event": "paddleocr_init_start"})
            ocr_predictor = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, enable_mkldnn=True) 
            logger.info("PaddleOCR predictor initialized.", extra={"event": "paddleocr_init_success"})
        except Exception as e:
            logger.critical(f"Failed to initialize PaddleOCR predictor: {e}", extra={"event": "paddleocr_init_failed", "error": str(e)})
            logger.critical(traceback.format_exc())
            ocr_predictor = None
            raise

@app.on_event("startup")
async def startup_event():
    logger.info("OCR Microservice starting up...", extra={"event": "service_startup"})
    try:
        initialize_ocr_models()
        logger.info("OCR Microservice models initialized successfully.", extra={"event": "models_init_success"})
    except Exception as e:
        logger.critical(f"FATAL: OCR Microservice startup failed: {e}", extra={"event": "service_startup_failed", "error": str(e)})
        pass

@app.on_event("shutdown")
def shutdown_event():
    logger.info("OCR Microservice shutting down.", extra={"event": "service_shutdown"})

# IMPROVEMENT 1: Add lightweight caching to get_text_from_paddle_ocr_results
@lru_cache(maxsize=128) # Cache up to 128 OCR results
def _get_text_from_paddle_ocr_results_cached_wrapper(ocr_results_tuple: Tuple[Tuple]) -> str:
    """Wrapper to allow caching of get_text_from_paddle_ocr_results."""
    # Convert tuple back to list for internal processing
    ocr_results_list = [list(item) for item in ocr_results_tuple]
    return get_text_from_paddle_ocr_results_internal(ocr_results_list)

def get_text_from_paddle_ocr_results_internal(ocr_results: list) -> str:
    """
    Extracts concatenated text from PaddleOCR results, applying post-processing
    to improve structure and reduce noise.
    """
    full_text_lines = []
    if not ocr_results or not isinstance(ocr_results, list) or not ocr_results[0]:
        logger.debug("PaddleOCR returned empty or invalid results.", extra={"event": "empty_ocr_results"})
        return ""

    # Sort results by Y-coordinate then X-coordinate to improve reading order
    sorted_lines = sorted(ocr_results[0], key=lambda x: (x[0][0][1] + x[0][2][1]) / 2 + (x[0][0][0] + x[0][1][0]) / 10000.0)

    previous_line_bbox = None
    paragraph_buffer = []

    for line_info in sorted_lines:
        if not isinstance(line_info, (list, tuple)) or len(line_info) < 2:
            logger.debug(f"Skipping malformed OCR line_info: {line_info}", extra={"event": "malformed_ocr_line_info", "line_info": line_info})
            continue

        if not isinstance(line_info[1], tuple) or len(line_info[1]) < 1:
            logger.debug(f"Skipping malformed OCR text_info: {line_info[1]}", extra={"event": "malformed_ocr_text_info", "text_info": line_info[1]})
            continue
            
        text = line_info[1][0]
        current_line_bbox = line_info[0]

        # Aggressive Sanitization and Noise Reduction
        cleaned_text = text.strip()
        
        # Remove specific noise patterns like standalone 'l' or 'I' that are likely bullet points
        cleaned_text = re.sub(r'^\s*[lI]\s*$', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^\s*[\-\*\•]\s*', '', cleaned_text)
        
        # Remove non-alphanumeric characters that are not common punctuation or spaces
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-@_+#%&/()\[\]\{\}]+$', '', cleaned_text).strip()
        
        # Remove multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        if not cleaned_text:
            continue

        # IMPROVEMENT 3: Heuristic to handle fragmented lines/paragraphs and logical grouping
        should_join = False
        if previous_line_bbox:
            prev_x1, prev_y1 = previous_line_bbox[0][0], previous_line_bbox[0][1]
            prev_x2, prev_y2 = previous_line_bbox[2][0], previous_line_bbox[2][1]
            
            curr_x1, curr_y1 = current_line_bbox[0][0], current_line_bbox[0][1]
            curr_x2, curr_y2 = current_line_bbox[2][0], current_line_bbox[2][1]

            avg_line_height = (prev_y2 - prev_y1 + curr_y2 - curr_y1) / 2.0
            
            # Condition for joining:
            # 1. Vertical proximity: Lines are on roughly the same horizontal level (for multi-column)
            # 2. Horizontal proximity: The gap between them is small
            # 3. Semantic clues: Previous line doesn't end with strong punctuation, current line starts lowercase

            # Heuristic for multi-column or wrap-around text
            # Check if lines are horizontally close enough and vertically aligned
            if abs(curr_y1 - prev_y1) < avg_line_height * 0.8 and (curr_x1 - prev_x2) < avg_line_height * 3:
                if paragraph_buffer and not re.search(r'[.!?]$', paragraph_buffer[-1]):
                    should_join = True
            
            # Heuristic for simple line continuation (e.g., word broken by newline)
            # Check if the current line starts just below the previous one, and previous line is not a sentence end
            if not should_join and (curr_y1 - prev_y2) < avg_line_height * 0.8:
                if paragraph_buffer and not re.search(r'[.!?]$', paragraph_buffer[-1]):
                    should_join = True

        if should_join and paragraph_buffer:
            paragraph_buffer[-1] += " " + cleaned_text
        else:
            if paragraph_buffer:
                full_text_lines.append(" ".join(paragraph_buffer))
            paragraph_buffer = [cleaned_text]
        
        previous_line_bbox = current_line_bbox

    if paragraph_buffer:
        full_text_lines.append(" ".join(paragraph_buffer))

    final_text = "\n\n".join(full_text_lines)
    
    final_text = re.sub(r'\n\s*\n', '\n\n', final_text) 
    final_text = re.sub(r'(\w)\n(\w)', r'\1 \2', final_text) 
    final_text = "\n".join([line.strip() for line in final_text.split('\n')])
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    # Remove specific repetitive noise patterns that might remain
    # This should be done carefully to avoid removing actual content.
    # Example: if "All-in-One LED Solar Street-light" appears multiple times due to OCR errors,
    # try to normalize it.
    final_text = re.sub(r'(?i)(all-in-one\s+led\s+solar\s+street-light)\s+(ph-11-a-l-wxoa)', r'\1 \2', final_text)
    final_text = re.sub(r'(?i)(technical\s+specifications?)\s*(\n|$)', r'TECHNICAL SPECIFICATIONS\n', final_text)
    final_text = re.sub(r'^\s*[lI]\s*', '', final_text, flags=re.MULTILINE) # Ensure these are gone

    return final_text.strip()


@app.get("/health")
async def health_check():
    """
    Health check endpoint for the OCR microservice.
    Returns 200 OK if the service is running and OCR predictor is loaded.
    """
    if ocr_predictor is None:
        logger.error("OCR predictor is not loaded. Health check failed.", extra={"status": "unhealthy", "reason": "ocr_predictor_not_loaded"})
        raise HTTPException(status_code=503, detail="OCR predictor not loaded. Service is not ready.")
    logger.info("OCR microservice health check successful.", extra={"status": "healthy"})
    return JSONResponse(content={"status": "healthy", "message": "OCR service is running and models are loaded."})

# File-level caching using a simple hash check
# This dictionary will store hashes of processed files and their OCR results
# In a production system, this would be a more persistent cache (e.g., Redis, diskcache)
FILE_OCR_CACHE = {}

def compute_file_hash(file_bytes: bytes) -> str:
    """Computes SHA256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


@app.post("/process_document")
async def process_document_endpoint(
    file: UploadFile = File(...),
    coordinates: Optional[str] = Form(None) 
):
    """
    Processes an uploaded document (PDF, PNG, JPG, JPEG, TXT) using direct PaddleOCR.
    Can optionally crop a specific region of an image/PDF page if coordinates are provided.
    Returns extracted text. (Table extraction is not supported in this mode).
    Includes file-level caching.
    """
>>>>>>> dd6123601b3b1df1e426d6eac04f58ebd5c6ba7f
    start_time = time.perf_counter()
    if ocr_predictor is None:
        logger.error("OCR predictor is not loaded. Cannot process document.", extra={"status": "unhealthy", "reason": "ocr_predictor_not_loaded"})
        raise HTTPException(status_code=503, detail="OCR predictor not loaded. Service is not ready.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    extracted_text = ""
    extracted_tables = [] # Will remain empty in this mode

    logger.info(f"Received file for processing: {file.filename} ({file.content_type})",
                extra={"original_file_name_log": file.filename, "content_type": file.content_type, "event": "file_received"})

    file_bytes = await file.read()
    if not file_bytes:
        logger.warning(f"Received empty file: '{file.filename}'.", extra={"doc_filename": file.filename, "event": "empty_file_received"})
        raise HTTPException(status_code=400, detail="Empty file received.")

    file_hash = compute_file_hash(file_bytes)
    cache_key = f"{file_hash}_{coordinates or 'no_coords'}"

    if cache_key in FILE_OCR_CACHE:
        extracted_text = FILE_OCR_CACHE[cache_key]
        logger.info(f"Cache hit for file '{file.filename}'. Returning cached OCR result.",
                    extra={"doc_filename": file.filename, "cache_status": "hit", "duration_seconds": 0})
        # MODIFIED: Include ocr_kv_fallback even from cache if needed
        ocr_kv_pairs = extract_key_value_pairs_from_text(extracted_text)
        ocr_kv_markdown = "| Parameter | Value |\n|---|---|\n" + "\n".join([f"| {k} | {v} |" for k, v in ocr_kv_pairs]) if ocr_kv_pairs else None
        return JSONResponse({
            "status": "success",
            "message": "Document processed successfully (from cache).",
            "extracted_text": extracted_text,
            "extracted_tables": extracted_tables,
            "ocr_kv_fallback": ocr_kv_markdown, # ADDED: Fallback KV pairs
            "original_filename": file.filename
        })

    try:
        pil_img = None

        if file_extension == ".pdf":
            logger.info("Processing PDF file...", extra={"original_file_name_log": file.filename, "file_type": "pdf"})
            try:
                if coordinates:
                    images = convert_from_bytes(file_bytes, dpi=300, poppler_path=POPPLER_PATH, first_page=1, last_page=1)
                else:
                    images = convert_from_bytes(file_bytes, dpi=300, poppler_path=POPPLER_PATH) 
            except Exception as e:
                logger.error(f"Error converting PDF '{file.filename}' to images: {e}",
                             extra={"original_file_name_log": file.filename, "error": str(e), "event": "pdf_conversion_failed"})
                raise HTTPException(status_code=400, detail=f"Could not convert PDF to images. Ensure Poppler is correctly installed and accessible. Error: {e}")
            
            if not images:
                logger.warning("PDF conversion resulted in no images.", extra={"original_file_name_log": file.filename, "warning": "no_images_from_pdf"})
                raise HTTPException(status_code=400, detail="PDF conversion resulted in no images.")

            page_texts = []
            for i, pil_page_img in enumerate(images):
                current_pil_img = pil_page_img
                if coordinates:
                    try:
                        coords_list = list(map(int, coordinates.split(',')))
                        if len(coords_list) == 4:
                            current_pil_img = current_pil_img.crop((coords_list[0], coords_list[1], coords_list[2], coords_list[3]))
                            logger.info(f"Cropped image for OCR based on coordinates: {coordinates}",
                                        extra={"original_file_name_log": file.filename, "page": i+1, "coordinates": coordinates, "action": "image_cropped"})
                        else:
                            logger.warning(f"Invalid coordinates format: {coordinates}. Skipping crop.",
                                           extra={"original_file_name_log": file.filename, "page": i+1, "coordinates": coordinates, "warning": "invalid_coords_format"})
                    except ValueError:
                        logger.warning(f"Could not parse coordinates: {coordinates}. Skipping crop.",
                                       extra={"original_file_name_log": file.filename, "page": i+1, "coordinates": coordinates, "warning": "coords_parsing_failed"})
                
                # Convert PIL image to bytes for caching wrapper
                img_byte_arr = io.BytesIO()
                current_pil_img.save(img_byte_arr, format="PNG")
                img_bytes_for_cache = img_byte_arr.getvalue()

                img_np = _preprocess_image_cached_wrapper(img_bytes_for_cache)
                
                if img_np is None or img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
                    logger.warning(f"Skipping empty or invalid image (NumPy array) on page {i+1} of '{file.filename}'.",
                                   extra={"original_file_name_log": file.filename, "page": i+1, "warning": "empty_invalid_image_np"})
                    continue
                
                logger.info(f"Performing OCR on page {i+1} of {file.filename} (preprocessed)...",
                            extra={"original_file_name_log": file.filename, "page": i+1, "event": "ocr_start"})
                
                ocr_result = ocr_predictor.ocr(img_np, cls=False)
                # Convert list of lists/tuples to tuple of tuples for caching
                ocr_result_hashable = tuple(tuple(item) for item in ocr_result[0]) if ocr_result and ocr_result[0] else tuple()
                
                page_block_text = _get_text_from_paddle_ocr_results_cached_wrapper(ocr_result_hashable)
                
                if page_block_text:
                    page_texts.append(page_block_text)
                    logger.info(f"Extracted text from page {i+1} (first 50 chars): {page_block_text[:50]}...",
                                extra={"original_file_name_log": file.filename, "page": i+1, "extracted_text_snippet": page_block_text[:50]})
                else:
                    logger.info(f"No meaningful text extracted from page {i+1}.",
                                extra={"original_file_name_log": file.filename, "page": i+1, "event": "no_meaningful_text"})

            extracted_text = "\n\n".join(page_texts)

        elif file_extension in [".png", ".jpg", ".jpeg"]:
            logger.info(f"Processing image file: {file.filename} directly with OCR...",
                        extra={"original_file_name_log": file.filename, "file_type": "image"})
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

            current_pil_img = pil_img
            if coordinates:
                try:
                    coords_list = list(map(int, coordinates.split(',')))
                    if len(coords_list) == 4:
                        current_pil_img = current_pil_img.crop((coords_list[0], coords_list[1], coords_list[2], coords_list[3]))
                        logger.info(f"Cropped image for OCR based on coordinates: {coordinates}",
                                    extra={"original_file_name_log": file.filename, "coordinates": coordinates, "action": "image_cropped"})
                    else:
                        logger.warning(f"Invalid coordinates format: {coordinates}. Skipping crop.",
                                       extra={"original_file_name_log": file.filename, "coordinates": coordinates, "warning": "invalid_coords_format"})
                except ValueError:
                    logger.warning(f"Could not parse coordinates: {coordinates}. Skipping crop.",
                                   extra={"original_file_name_log": file.filename, "coordinates": coordinates, "warning": "coords_parsing_failed"})
            
            # Convert PIL image to bytes for caching wrapper
            img_byte_arr = io.BytesIO()
            current_pil_img.save(img_byte_arr, format="PNG")
            img_bytes_for_cache = img_byte_arr.getvalue()

            img_np = _preprocess_image_cached_wrapper(img_bytes_for_cache)

            if img_np is None or img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
                logger.error(f"Image file '{file.filename}' is empty or invalid after preprocessing.",
                             extra={"original_file_name_log": file.filename, "error": "empty_invalid_image_after_preprocess"})
                raise HTTPException(status_code=400, detail=f"Image file '{file.filename}' is empty or invalid after preprocessing.")
            
            logger.info(f"Performing OCR on image '{file.filename}' (preprocessed)...",
                        extra={"original_file_name_log": file.filename, "event": "ocr_start"})
            ocr_result = ocr_predictor.ocr(img_np, cls=False)
            # Convert nested lists to tuples for hashing/caching compatibility
            ocr_result_hashable = nested_list_to_tuple(ocr_result)

            image_block_text = _get_text_from_paddle_ocr_results_cached_wrapper(ocr_result_hashable[0] if ocr_result_hashable and ocr_result_hashable[0] else tuple())

            # ADDED: Table-aware post-processing
            ocr_tables = cluster_ocr_boxes_into_table(ocr_result)
            extracted_tables = None
            if ocr_tables and any(len(row) > 1 for row in ocr_tables):
                # Convert to Markdown
                max_cols = max(len(row) for row in ocr_tables)
                header = [f"Col{i+1}" for i in range(max_cols)]
                md = "| " + " | ".join(header) + " |\n|" + "---|" * max_cols + "\n"
                for row in ocr_tables:
                    row_md = [cell if idx < len(row) else "" for idx, cell in enumerate(header)]
                    row_md = row + ["" for _ in range(max_cols - len(row))]
                    md += "| " + " | ".join(row_md) + " |\n"
                extracted_tables = md

            if image_block_text:
                extracted_text = image_block_text
                logger.info(f"Extracted text from image '{file.filename}' (first 50 chars): {extracted_text[:50]}...",
                            extra={"original_file_name_log": file.filename, "extracted_text_snippet": extracted_text[:50]})
            else:
                logger.info(f"No meaningful text extracted from image '{file.filename}'.",
                            extra={"original_file_name_log": file.filename, "event": "no_meaningful_text"})

        elif file_extension == ".txt":
            logger.info("Processing TXT file (no OCR needed).", extra={"original_file_name_log": file.filename, "file_type": "txt"})
            extracted_text = file_bytes.decode('utf-8')

        else:
            logger.error(f"Unsupported file type: {file_extension}. Only PDF, PNG, JPG, JPEG, TXT are supported.",
                         extra={"original_file_name_log": file.filename, "file_extension": file_extension, "error": "unsupported_file_type"})
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}. Only PDF, PNG, JPG, JPEG, TXT are supported.")

        # Store result in cache before returning
        FILE_OCR_CACHE[cache_key] = extracted_text

        # ADDED: Extract key-value pairs from OCR text as a fallback
        ocr_kv_pairs = extract_key_value_pairs_from_text(extracted_text)
        ocr_kv_markdown = "| Parameter | Value |\n|---|---|\n" + "\n".join([f"| {k} | {v} |" for k, v in ocr_kv_pairs]) if ocr_kv_pairs else None

        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"Document processing completed successfully for '{file.filename}' in {duration:.2f} seconds.",
                    extra={"original_file_name_log": file.filename, "status": "success", "duration_seconds": duration})
        return JSONResponse({
            "status": "success",
            "message": "Document processed successfully.",
            "extracted_text": extracted_text,
            "extracted_tables": extracted_tables,
            "ocr_kv_fallback": ocr_kv_markdown, # ADDED: Fallback KV pairs
            "original_filename": file.filename
        })

    except Exception as e:
        logger.error(f"Error processing file '{file.filename}': {e}", extra={"original_file_name_log": file.filename, "error": str(e)})
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

# Utility: Recursively convert nested lists to tuples for hashing/caching
def nested_list_to_tuple(obj):
    """
    Recursively convert nested lists to tuples for hashing/caching.
    Handles lists of lists, lists of dicts, etc.
    """
    if isinstance(obj, list):
        return tuple(nested_list_to_tuple(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((k, nested_list_to_tuple(v)) for k, v in obj.items())
    else:
        return obj

# Utility: Cluster OCR boxes into table rows/columns
def cluster_ocr_boxes_into_table(ocr_results, y_tol=15, x_tol=30):
    """
    Given PaddleOCR results, cluster boxes into a table grid (list of rows of cell texts).
    Args:
        ocr_results: PaddleOCR results (list of [box, (text, score)]).
        y_tol: vertical tolerance for row clustering.
        x_tol: horizontal tolerance for column sorting (not used here, but can be extended).
    Returns:
        List of rows, each row is a list of cell texts.
    """
    if not ocr_results or not ocr_results[0]:
        return []
    boxes = ocr_results[0]
    # Each box: [box_points, (text, score)]
    # Sort by y (top of bounding box)
    def box_center_y(box):
        pts = box[0]
        return int((pts[0][1] + pts[2][1]) / 2)
    boxes_sorted = sorted(boxes, key=lambda x: (box_center_y(x), x[0][0][0]))
    rows = []
    current_row = []
    last_y = None
    for box in boxes_sorted:
        y = box_center_y(box)
        text = box[1][0]
        if last_y is None or abs(y - last_y) < y_tol:
            current_row.append((box[0][0][0], text))  # x, text
        else:
            # Sort current row by x
            current_row = [t for _, t in sorted(current_row, key=lambda x: x[0])]
            rows.append(current_row)
            current_row = [(box[0][0][0], text)]
        last_y = y
    if current_row:
        current_row = [t for _, t in sorted(current_row, key=lambda x: x[0])]
        rows.append(current_row)
    return rows

# ADDED: New function for key-value pair extraction
def extract_key_value_pairs_from_text(raw_text: str) -> list:
    """
    Parses flat OCR text to recover key-value pairs based on label + value proximity.
    Returns a list of tuples (key, value).
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    pairs = []
    i = 0
    while i < len(lines) - 1:
        key = lines[i]
        val = lines[i+1]

        # Key heuristics
        # Expanded regex to be more robust for common key patterns
        if re.match(r'^[A-Z][\w\s]*(\(.*\))?$|.*(Voltage|Power|Enclosure|Battery|Module|Sensor|Efficiency|Dimensions|Weight|Model|Part No|SKU|Type|Category|Color|Material|Frequency|Input|Output|Temperature|Protection|Certification|Standard|Warranty|Mounting|Application|Features|Series|Product|Description)\s*$', key, re.IGNORECASE):
            # Value heuristics: alphanumeric, percentages, units, common symbols
            if re.match(r'^[-+]?[\w\d\s%°/.,\(\)\[\]\{\}]+$', val):
                pairs.append((key, val))
                i += 2
                continue
        i += 1
    return pairs
