# layout_detector_service.py
import os
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback
from typing import List, Dict, Optional, Any
import mimetypes # For better MIME type detection
import uuid # For unique IDs in blocks
import re # For regex to parse key-value pairs
import time # For performance logging
from pdf_utils import is_pdf_encrypted

# For PDF metadata extraction
try:
    import fitz # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF (fitz) not found. PDF metadata extraction will be skipped. Install with 'pip install PyMuPDF'.")
    fitz = None

# Import unstructured libraries
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    CompositeElement, NarrativeText, Title, List as UnstructuredList, Table, Image as UnstructuredImage, ElementMetadata, Text
)
from unstructured.staging.base import elements_to_json # For structured output of elements
from unstructured.partition.html import partition_html # NEW: Import partition_html for HTML files

# REMOVED: Import extract_key_value_pairs_from_text from ocr_service
# from ocr_service import extract_key_value_pairs_from_text

# Set up logging for the layout detector service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
POPPLER_PATH = os.getenv("POPPLER_BIN_PATH", "/usr/bin") 

app = FastAPI(
    title="Pyrotech Unstructured Layout/Table Detector Microservice",
    description="Dedicated service for document layout and table region detection using Unstructured.io, with enhanced metadata extraction and post-processing for improved structure.",
    version="1.5.1", # Incrementing version due to strategy change
)

@app.on_event("startup")
async def startup_event():
    logger.info("Unstructured Layout Detector Microservice starting up.", extra={"event": "service_startup"})
    logger.info("Unstructured Layout Detector Microservice ready to process documents.", extra={"event": "service_ready"})

@app.get("/health")
async def health_check():
    """
    Health check endpoint for the Layout Detector microservice.
    Returns 200 OK if the service is running.
    """
    logger.info("Layout Detector microservice health check successful.", extra={"status": "healthy"})
    return JSONResponse(content={"status": "healthy", "message": "Layout Detector service is running."})

# Add Confidence Score Heuristic to Extracted Tables
def estimate_table_confidence(table_md: str) -> float:
    """Estimates a confidence score for an extracted markdown table."""
    num_rows = table_md.count("\n") - 2 # Subtract header and separator
    num_cols = table_md.count("|") // num_rows if num_rows > 0 else 0
    
    # Basic heuristics:
    # - More rows and columns generally means higher confidence.
    # - Presence of a header separator (---|---) is good.
    # - If it's a very short table, confidence might be lower.
    
    confidence = 0.0
    if "|---|---|" in table_md: # Check for markdown table separator
        confidence += 0.5
    
    if num_rows > 0 and num_cols > 0:
        confidence += min(0.4, (num_rows * num_cols) / 20.0) # Scale based on size, max 0.4
    
    # If it's a single row, it might be a false positive, unless it's a clear key-value
    if num_rows == 1 and num_cols == 2:
        confidence = max(confidence, 0.6) # Boost for clear key-value like single rows

    return min(1.0, confidence) # Cap at 1.0

# Add PDF Table Extraction (pdfplumber)

def extract_tables_from_pdf(file_bytes: bytes):
    """
    Extract tables from a PDF file using pdfplumber.
    Returns a list of tables, each as a list of rows (each row is a list of cell strings).
    """
    import io
    tables = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and any(any(cell for cell in row) for row in table):
                        tables.append(table)
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed: {e}")
    return tables

# Add PDF Metadata Extraction
def extract_pdf_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """Extracts metadata from a PDF file using PyMuPDF."""
    if fitz is None:
        logger.warning("PyMuPDF not installed, skipping PDF metadata extraction.")
        return {}
    
    metadata = {}
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            meta = doc.metadata
            for key, value in meta.items():
                if value is not None:
                    metadata[key] = value
            logger.info(f"Extracted PDF metadata: {metadata}", extra={"pdf_metadata": metadata})
    except Exception as e:
        logger.error(f"Error extracting PDF metadata: {e}", extra={"error_detail": str(e)})
    return metadata


@app.post("/detect_layout")
async def detect_layout(file: UploadFile = File(...)):
    """
    Detects layout elements (text, tables, images) in a document using Unstructured.io.
    Extracts full text, structured tables (Markdown), and detailed block metadata.
    Includes general logic for identifying and formatting key-value pair tables.
    Now supports HTML files directly via partition_html.
    """
    start_time = time.perf_counter() # Start performance tracking
    logger.info(f"Received file for layout detection: '{file.filename}' (Type: {file.content_type})",
                extra={"doc_filename": file.filename, "content_type": file.content_type, "event": "file_received"})

    if not file.filename:
        logger.warning("No file name provided.", extra={"event": "no_filename_provided"})
        raise HTTPException(status_code=400, detail="No file name provided.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    detected_mime_type = mimetypes.guess_type(file.filename)[0] or file.content_type

    file_bytes = await file.read()
    
    if not file_bytes:
        logger.warning(f"Received empty file: '{file.filename}'.", extra={"doc_filename": file.filename, "event": "empty_file_received"})
        raise HTTPException(status_code=400, detail="Empty file received.")

    pdf_metadata = {}
    pdfplumber_tables_md = []
    if file_extension == ".pdf":
        pdf_metadata = extract_pdf_metadata(file_bytes)
        # Try pdfplumber table extraction first
        pdfplumber_tables = extract_tables_from_pdf(file_bytes)
        for table in pdfplumber_tables:
            if not table or not any(table):
                continue
            max_cols = max(len(row) for row in table)
            header = [f"Col{i+1}" for i in range(max_cols)]
            md = "| " + " | ".join(header) + " |\n|" + "---|" * max_cols + "\n"
            for row in table:
                row_md = row + ["" for _ in range(max_cols - len(row))]
                md += "| " + " | ".join([str(cell) if cell is not None else "" for cell in row_md]) + " |\n"
            pdfplumber_tables_md.append(md)
        if is_pdf_encrypted(file_bytes):
            logger.error(f"PDF is password protected: {file.filename}", extra={"doc_filename": file.filename})
            raise HTTPException(status_code=400, detail="PDF is password protected and cannot be processed.")
        pdf_metadata = extract_pdf_metadata(file_bytes)
        # Check if PDF is encrypted (password-protected)
        if fitz is not None:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if doc.is_encrypted:
                        logger.error(f"PDF '{file.filename}' is password-protected. Cannot process encrypted PDFs.", extra={"doc_filename": file.filename, "event": "pdf_encrypted"})
                        raise HTTPException(status_code=400, detail="PDF is password-protected. Please provide an unencrypted PDF.")
            except Exception as e:
                logger.error(f"Error checking PDF encryption: {e}", extra={"doc_filename": file.filename, "event": "pdf_encryption_check_error"})
                raise HTTPException(status_code=400, detail="Error processing PDF file. Possibly password-protected or corrupted.")

    # Initialize final_blocks to an empty list as a fallback
    final_blocks: List[Dict[str, Any]] = []

    try:
        partition_start_time = time.perf_counter()
        
        # NEW: Conditional partitioning based on file type
        is_html = (file.content_type == "text/html" or file.filename.lower().endswith(".html"))

        if is_html:
            logger.info(f"Processing HTML file '{file.filename}' using partition_html.",
                        extra={"doc_filename": file.filename, "processing_method": "partition_html"})
            # partition_html expects a file-like object or a path
            elements = partition_html(file=io.BytesIO(file_bytes))
        else:
            logger.info(f"Processing non-HTML file '{file.filename}' using partition (auto strategy).",
                        extra={"doc_filename": file.filename, "processing_method": "partition_auto"})
            elements = partition(
                file=io.BytesIO(file_bytes),
                file_filename=file.filename,
                content_type=detected_mime_type,
                strategy="auto" # Reverted to "auto" as user does not use detectron2
            )
        partition_end_time = time.perf_counter()
        logger.info(f"Unstructured partition took {partition_end_time - partition_start_time:.2f} seconds.",
                    extra={"partition_duration_seconds": partition_end_time - partition_start_time})


        detected_blocks_output: List[Dict[str, Any]] = []
        
        # Regex for general key-value pair patterns (more flexible)
        key_value_pattern = re.compile(
            r'^\s*(?:\"(.*?)\"|\b([\w\s\-\(\)\/\.]+?)\b)\s*[,:]?\s*(?:\"(.*?)\"|\b(.*?)\b)\s*(?:,\s*\"(.*?)\")?\s*$', 
            re.DOTALL | re.IGNORECASE
        )

        current_key_value_group = [] # To accumulate consecutive key-value like lines
        
        for i, element in enumerate(elements):
            block_info: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "type": "unknown",
                "content": element.text if element.text else "",
                "coordinates": None,
                "page": None,
                "metadata": {}
            }

            # Extract coordinates and page number if available
            if hasattr(element.metadata, "coordinates") and element.metadata.coordinates:
                points = element.metadata.coordinates.points
                if points and len(points) > 0:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    min_x, max_x = int(min(x_coords)), int(max(x_coords))
                    min_y, max_y = int(min(y_coords)), int(max(y_coords))
                    block_info["coordinates"] = [min_x, min_y, max_x, max_y]
                else:
                    logger.warning(f"Coordinates metadata present but points are empty for element: {block_info['id']}",
                                   extra={"block_id": block_info['id'], "metadata_coordinates": element.metadata.coordinates, "doc_filename": file.filename})
                block_info["metadata"]["coordinate_system"] = element.metadata.coordinate_system if hasattr(element.metadata, 'coordinate_system') else "pixels"
            
            if hasattr(element.metadata, "page_number") and element.metadata.page_number is not None:
                block_info["page"] = element.metadata.page_number
            
            # Add other useful metadata from Unstructured.io
            for attr in ['filetype', 'filename', 'detection_class', 'category_depth', 'header_footer_type', 'emphasized_text_contents', 'emphasized_text_tags']:
                if hasattr(element.metadata, attr) and getattr(element.metadata, attr) is not None:
                    block_info["metadata"][attr] = getattr(element.metadata, attr)

            # --- Primary Unstructured.io Table Handling ---
            # This is the most reliable way to get structured tables if Unstructured detects them.
            if isinstance(element, Table):
                block_info["type"] = "table"
                if hasattr(element, "text_as_md") and element.text_as_md:
                    block_info["content"] = element.text_as_md
                elif hasattr(element, "text_as_html") and element.text_as_html:
                    block_info["content"] = element.text_as_html
                elif hasattr(element, "text_as_csv") and element.text_as_csv:
                    block_info["content"] = element.text_as_csv
                else:
                    block_info["content"] = element.text
                
                # Add confidence score to table blocks
                block_info["metadata"]["confidence"] = estimate_table_confidence(block_info["content"])

                logger.info(f"Detected Unstructured Table (ID: {block_info['id']}) on page {block_info['page']} with confidence {block_info['metadata']['confidence']:.2f}",
                            extra={"block_id": block_info['id'], "page": block_info['page'], "block_type": "table", "doc_filename": file.filename, "confidence": block_info['metadata']['confidence']})
                
                # Reset key-value group as a new table element breaks the sequence
                if current_key_value_group:
                    custom_table_md = format_key_value_group_as_markdown(current_key_value_group)
                    custom_table_block_info = {
                        "id": str(uuid.uuid4()),
                        "type": "custom_key_value_table",
                        "content": custom_table_md,
                        "coordinates": current_key_value_group[0].get("coordinates"), # Use coords of first item
                        "page": current_key_value_group[0].get("page"), # Use page of first item
                        "metadata": {"source_filename": file.filename, "extracted_count": len(current_key_value_group), "confidence": estimate_table_confidence(custom_table_md)}
                    }
                    detected_blocks_output.append(custom_table_block_info)
                    logger.info(f"Formatted {len(current_key_value_group)} key-value pairs into a custom table due to new Unstructured table.",
                                extra={"doc_filename": file.filename, "num_key_value_pairs": len(current_key_value_group)})
                    current_key_value_group = []
                
                detected_blocks_output.append(block_info)
                continue # Move to next element as this table is handled

            # --- General Key-Value Pair / Semi-Structured Data Handling ---
            # Apply the flexible regex to NarrativeText or Text elements
            if isinstance(element, (NarrativeText, Text, UnstructuredList)): # Also check List elements
                text_to_check = element.text.strip()
                match = key_value_pattern.match(text_to_check)
                
                if match:
                    # Extract key and value from the appropriate group (quoted or unquoted)
                    key = match.group(1) or match.group(2)
                    value = match.group(3) or match.group(4)

                    if key and value:
                        # Clean up newlines within captured key/value
                        key = re.sub(r'\s*\n\s*', ' ', key).strip()
                        value = re.sub(r'\s*\n\s*', ' ', value).strip()
                        
                        # Store coordinates and page for key-value items
                        kv_item = {"parameter": key, "value": value, "coordinates": block_info["coordinates"], "page": block_info["page"]}
                        current_key_value_group.append(kv_item)
                        logger.debug(f"  --> Identified potential key-value pair: {key}: {value} (Page: {block_info['page']})")
                        
                        continue # Continue to next element to see if it's part of the same group
                
                # If the current element does NOT match the key-value pattern,
                # and we have a pending group, format and add the previous group.
                if current_key_value_group:
                    custom_table_md = format_key_value_group_as_markdown(current_key_value_group)
                    custom_table_block_info = {
                        "id": str(uuid.uuid4()),
                        "type": "custom_key_value_table",
                        "content": custom_table_md,
                        "coordinates": current_key_value_group[0].get("coordinates"), # Use coords of first item
                        "page": current_key_value_group[0].get("page"), # Use page of first item
                        "metadata": {"source_filename": file.filename, "extracted_count": len(current_key_value_group), "confidence": estimate_table_confidence(custom_table_md)}
                    }
                    detected_blocks_output.append(custom_table_block_info)
                    logger.info(f"Formatted {len(current_key_value_group)} key-value pairs into a custom table.",
                                extra={"doc_filename": file.filename, "num_key_value_pairs": len(current_key_value_group)})
                    current_key_value_group = [] # Reset for next group

            # --- Standard Unstructured Element Handling (for elements not handled above) ---
            # This block will be executed for elements that are not Tables and not part of a key-value group.
            if isinstance(element, NarrativeText):
                block_info["type"] = "narrative_text"
            elif isinstance(element, Title):
                block_info["type"] = "title"
            elif isinstance(element, UnstructuredList):
                # Normalize bullet points to a consistent format
                block_info["type"] = "list"
                # Ensure list items are prefixed with a consistent bullet
                block_info["content"] = "\n".join([f"- {line.strip()}" for line in element.text.split('\n') if line.strip()])
            elif isinstance(element, UnstructuredImage):
                block_info["type"] = "image"
                block_info["content"] = f"[Image detected: {element.metadata.image_path if hasattr(element.metadata, 'image_path') else 'No path'}]"
                block_info["metadata"]["image_caption"] = "Awaiting image captioning service."
                logger.info(f"Detected image (ID: {block_info['id']}) on page {block_info['page']}",
                            extra={"block_id": block_info['id'], "page": block_info['page'], "block_type": "image", "doc_filename": file.filename})
            elif isinstance(element, CompositeElement):
                block_info["type"] = "composite"
            else:
                block_info["type"] = "other_element"

            # Append this block_info only if it wasn't already handled by a 'continue' statement above
            detected_blocks_output.append(block_info)

        # After the loop, check if there's any pending key-value group
        if current_key_value_group:
            custom_table_md = format_key_value_group_as_markdown(current_key_value_group)
            custom_table_block_info = {
                "id": str(uuid.uuid4()),
                "type": "custom_key_value_table",
                "content": custom_table_md,
                "coordinates": current_key_value_group[0].get("coordinates"), # Use coords of first item
                "page": current_key_value_group[0].get("page"), # Use page of first item
                "metadata": {"source_filename": file.filename, "extracted_count": len(current_key_value_group), "confidence": estimate_table_confidence(custom_table_md)}
            }
            detected_blocks_output.append(custom_table_block_info)
<<<<<<< HEAD
            logger.info(f"Formatted remaining {len(current_key_value_group)} key-value pairs into a custom table at end of document.",
                        extra={"doc_filename": file.filename, "num_key_value_pairs": len(current_key_value_group)})

        try:
            # Post-process detected blocks for merging and cleanup, and ensure clean JSON blocks
            final_blocks = structure_normalizer(detected_blocks_output)
        except Exception as e:
            logger.error(f"Normalization failed: {e}", extra={"error_detail": str(e)})
            final_blocks = []  # fallback to empty list if structure_normalizer fails
        
        # Re-generate full_document_text_parts and extracted_structured_tables from normalized blocks
        full_document_text_parts = [block["content"] for block in final_blocks if block["content"] and block["type"] != "image"]
        extracted_structured_tables = [block["content"] for block in final_blocks if block["type"] in ["table", "custom_key_value_table"] and block["content"]]

        # Combine all non-image text for a full document text output
        extracted_full_text = "\n\n".join(full_document_text_parts).strip()
        
        # Optional Safety Check (Defensive Code) - ensure final_blocks is defined
        if "final_blocks" not in locals():
            final_blocks = []

        end_time = time.perf_counter() # End performance tracking
        duration = end_time - start_time

        logger.info(f"Layout detection completed for '{file.filename}'. Detected {len(final_blocks)} blocks after normalization in {duration:.2f} seconds.",
                    extra={"doc_filename": file.filename, "num_blocks_after_normalization": len(final_blocks), "duration_seconds": duration})
        logger.info(f"Extracted {len(extracted_structured_tables)} markdown tables (including custom specs) and 0 structured tables.",
                    extra={"doc_filename": file.filename, "num_markdown_tables": len(extracted_structured_tables), "num_structured_tables": 0})

=======
            logger.info(f"Formatted remaining {len(current_key_value_group)} key-value pairs into a custom table at end of document.",
                        extra={"doc_filename": file.filename, "num_key_value_pairs": len(current_key_value_group)})

        try:
            # Post-process detected blocks for merging and cleanup, and ensure clean JSON blocks
            final_blocks = structure_normalizer(detected_blocks_output)
        except Exception as e:
            logger.error(f"Normalization failed: {e}", extra={"error_detail": str(e)})
            final_blocks = []  # fallback to empty list if structure_normalizer fails
        
        # Re-generate full_document_text_parts and extracted_structured_tables from normalized blocks
        full_document_text_parts = [block["content"] for block in final_blocks if block["content"] and block["type"] != "image"]
        extracted_structured_tables = [block["content"] for block in final_blocks if block["type"] in ["table", "custom_key_value_table"] and block["content"]]

        # Combine all non-image text for a full document text output
        extracted_full_text = "\n\n".join(full_document_text_parts).strip()
        
        # Optional Safety Check (Defensive Code) - ensure final_blocks is defined
        if "final_blocks" not in locals():
            final_blocks = []

        end_time = time.perf_counter() # End performance tracking
        duration = end_time - start_time

        logger.info(f"Layout detection completed for '{file.filename}'. Detected {len(final_blocks)} blocks after normalization in {duration:.2f} seconds.",
                    extra={"doc_filename": file.filename, "num_blocks_after_normalization": len(final_blocks), "duration_seconds": duration})
        logger.info(f"Extracted {len(extracted_structured_tables)} markdown tables (including custom specs) and 0 structured tables.",
                    extra={"doc_filename": file.filename, "num_markdown_tables": len(extracted_structured_tables), "num_structured_tables": 0})

>>>>>>> dd6123601b3b1df1e426d6eac04f58ebd5c6ba7f
        return JSONResponse({
            "status": "success",
            "message": "Layout detection completed and elements extracted successfully.",
            "extracted_text": extracted_full_text,
            "extracted_markdown_tables": extracted_structured_tables, # Returns markdown tables
            "extracted_pdfplumber_tables": pdfplumber_tables_md,      # <--- ADDED: pdfplumber tables
            "extracted_camelot_tables": [], # Will be empty from this service, but kept for consistency
            "detected_blocks_raw": final_blocks, # All detected blocks with coordinates, type etc.
            "original_filename": file.filename,
            "pdf_metadata": pdf_metadata # Add PDF metadata
<<<<<<< HEAD
        })

    except Exception as e:
        logger.error(f"Error processing file for layout detection '{file.filename}': {e}",
                     extra={"doc_filename": file.filename, "error_detail": str(e), "event": "layout_detection_failed"})
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document for layout detection: {e}")


def format_key_value_group_as_markdown(key_value_pairs: List[Dict[str, Any]]) -> str:
    """
    Formats a list of key-value dictionaries into a Markdown table string.
    Ensures keys/values are escaped for Markdown table compatibility.
    """
    if not key_value_pairs:
        return ""

    header = "| Parameter | Value |\n|---|---|\n"
    rows = []
    for item in key_value_pairs:
        # Escape pipe characters within content to prevent breaking Markdown table format
        param = item.get("parameter", "").replace('|', '\\|').strip()
        value = item.get("value", "").replace('|', '\\|').strip()
        rows.append(f"| {param} | {value} |")
    
    return header + "\n".join(rows)

def structure_normalizer(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies post-processing rules to a list of detected blocks to improve
    logical grouping, merge fragmented content, and reduce noise.
    """
    logger.info(f"Starting structure normalization for {len(blocks)} blocks.", extra={"initial_block_count": len(blocks)})
    normalized_blocks = []
    i = 0
    while i < len(blocks):
        current_block = blocks[i]
        
        # Rule 1: Merge fragmented text blocks (e.g., lines that should be one paragraph/item)
        # This is a heuristic. Look for short, consecutive narrative/text blocks
        # that appear to be fragments of a single logical item.
        if current_block["type"] in ["narrative_text", "text"] and current_block["content"].strip():
            merged_content = current_block["content"].strip()
            merged_coords = list(current_block["coordinates"]) if current_block["coordinates"] else None
            merged_page = current_block["page"]
            
            j = i + 1
            while j < len(blocks):
                next_block = blocks[j]
                # Check for proximity (same page, close vertically) and similar type
                if next_block["page"] == merged_page and \
                   next_block["type"] in ["narrative_text", "text"] and \
                   next_block["content"].strip():
                    
                    # Calculate vertical distance between current block's bottom and next block's top
                    vertical_distance = 0
                    if merged_coords and next_block["coordinates"]:
                        vertical_distance = next_block["coordinates"][1] - merged_coords[3]
                    
                    # Heuristic for continuation: if current merged content doesn't end with sentence punctuation
                    # AND the vertical distance is small (implies it's part of the same paragraph/item)
                    # OR if the next line starts with a lowercase letter (strong indicator of continuation)
                    if (not re.search(r'[.!?]$', merged_content) and vertical_distance < 20) or \
                       (next_block["content"].strip() and next_block["content"].strip()[0].islower() and vertical_distance < 50): # More lenient for lowercase continuation
                        
                        merged_content += " " + next_block["content"].strip()
                        # Update merged coordinates to encompass both blocks
                        if next_block["coordinates"] and merged_coords:
                            merged_coords[2] = max(merged_coords[2], next_block["coordinates"][2]) # Max X2
                            merged_coords[3] = max(merged_coords[3], next_block["coordinates"][3]) # Max Y2
                        j += 1
                        logger.debug(f"Merged fragmented text block: '{current_block['content'][:30]}...' with '{next_block['content'][:30]}...'")
                    else:
                        break # Not a continuation
                else:
                    break # Not a merge candidate
            
            # Create a new block for the merged content
            new_block = {
                "id": str(uuid.uuid4()),
                "type": "narrative_text", # Consolidated type for merged text
                "content": merged_content,
                "coordinates": merged_coords,
                "page": merged_page,
                "metadata": current_block["metadata"]
            }
            normalized_blocks.append(new_block)
            i = j # Move index past merged blocks
            continue # Continue to next iteration of outer loop

        # Rule 2: Handle repetitive headers/noise (e.g., multiple "TECHNICAL SPECIFICATIONS")
        # If a block is a title and its content is very similar to the previous title, skip it
        if current_block["type"] == "title" and normalized_blocks and normalized_blocks[-1]["type"] == "title":
            # Simple string comparison, could use fuzzy matching for robustness
            if current_block["content"].strip().upper() == normalized_blocks[-1]["content"].strip().upper():
                logger.debug(f"Skipping repetitive title: '{current_block['content'][:50]}...'")
                i += 1
                continue
        
        normalized_blocks.append(current_block)
        i += 1
    
    # Rule 3: Bullet Point Normalization (post-processing on normalized_blocks)
    final_blocks = []
    for block in normalized_blocks:
        if block["type"] == "list":
            # Ensure consistent bullet format, remove leading numbers/noise
            cleaned_lines = []
            for line in block["content"].split('\n'):
                line_strip = line.strip()
                if not line_strip: continue
                # Remove common list prefixes (1), a), -, *, l, I, •)
                # FIX: Replaced '•' with its Unicode escape sequence '\u2022'
                line_strip = re.sub(r'^\s*(\d+\)|\w+\)|\-|\*|\u2022|[lI])\s*', '', line_strip, flags=re.IGNORECASE)
                # Add consistent markdown bullet
                cleaned_lines.append(f"- {line_strip.strip()}")
            if cleaned_lines:
                block["content"] = "\n".join(cleaned_lines)
                final_blocks.append(block)
            else:
                logger.debug(f"Skipping empty list block after normalization: {block['id']}")
        elif block["type"] == "narrative_text" and re.match(r'^\s*(\d+\)|\w+\)|\-|\*|\u2022|[lI])\s*', block["content"].strip()):
            # Catch narrative text that looks like a list item and convert it
            cleaned_lines = []
            for line in block["content"].split('\n'):
                line_strip = line.strip()
                if not line_strip: continue
                # FIX: Replaced '•' with its Unicode escape sequence '\u2022'
                line_strip = re.sub(r'^\s*(\d+\)|\w+\)|\-|\*|\u2022|[lI])\s*', '', line_strip, flags=re.IGNORECASE)
                cleaned_lines.append(f"- {line_strip.strip()}")
            if cleaned_lines:
                block["type"] = "list" # Change type to list
                block["content"] = "\n".join(cleaned_lines)
                final_blocks.append(block)
            else:
                logger.debug(f"Skipping empty narrative text block after list normalization: {block['id']}")
        else:
            final_blocks.append(block)

    # Merge split Markdown table rows into complete key-value pairs
    for block in final_blocks:
        if block["type"] in ["table", "custom_key_value_table"]:
            lines = block["content"].split("\n")
            fixed_rows = []
            buffer = []
            for line in lines:
                # This regex checks for a line that starts and ends with a pipe and has no other pipes in between
                if re.match(r'^\|\s*[^|]+\s*\|$', line):  # Single column row
                    buffer.append(line.strip('|').strip())
                elif buffer:
                    row = " ".join(buffer)
                    fixed_rows.append(f"| {row} | {line.strip('|').strip()} |")
                    buffer = []
                else:
                    fixed_rows.append(line)
            if buffer:
                # If there's content in the buffer but no subsequent line to merge with, add it with a placeholder
                fixed_rows.append(f"| {buffer[0]} | - |")
            block["content"] = "\n".join(fixed_rows)

    logger.info(f"Structure normalization complete. Final block count: {len(final_blocks)}", extra={"final_block_count": len(final_blocks)})
    return final_blocks
=======
        })

    except Exception as e:
        logger.error(f"Error processing file for layout detection '{file.filename}': {e}",
                     extra={"doc_filename": file.filename, "error_detail": str(e), "event": "layout_detection_failed"})
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document for layout detection: {e}")


def format_key_value_group_as_markdown(key_value_pairs: List[Dict[str, Any]]) -> str:
    """
    Formats a list of key-value dictionaries into a Markdown table string.
    Ensures keys/values are escaped for Markdown table compatibility.
    """
    if not key_value_pairs:
        return ""

    header = "| Parameter | Value |\n|---|---|\n"
    rows = []
    for item in key_value_pairs:
        # Escape pipe characters within content to prevent breaking Markdown table format
        param = item.get("parameter", "").replace('|', '\\|').strip()
        value = item.get("value", "").replace('|', '\\|').strip()
        rows.append(f"| {param} | {value} |")
    
    return header + "\n".join(rows)

def structure_normalizer(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies post-processing rules to a list of detected blocks to improve
    logical grouping, merge fragmented content, and reduce noise.
    """
    logger.info(f"Starting structure normalization for {len(blocks)} blocks.", extra={"initial_block_count": len(blocks)})
    normalized_blocks = []
    i = 0
    while i < len(blocks):
        current_block = blocks[i]
        
        # Rule 1: Merge fragmented text blocks (e.g., lines that should be one paragraph/item)
        # This is a heuristic. Look for short, consecutive narrative/text blocks
        # that appear to be fragments of a single logical item.
        if current_block["type"] in ["narrative_text", "text"] and current_block["content"].strip():
            merged_content = current_block["content"].strip()
            merged_coords = list(current_block["coordinates"]) if current_block["coordinates"] else None
            merged_page = current_block["page"]
            
            j = i + 1
            while j < len(blocks):
                next_block = blocks[j]
                # Check for proximity (same page, close vertically) and similar type
                if next_block["page"] == merged_page and \
                   next_block["type"] in ["narrative_text", "text"] and \
                   next_block["content"].strip():
                    
                    # Calculate vertical distance between current block's bottom and next block's top
                    vertical_distance = 0
                    if merged_coords and next_block["coordinates"]:
                        vertical_distance = next_block["coordinates"][1] - merged_coords[3]
                    
                    # Heuristic for continuation: if current merged content doesn't end with sentence punctuation
                    # AND the vertical distance is small (implies it's part of the same paragraph/item)
                    # OR if the next line starts with a lowercase letter (strong indicator of continuation)
                    if (not re.search(r'[.!?]$', merged_content) and vertical_distance < 20) or \
                       (next_block["content"].strip() and next_block["content"].strip()[0].islower() and vertical_distance < 50): # More lenient for lowercase continuation
                        
                        merged_content += " " + next_block["content"].strip()
                        # Update merged coordinates to encompass both blocks
                        if next_block["coordinates"] and merged_coords:
                            merged_coords[2] = max(merged_coords[2], next_block["coordinates"][2]) # Max X2
                            merged_coords[3] = max(merged_coords[3], next_block["coordinates"][3]) # Max Y2
                        j += 1
                        logger.debug(f"Merged fragmented text block: '{current_block['content'][:30]}...' with '{next_block['content'][:30]}...'")
                    else:
                        break # Not a continuation
                else:
                    break # Not a merge candidate
            
            # Create a new block for the merged content
            new_block = {
                "id": str(uuid.uuid4()),
                "type": "narrative_text", # Consolidated type for merged text
                "content": merged_content,
                "coordinates": merged_coords,
                "page": merged_page,
                "metadata": current_block["metadata"]
            }
            normalized_blocks.append(new_block)
            i = j # Move index past merged blocks
            continue # Continue to next iteration of outer loop

        # Rule 2: Handle repetitive headers/noise (e.g., multiple "TECHNICAL SPECIFICATIONS")
        # If a block is a title and its content is very similar to the previous title, skip it
        if current_block["type"] == "title" and normalized_blocks and normalized_blocks[-1]["type"] == "title":
            # Simple string comparison, could use fuzzy matching for robustness
            if current_block["content"].strip().upper() == normalized_blocks[-1]["content"].strip().upper():
                logger.debug(f"Skipping repetitive title: '{current_block['content'][:50]}...'")
                i += 1
                continue
        
        normalized_blocks.append(current_block)
        i += 1
    
    # Rule 3: Bullet Point Normalization (post-processing on normalized_blocks)
    final_blocks = []
    for block in normalized_blocks:
        if block["type"] == "list":
            # Ensure consistent bullet format, remove leading numbers/noise
            cleaned_lines = []
            for line in block["content"].split('\n'):
                line_strip = line.strip()
                if not line_strip: continue
                # Remove common list prefixes (1), a), -, *, l, I, •)
                # FIX: Replaced '•' with its Unicode escape sequence '\u2022'
                line_strip = re.sub(r'^\s*(\d+\)|\w+\)|\-|\*|\u2022|[lI])\s*', '', line_strip, flags=re.IGNORECASE)
                # Add consistent markdown bullet
                cleaned_lines.append(f"- {line_strip.strip()}")
            if cleaned_lines:
                block["content"] = "\n".join(cleaned_lines)
                final_blocks.append(block)
            else:
                logger.debug(f"Skipping empty list block after normalization: {block['id']}")
        elif block["type"] == "narrative_text" and re.match(r'^\s*(\d+\)|\w+\)|\-|\*|\u2022|[lI])\s*', block["content"].strip()):
            # Catch narrative text that looks like a list item and convert it
            cleaned_lines = []
            for line in block["content"].split('\n'):
                line_strip = line.strip()
                if not line_strip: continue
                # FIX: Replaced '•' with its Unicode escape sequence '\u2022'
                line_strip = re.sub(r'^\s*(\d+\)|\w+\)|\-|\*|\u2022|[lI])\s*', '', line_strip, flags=re.IGNORECASE)
                cleaned_lines.append(f"- {line_strip.strip()}")
            if cleaned_lines:
                block["type"] = "list" # Change type to list
                block["content"] = "\n".join(cleaned_lines)
                final_blocks.append(block)
            else:
                logger.debug(f"Skipping empty narrative text block after list normalization: {block['id']}")
        else:
            final_blocks.append(block)

    # Merge split Markdown table rows into complete key-value pairs
    for block in final_blocks:
        if block["type"] in ["table", "custom_key_value_table"]:
            lines = block["content"].split("\n")
            fixed_rows = []
            buffer = []
            for line in lines:
                # This regex checks for a line that starts and ends with a pipe and has no other pipes in between
                if re.match(r'^\|\s*[^|]+\s*\|$', line):  # Single column row
                    buffer.append(line.strip('|').strip())
                elif buffer:
                    row = " ".join(buffer)
                    fixed_rows.append(f"| {row} | {line.strip('|').strip()} |")
                    buffer = []
                else:
                    fixed_rows.append(line)
            if buffer:
                # If there's content in the buffer but no subsequent line to merge with, add it with a placeholder
                fixed_rows.append(f"| {buffer[0]} | - |")
            block["content"] = "\n".join(fixed_rows)

    logger.info(f"Structure normalization complete. Final block count: {len(final_blocks)}", extra={"final_block_count": len(final_blocks)})
    return final_blocks
>>>>>>> dd6123601b3b1df1e426d6eac04f58ebd5c6ba7f
