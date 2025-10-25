<<<<<<< HEAD
# semantic_chunker.py

import re
import logging
from typing import List, Dict, Optional, Any

# Import Unstructured elements for type hinting
try:
    from unstructured.documents.elements import Element, Text, NarrativeText, Title, ListItem, Table
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    logging.warning("Unstructured library not found. Semantic chunking for Elements will be limited.")
    UNSTRUCTURED_AVAILABLE = False
    class Element: pass # Dummy class for type hinting if Unstructured is not installed


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _split_text_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Splits a long text into smaller chunks based on character limits and overlap.
    Tries to split at natural sentence or paragraph breaks.
    """
    if not text:
        return []

    chunks = []
    current_chunk = []
    current_len = 0

    # Split by paragraphs first
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para) # Split by sentences
        for sentence in sentences:
            sentence_len = len(sentence) + 1 # +1 for space/newline

            if current_len + sentence_len <= max_chars:
                current_chunk.append(sentence)
                current_len += sentence_len
            else:
                # Current sentence doesn't fit, finalize current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())

                # Start a new chunk with overlap
                if overlap > 0 and chunks:
                    # Take last part of previous chunk for overlap
                    overlap_text = chunks[-1][-overlap:]
                    current_chunk = [overlap_text.strip(), sentence]
                    current_len = len(overlap_text.strip()) + sentence_len
                else:
                    current_chunk = [sentence]
                    current_len = sentence_len

                # If a single sentence is larger than max_chars, split it forcibly
                while current_len > max_chars:
                    chunks.append(" ".join(current_chunk).strip())
                    current_chunk = [current_chunk[-1][max_chars - overlap:].strip()] if overlap > 0 else []
                    current_len = len(" ".join(current_chunk))


    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Filter out empty or very small chunks
    return [chunk for chunk in chunks if len(chunk) >= 30]


def semantic_chunk_unstructured(elements: List[Element], max_chars: int = 800, overlap: int = 50) -> List[Any]:
    """
    Chunks Unstructured.io elements semantically, respecting element types.
    Combines small elements and splits large ones.
    Returns a list of chunk objects (which can be elements or custom dicts with text).
    """
    if not UNSTRUCTURED_AVAILABLE:
        logger.error("Unstructured library not available. Cannot perform semantic_chunk_unstructured.")
        return []

    if not elements:
        return []

    chunks = []
    current_chunk_text = ""
    current_chunk_elements = []
    current_chunk_len = 0

    for i, element in enumerate(elements):
        element_text = str(element) # Convert element to string for its text content
        element_len = len(element_text)

        # Check if adding this element exceeds max_chars
        if current_chunk_len + element_len <= max_chars:
            current_chunk_text += ("\n\n" if current_chunk_text else "") + element_text
            current_chunk_elements.append(element)
            current_chunk_len += element_len
        else:
            # Current element doesn't fit, finalize current chunk
            if current_chunk_text:
                chunks.extend(_split_text_into_chunks(current_chunk_text, max_chars, overlap))
                # Optionally, you could try to preserve element boundaries more strictly here
                # by adding the whole element if it's small, or splitting it.
                # For simplicity, we split the combined text.

            # Start a new chunk with overlap from the previous combined text
            if overlap > 0 and chunks:
                overlap_text = chunks[-1][-overlap:]
                current_chunk_text = overlap_text.strip() + ("\n\n" if overlap_text.strip() else "") + element_text
                current_chunk_len = len(current_chunk_text)
            else:
                current_chunk_text = element_text
                current_chunk_len = element_len

            current_chunk_elements = [element] # Reset elements for the new chunk

            # If the current element itself is larger than max_chars, split it
            if element_len > max_chars:
                chunks.extend(_split_text_into_chunks(element_text, max_chars, overlap))
                current_chunk_text = "" # Reset as it was fully processed
                current_chunk_elements = []
                current_chunk_len = 0

    # Add the last accumulated chunk
    if current_chunk_text:
        chunks.extend(_split_text_into_chunks(current_chunk_text, max_chars, overlap))

    # Return chunks as simple strings for now, as metadata extraction will re-process them.
    # If you need to retain Unstructured Element objects, this function would need to return
    # a list of Element objects or custom chunk objects with associated metadata.
    # For this plan, we return raw text chunks, and metadata is re-inferred from text.
    return [Text(text=chunk) for chunk in chunks if chunk.strip()] # Wrap as Unstructured Text elements


def semantic_chunk_plain_text(text: str, max_chars: int = 800, overlap: int = 50) -> List[Text]:
    """
    Chunks a plain text string semantically.
    """
    if not text:
        return []
    chunks = _split_text_into_chunks(text, max_chars, overlap)
    return [Text(text=chunk) for chunk in chunks if chunk.strip()] # Wrap as Unstructured Text elements
=======
# semantic_chunker.py

import re
import logging
from typing import List, Dict, Optional, Any

# Import Unstructured elements for type hinting
try:
    from unstructured.documents.elements import Element, Text, NarrativeText, Title, ListItem, Table
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    logging.warning("Unstructured library not found. Semantic chunking for Elements will be limited.")
    UNSTRUCTURED_AVAILABLE = False
    class Element: pass # Dummy class for type hinting if Unstructured is not installed


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _split_text_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Splits a long text into smaller chunks based on character limits and overlap.
    Tries to split at natural sentence or paragraph breaks.
    """
    if not text:
        return []

    chunks = []
    current_chunk = []
    current_len = 0

    # Split by paragraphs first
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para) # Split by sentences
        for sentence in sentences:
            sentence_len = len(sentence) + 1 # +1 for space/newline

            if current_len + sentence_len <= max_chars:
                current_chunk.append(sentence)
                current_len += sentence_len
            else:
                # Current sentence doesn't fit, finalize current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())

                # Start a new chunk with overlap
                if overlap > 0 and chunks:
                    # Take last part of previous chunk for overlap
                    overlap_text = chunks[-1][-overlap:]
                    current_chunk = [overlap_text.strip(), sentence]
                    current_len = len(overlap_text.strip()) + sentence_len
                else:
                    current_chunk = [sentence]
                    current_len = sentence_len

                # If a single sentence is larger than max_chars, split it forcibly
                while current_len > max_chars:
                    chunks.append(" ".join(current_chunk).strip())
                    current_chunk = [current_chunk[-1][max_chars - overlap:].strip()] if overlap > 0 else []
                    current_len = len(" ".join(current_chunk))


    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Filter out empty or very small chunks
    return [chunk for chunk in chunks if len(chunk) >= 30]


def semantic_chunk_unstructured(elements: List[Element], max_chars: int = 800, overlap: int = 50) -> List[Any]:
    """
    Chunks Unstructured.io elements semantically, respecting element types.
    Combines small elements and splits large ones.
    Returns a list of chunk objects (which can be elements or custom dicts with text).
    """
    if not UNSTRUCTURED_AVAILABLE:
        logger.error("Unstructured library not available. Cannot perform semantic_chunk_unstructured.")
        return []

    if not elements:
        return []

    chunks = []
    current_chunk_text = ""
    current_chunk_elements = []
    current_chunk_len = 0

    for i, element in enumerate(elements):
        element_text = str(element) # Convert element to string for its text content
        element_len = len(element_text)

        # Check if adding this element exceeds max_chars
        if current_chunk_len + element_len <= max_chars:
            current_chunk_text += ("\n\n" if current_chunk_text else "") + element_text
            current_chunk_elements.append(element)
            current_chunk_len += element_len
        else:
            # Current element doesn't fit, finalize current chunk
            if current_chunk_text:
                chunks.extend(_split_text_into_chunks(current_chunk_text, max_chars, overlap))
                # Optionally, you could try to preserve element boundaries more strictly here
                # by adding the whole element if it's small, or splitting it.
                # For simplicity, we split the combined text.

            # Start a new chunk with overlap from the previous combined text
            if overlap > 0 and chunks:
                overlap_text = chunks[-1][-overlap:]
                current_chunk_text = overlap_text.strip() + ("\n\n" if overlap_text.strip() else "") + element_text
                current_chunk_len = len(current_chunk_text)
            else:
                current_chunk_text = element_text
                current_chunk_len = element_len

            current_chunk_elements = [element] # Reset elements for the new chunk

            # If the current element itself is larger than max_chars, split it
            if element_len > max_chars:
                chunks.extend(_split_text_into_chunks(element_text, max_chars, overlap))
                current_chunk_text = "" # Reset as it was fully processed
                current_chunk_elements = []
                current_chunk_len = 0

    # Add the last accumulated chunk
    if current_chunk_text:
        chunks.extend(_split_text_into_chunks(current_chunk_text, max_chars, overlap))

    # Return chunks as simple strings for now, as metadata extraction will re-process them.
    # If you need to retain Unstructured Element objects, this function would need to return
    # a list of Element objects or custom chunk objects with associated metadata.
    # For this plan, we return raw text chunks, and metadata is re-inferred from text.
    return [Text(text=chunk) for chunk in chunks if chunk.strip()] # Wrap as Unstructured Text elements


def semantic_chunk_plain_text(text: str, max_chars: int = 800, overlap: int = 50) -> List[Text]:
    """
    Chunks a plain text string semantically.
    """
    if not text:
        return []
    chunks = _split_text_into_chunks(text, max_chars, overlap)
    return [Text(text=chunk) for chunk in chunks if chunk.strip()] # Wrap as Unstructured Text elements
>>>>>>> dd6123601b3b1df1e426d6eac04f58ebd5c6ba7f
