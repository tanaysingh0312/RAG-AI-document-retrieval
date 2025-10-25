from rapidfuzz import fuzz
# Levenshtein is not strictly needed if only token_set_ratio is used,
# but keeping it as per your instruction not to remove anything.
from rapidfuzz.distance import Levenshtein

# Import logging here to make it available for the debug statement
import logging
logger = logging.getLogger(__name__)

def fuzzy_match_product(query_name: str, metadata_name: str, threshold: int = 80) -> bool:
    """
    Compares two product names using the Token Set Ratio from RapidFuzz.
    This method is robust to differences in word order and additional words
    in one of the strings, which is common in product names.

    Args:
        query_name (str): The product name from the user's query or extracted attributes.
        metadata_name (str): The product name from the document's metadata.
        threshold (int): The minimum similarity score (0-100) required for a match.
                         Default is 80, which can be tuned based on desired strictness.

    Returns:
        bool: True if the similarity score is above the threshold, False otherwise.
    """
    # Ensure both inputs are valid strings before processing
    if not isinstance(query_name, str) or not isinstance(metadata_name, str) or not query_name or not metadata_name:
        # Log a debug message if inputs are invalid, for easier debugging
        logger.debug(f"Invalid input for fuzzy_match_product: query_name='{query_name}', metadata_name='{metadata_name}'")
        return False

    # Normalize both strings to lowercase to ensure case-insensitivity.
    # token_set_ratio inherently handles tokenization, so we just pass the lowercased strings.
    score = fuzz.token_set_ratio(query_name.lower(), metadata_name.lower())

    # Always log the score for debugging and tuning purposes
    logger.debug(f"Fuzzy match '{query_name}' vs '{metadata_name}': Score = {score} (Threshold = {threshold})")

    return score >= threshold

