import re
from typing import Optional, Dict, Any

def parse_power_consumption(value: Any) -> Optional[float]:
    """
    Parses a power consumption string (e.g., '15-40W', '20W', 'unknown') into a single float value.
    For ranges, it returns the average. Returns None for invalid or 'unknown' inputs.
    """
    if not isinstance(value, str):
        return None
    value = value.lower().strip()
    if not value or value == 'unknown':
        return None
    # Handle ranges like '15-40W'
    match = re.match(r'(\d+\.?\d*)[-–](\d+\.?\d*)w?', value)
    if match:
        try:
            low = float(match.group(1))
            high = float(match.group(2))
            return (low + high) / 2.0
        except (ValueError, IndexError):
            return None
    # Handle single values like '20W'
    match = re.match(r'(\d+\.?\d*)w?', value)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def sanitize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes a metadata dictionary to ensure all values are compatible with ChromaDB.
    - Replaces None with safe defaults (empty string or 0.0).
    - Parses 'power_consumption' into a float.
    - Ensures all values are str, int, float, or bool.
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            if key in ["power_consumption", "wattage"]:
                sanitized[key] = 0.0
            else:
                sanitized[key] = ""
            continue
        if key in ["power_consumption", "wattage"]:
            parsed_power = parse_power_consumption(str(value))
            sanitized[key] = parsed_power if parsed_power is not None else 0.0
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value) # Fallback to string conversion
    return sanitized

def extract_metadata_with_llm(text, chat_history=None, llm_call_fn=None):
    """
    Calls the LLM to extract unified structured metadata from a document chunk and chat history using a single unified prompt.
    Returns a dict with unified keys: product_name, product_type, attributes, history_context, etc.
    """
    from unified_metadata_prompt import build_unified_metadata_prompt
    from rag_llm_json_utils import safe_parse_llm_json_response
    if llm_call_fn is None:
        from .rag import ask_llm_model_mistral  # fallback to local import if not provided
        llm_call_fn = ask_llm_model_mistral
    prompt = build_unified_metadata_prompt(query=text, chat_history=chat_history or "")
    prompt += "\n\nRespond ONLY in compact JSON format. Do not include any explanatory text or markdown formatting."
    llm_response = llm_call_fn(prompt)
    metadata = safe_parse_llm_json_response(llm_response)
    return metadata
