# In rag_llm_json_utils.py

import re
import json
import logging
import unicodedata # For robust string cleaning
from typing import Any, Dict, List, Optional, Tuple, Union

# Robust import for jsonschema with fallback
# jsonschema is used for validating parsed JSON against a predefined schema,
# ensuring the structure and types of extracted metadata are correct.
def _import_jsonschema():
    try:
        import jsonschema
        return jsonschema
    except ImportError:
        # Log a warning if jsonschema is not installed, as validation will be skipped.
        logger.warning("WARNING: 'jsonschema' not found. LLM response validation will be skipped. Install with 'pip install jsonschema'.")
        return None
jsonschema = _import_jsonschema()

# Initialize logger for this module
logger = logging.getLogger(__name__)

def _clean_json_string(raw_json_str: str) -> str:
    """
    Cleans a raw JSON string by removing common LLM-related artifacts
    like markdown code block fences and attempting to fix some common JSON issues.
    This function is crucial for handling malformed or incomplete JSON outputs from LLMs.

    Args:
        raw_json_str (str): The raw string response from the LLM, potentially containing JSON.

    Returns:
        str: A cleaned string that is more likely to be valid JSON.
    """
    if not isinstance(raw_json_str, str):
        logger.warning(f"Expected string for _clean_json_string, got {type(raw_json_str)}. Returning empty string.")
        return ""

    # Remove markdown code block fences (```json, ```)
    cleaned_str = re.sub(r'```json\n|```', '', raw_json_str).strip()

    # Remove any conversational text before or after the JSON structure.
    # Try to find the first '{' or '[' and the last '}' or ']'.
    first_brace = cleaned_str.find('{')
    first_bracket = cleaned_str.find('[')
    
    start_index = -1
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_index = first_brace
    elif first_bracket != -1:
        start_index = first_bracket

    if start_index != -1:
        # Find the corresponding closing brace/bracket from the end
        last_brace = cleaned_str.rfind('}')
        last_bracket = cleaned_str.rfind(']')

        end_index = -1
        if last_brace != -1 and (last_bracket == -1 or last_brace > last_bracket):
            end_index = last_brace
        elif last_bracket != -1:
            end_index = last_bracket
        
        if end_index != -1 and end_index > start_index:
            cleaned_str = cleaned_str[start_index : end_index + 1]
        else:
            logger.warning(f"Could not find matching closing brace/bracket. Snippet: '{cleaned_str[:100]}'")
            # Fallback to aggressive stripping if structure is very broken
            cleaned_str = re.sub(r'^[^{[]*|[^}\]]*$', '', cleaned_str).strip()
    else:
        logger.warning(f"No starting JSON brace/bracket found. Snippet: '{cleaned_str[:100]}'")
        return "" # No JSON structure found at all

    # Remove trailing commas before closing braces/brackets, which can cause JSON.parse errors
    cleaned_str = re.sub(r',\s*([}\]])', r'\1', cleaned_str)

    # Replace newlines within JSON strings (if any) with \n to keep it valid single line for parsing
    # This is less common if LLM is well-behaved but good for robustness
    # Using a more robust approach to handle newlines in JSON strings
    def replace_newlines(match):
        # Replace newlines with \n in the matched string
        return match.group(0).replace('\n', '\\n')
    
    # Find all quoted strings and process their content
    cleaned_str = re.sub(r'"(?:\\.|[^"\\])*"', replace_newlines, cleaned_str)
    
    # Remove control characters that might break JSON parsing
    cleaned_str = ''.join(c for c in cleaned_str if unicodedata.category(c)[0]!='C')

    return cleaned_str


def safe_parse_llm_json_response(json_string: str) -> Dict[str, Any]:
    """
    Safely parses a JSON string received from an LLM, applying cleaning and
    structural recovery attempts. Adds robust fallback for power consumption extraction.
    """
    import re
    
    def extract_power_via_regex(text):
        match = re.search(r'(?:Power Consumption|Total Power Consumption).*?(\d+\s*[-/]?\s*\d*)\s*W', text, re.IGNORECASE)
        if match:
            power = match.group(1).replace(' ', '').replace('/', '-')
            return power + 'W'
        return None

    # Original logic...
    cleaned = _clean_json_string(json_string)
    if not cleaned:
        # Fallback: try regex if prose mentions power consumption
        fallback_metadata = {}
        if "power consumption" in json_string.lower():
            power = extract_power_via_regex(json_string)
            if power:
                fallback_metadata["power_consumption"] = power
        return fallback_metadata
    try:
        parsed = json.loads(cleaned)
        return parsed
    except Exception:
        # Fallback: try regex if prose mentions power consumption
        fallback_metadata = {}
        if "power consumption" in json_string.lower():
            power = extract_power_via_regex(json_string)
            if power:
                fallback_metadata["power_consumption"] = power
        return fallback_metadata

    """
    Safely parses a JSON string received from an LLM, applying cleaning and
    structural recovery attempts.

    Args:
        json_string (str): The raw string response from the LLM.

    Returns:
        Dict[str, Any]: The parsed JSON dictionary, or an empty dictionary if parsing fails.
    """
    if not isinstance(json_string, str):
        logger.error(f"Expected string for safe_parse_llm_json_response, got {type(json_string)}. Returning empty dict.")
        return {}

    cleaned_json_str = _clean_json_string(json_string)

    if not cleaned_json_str:
        logger.warning("Cleaned JSON string is empty. Cannot parse.")
        return {}

    try:
        # Attempt to parse the cleaned string directly
        parsed_json = json.loads(cleaned_json_str)
        logger.info("Successfully parsed LLM response directly as JSON.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode JSON directly: {e}. Trying structural recovery. Snippet: '{cleaned_json_str[:100]}'")
        # Attempt structural recovery if direct parsing fails
        try:
            # This is a more aggressive attempt to find a valid JSON object
            # by looking for the first and last curly braces.
            match = re.search(r'\{.*\}', cleaned_json_str, re.DOTALL)
            if match:
                recovered_json_str = match.group(0)
                parsed_json = json.loads(recovered_json_str)
                logger.info("Successfully recovered and parsed JSON via structural search.")
                return parsed_json
            else:
                logger.warning("Structural recovery failed: No valid JSON structure found. Snippet: '%s'", cleaned_json_str[:100])
                return {}
        except json.JSONDecodeError as e_recovery:
            logger.error(f"Structural recovery also failed: {e_recovery}. Snippet: '{cleaned_json_str[:100]}'")
            return {}
        except Exception as e_general:
            logger.error(f"An unexpected error occurred during JSON recovery: {e_general}. Snippet: '{cleaned_json_str[:100]}'")
            return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}. Returning empty dict.", exc_info=True)
        return {}


def extract_kv_pairs(text: str) -> Dict[str, Any]:
    """
    Extracts key-value pairs from a text string, typically as a fallback
    when JSON parsing fails. It tries to find common patterns like "Key: Value"
    or "Key = Value".
    """
    extracted_data = {}
    # Pattern to find "Key: Value" or "Key = Value"
    # Key: alphanumeric, spaces, hyphens, underscores
    # Value: anything until a new line or another key pattern starts
    patterns = [
        re.compile(r"^\s*([A-Za-z0-9\s_-]+?)\s*:\s*(.+?)(?=\n\s*[A-Za-z0-9\s_-]+?\s*:|\n\s*[A-Za-z0-9\s_-]+?\s*=|\Z)", re.MULTILINE),
        re.compile(r"^\s*([A-Za-z0-9\s_-]+?)\s*=\s*(.+?)(?=\n\s*[A-Za-z0-9\s_-]+?\s*:|\n\s*[A-Za-z0-9\s_-]+?\s*=|\Z)", re.MULTILINE)
    ]

    for pattern in patterns:
        for match in pattern.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            # Basic cleaning for key and value
            key = re.sub(r'\s+', '_', key).lower()
            
            # Attempt to convert value to int/float if it looks like a number
            try:
                if '.' in value:
                    extracted_data[key] = float(value)
                else:
                    extracted_data[key] = int(value)
            except ValueError:
                extracted_data[key] = value # Keep as string if not a number
    return extracted_data


def flatten_json(data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flattens a nested JSON dictionary into a single-level dictionary
    with concatenated keys.
    """
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings for flattening
            items.append((new_key, ", ".join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)


def filter_noisy_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters out noisy or irrelevant key-value pairs from a metadata dictionary.
    This function is designed to clean up metadata extracted by LLMs or other
    inference methods, removing generic or placeholder values.

    Args:
        metadata (Dict[str, Any]): The input metadata dictionary.

    Returns:
        Dict[str, Any]: The cleaned metadata dictionary.
    """
    cleaned_metadata = {}
    noisy_values = {
        "none", "n/a", "unknown", "not applicable", "null", "false", "true",
        "undefined", "value", "string", "text", "number", "boolean", "array",
        "object", "example", "product", "model", "series", "unit", "light",
        "lamp", "box", "cleaner", "socket", "pole", "fan", "driver", "sensor",
        "kiosk", "thermometer", "workstation", "tower", "dispenser", "machine",
        "e", # Added 'e' based on your previous log for 'product_name'
        "general document", "general", "document", "report", "brochure", # Added common document/section types if they appear as product_name
        "technical specification", "features", "applications", "table", "list_item", "title", "hr_policy"
    }

    for key, value in metadata.items():
        if isinstance(value, str):
            normalized_value = value.lower().strip()
            # Remove values that are very short and generic, or match known noisy terms
            if normalized_value in noisy_values or len(normalized_value) < 2:
                logger.debug(f"Filtering noisy metadata: Key='{key}', Value='{value}'")
                continue
            # Remove values that are just punctuation or special characters
            if re.fullmatch(r"[^\w\s]+", normalized_value):
                logger.debug(f"Filtering noisy metadata (punctuation): Key='{key}', Value='{value}'")
                continue
            cleaned_metadata[key] = value
        elif isinstance(value, (int, float, bool)):
            # Keep numeric and boolean values as they are typically not noisy
            cleaned_metadata[key] = value
        elif isinstance(value, list):
            # Convert list to a comma-separated string for ChromaDB compatibility
            cleaned_metadata[key] = ", ".join(map(str, value))
            logger.debug(f"Converted list metadata for key '{key}' to string: '{cleaned_metadata[key]}'")
        elif isinstance(value, dict):
            # Recursively clean nested dictionaries
            cleaned_dict = filter_noisy_metadata(value)
            if cleaned_dict:
                cleaned_metadata[key] = cleaned_dict
        else:
            # Keep other types as is
            cleaned_metadata[key] = value
            
    return cleaned_metadata


def normalize_and_filter_metadata(llm_json: Dict[str, Any], doc_type: str = '', strict=False):
    """
    Normalizes metadata fields (e.g., product_name, document_type, section_type) for consistent retrieval.
    Also normalizes keys like 'Power Consumption (W)' to 'power_consumption'.
    Enforces schema for power_consumption: must be float/int or 'unknown'.
    """
    final_metadata = {}
    if not isinstance(llm_json, dict):
        logger.warning(f"Invalid LLM JSON response for metadata normalization: {llm_json}. Returning empty dict.")
        return {}

    # Step 1: Flatten the JSON to a single level
    flat_json = flatten_json(llm_json)

    # Step 2: Filter out noisy/irrelevant fields
    if strict:
        cleaned_metadata = filter_noisy_metadata(flat_json)
    else:
        # Only remove fields if value is None or empty string, keep most fields
        cleaned_metadata = {k: v for k, v in flat_json.items() if v is not None and v != ''}

    # Step 3: Normalize field names and values (e.g., lowercasing, stripping)
    normalized_metadata = {}
    for k, v in cleaned_metadata.items():
        key = k.lower().strip()
        if isinstance(v, str):
            value = v.strip()
        else:
            value = v
        normalized_metadata[key] = value

    # Step 4: Define patterns for specific fields
    numerical_fields_patterns = {
        "ip_rating": r"ip(\d{2,3})",
        "efficiency": r"(\d+(\.\d+)?)\s*(lumens\s*per\s*watt|lm/w)",
        "operating_temperature_range": r"(-?\d+)\s*°c",
        "thd": r"(\d+(\.\d+)?)\s*%",
        "power_factor": r"(\d+(\.\d+)?)"
    }
    # Now, you can use this dictionary in your logic, for example:
    for key, pattern_str in numerical_fields_patterns.items():
        if key in final_metadata and isinstance(final_metadata[key], str):
            value_str = final_metadata[key]
            try:
                match = re.search(pattern_str, value_str, re.IGNORECASE)
                if match:
                    try:
                        num_str = match.group(1).replace(',', '')
                        if '.' in num_str:
                            final_metadata[key] = float(num_str)
                        else:
                            final_metadata[key] = int(num_str)
                    except ValueError:
                        pass  # Keep as string if conversion fails
                else:
                    # If no pattern matched, try direct conversion if it's purely numeric
                    try:
                        if '.' in value_str:
                            final_metadata[key] = float(value_str)
                        else:
                            final_metadata[key] = int(value_str)
                    except ValueError:
                        pass  # Keep as string if not directly convertible
            except Exception as e:
                logger.error(f"Error converting numerical field '{key}': {e}")

    for key, pattern_str in numerical_fields_patterns.items():
        if key in final_metadata and isinstance(final_metadata[key], str):
            value_str = final_metadata[key]
            try:
                match = re.search(pattern_str, value_str, re.IGNORECASE)
                if match:
                    try:
                        num_str = match.group(1).replace(',', '')
                        if '.' in num_str:
                            final_metadata[key] = float(num_str)
                        else:
                            final_metadata[key] = int(num_str)
                    except ValueError:
                        pass # Keep as string if conversion fails
                else:
                    # If no pattern matched, try direct conversion if it's purely numeric
                    try:
                        if '.' in value_str:
                            final_metadata[key] = float(value_str)
                        else:
                            final_metadata[key] = int(value_str)
                    except ValueError:
                        pass # Keep as string if not directly convertible
            except Exception as e:
                logger.error(f"Error converting numerical field '{key}': {e}")

    # Special handling for "specific_wattage" and "specific_ip_rating"
    for specific_key in ["specific_wattage", "specific_ip_rating"]:
        if specific_key in final_metadata and isinstance(final_metadata[specific_key], str):
            try:
                match = re.search(r"(\d+(\.\d+)?)", final_metadata[specific_key])
                if match:
                    num_str = match.group(1)
                    if '.' in num_str:
                        final_metadata[specific_key] = float(num_str)
                    else:
                        final_metadata[specific_key] = int(num_str)
            except ValueError:
                pass # Keep as string if conversion fails

    # Step 5: Validate against a schema if jsonschema is available
    if jsonschema:
        # Define a simple schema for common metadata fields
        # This schema can be expanded as needed.
        metadata_schema = {
            "type": "object",
            "properties": {
                "product_name": {"type": ["string", "null"]},
                "product_type": {"type": ["string", "null"]},
                "document_type": {"type": ["string", "null"]},
                "section_type": {"type": ["string", "null"]},
                "query_type": {"type": ["string", "null"]},
                "domain": {"type": ["string", "null"]},
                "specific_document_type": {"type": ["string", "null"]},
                "attributes": {"type": "string"}, # Attributes are now comma-separated string
                "specific_wattage": {"type": ["number", "null"]},
                "specific_ip_rating": {"type": ["number", "null"]},
                "wattage": {"type": ["number", "string", "null"]}, # Allow string if not parsed
                "voltage": {"type": ["number", "string", "null"]},
                "ip_rating": {"type": ["number", "string", "null"]},
                "lumens_output": {"type": ["number", "string", "null"]},
                "lifespan": {"type": ["number", "string", "null"]},
                "efficiency": {"type": ["number", "string", "null"]},
                "operating_temperature_range": {"type": ["number", "string", "null"]},
                "thd": {"type": ["number", "string", "null"]},
                "power_factor": {"type": ["number", "string", "null"]},
                "colors": {"type": "string"}, # Converted to string
                "component": {"type": "string"}, # Converted to string
                "power_source": {"type": ["string", "null"]},
                "dimming_capabilities": {"type": ["string", "null"]},
                "table_column_names": {"type": "string"}, # Converted to string
                "source_file_name": {"type": "string"},
                "user": {"type": "string"},
                "chunk_id": {"type": "string"},
                "chunk_length": {"type": "number"},
                "chunk_hash": {"type": "string"},
                "original_element_type": {"type": "string"},
                "source_doc_type": {"type": "string"},
                "document_title": {"type": ["string", "null"]}
            },
            "additionalProperties": True # Allow other properties not explicitly defined
        }
        try:
            jsonschema.validate(instance=final_metadata, schema=metadata_schema)
            logger.info("Metadata schema validation successful.")
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Metadata schema validation failed for doc_type '{doc_type}': {e.message}. Data: {final_metadata}")
            # Attempt to fix common validation issues (e.g., lists that should be strings)
            for prop, schema_prop in metadata_schema["properties"].items():
                if prop in final_metadata and "type" in schema_prop:
                    expected_types = schema_prop["type"] if isinstance(schema_prop["type"], list) else [schema_prop["type"]]
                    
                    # If a list is found where a string is expected, convert it
                    if "string" in expected_types and isinstance(final_metadata[prop], list):
                        final_metadata[prop] = ", ".join(map(str, final_metadata[prop]))
                        logger.debug(f"Auto-corrected list to string for '{prop}' during validation.")
                    # If a number is found where string/null is expected (and vice-versa, if it's a known problematic field)
                    # This is covered by the numerical parsing above, but good to have a fallback.
                    elif "number" in expected_types and isinstance(final_metadata[prop], str):
                        try:
                            if '.' in final_metadata[prop]:
                                final_metadata[prop] = float(final_metadata[prop])
                            else:
                                final_metadata[prop] = int(final_metadata[prop])
                            logger.debug(f"Auto-corrected string to number for '{prop}' during validation.")
                        except ValueError:
                            pass # Keep as string if conversion fails
                    elif "string" in expected_types and isinstance(final_metadata[prop], (int, float)):
                         # If a number is found where a string is expected, convert to string
                         final_metadata[prop] = str(final_metadata[prop])
                         logger.debug(f"Auto-corrected number to string for '{prop}' during validation.")


    return final_metadata


def extract_metadata_with_schema(llm_json: Dict[str, Any], doc_type: str = "default", llm_call_fn=None) -> Dict[str, Any]:
    """
    Extracts and structures relevant fields from the unified LLM JSON response
    and applies schema-based normalization and filtering. This function acts as a wrapper
    around the `normalize_and_filter_metadata` pipeline.
    """
    processed_metadata = normalize_and_filter_metadata(llm_json, doc_type)
    # Enforce JSON schema for power_consumption: float/int or 'unknown'
    from jsonschema import validate, ValidationError
    power_schema = {
        "type": ["number", "string"],
        "oneOf": [
            {"type": "number"},
            {"type": "string", "enum": ["unknown"]}
        ]
    }
    if "power_consumption" in processed_metadata:
        try:
            validate(instance=processed_metadata["power_consumption"], schema=power_schema)
        except ValidationError:
            processed_metadata["power_consumption"] = "unknown"
    return processed_metadata

    """
    Extracts and structures relevant fields from the unified LLM JSON response
    and applies schema-based normalization and filtering. This function acts as a wrapper
    around the `normalize_and_filter_metadata` pipeline.

    Args:
        llm_json (Dict[str, Any]): The initial JSON dictionary parsed from the LLM response.
        doc_type (str): The type of document, used to guide normalization and schema validation.
        llm_call_fn (Callable, optional): A function to call the LLM, if needed for further refinement.
                                         Not directly used in this function's current implementation,
                                         but kept for API consistency.

    Returns:
        Dict[str, Any]: The fully processed, normalized, and filtered metadata dictionary.
    """
    if not isinstance(llm_json, dict):
        logger.warning(f"Invalid LLM JSON response for metadata parsing: {llm_json}. Returning default structure.")
        return {}

    # The core logic of metadata processing is encapsulated in normalize_and_filter_metadata.
    processed_metadata = normalize_and_filter_metadata(llm_json, doc_type)

    return processed_metadata
