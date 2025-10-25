from typing import Optional, Dict, Any
from pydantic import BaseModel

class MetadataSchema(BaseModel):
    query: str
    product_name: Optional[str]
    product_type: Optional[str]
    attributes: Optional[Dict[str, Any]]
    history_context: Optional[Dict[str, Any]]
    # Add more fields as needed for downstream use
