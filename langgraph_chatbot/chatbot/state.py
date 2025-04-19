from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatState(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    current_response: Optional[str] = None
    uploaded_file_data: Optional[Any] = None
