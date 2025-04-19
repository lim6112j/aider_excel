from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class SettlementSheets(BaseModel):
    sales: str
    policy: str


class ChatState(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    current_response: Optional[str] = None
    uploaded_file_data: Optional[Any] = None
    excel_analysis_result: Optional[str] = None
    excel_formula: Optional[str] = None
    settlements_sheets: Dict[SettlementSheets] = None
