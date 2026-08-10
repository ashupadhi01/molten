from pydantic import BaseModel
from typing import Optional
from enum import Enum

class FinishReason(str, Enum):
    EOS = "EOS"
    MAX_TOKEN_REACHED = "MAX_TOKEN_REACHED"
    ERROR = "ERROR"

class EventType(str, Enum):
    TOKEN = "TOKEN"
    COMPLETION = "COMPLETION"
    ERROR = "ERROR"

class GenerationEvent(BaseModel):
    token: Optional[str] = None
    ttft: Optional[float] = None
    itl: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    average_tps: Optional[float]= None
    total_generation_time: Optional[float] = None
    finish_reason: Optional[FinishReason] = None
    event_type: EventType


class SamplingConfig(BaseModel):
    top_k: int = None
    top_p: float = None
    do_sample: bool = False
    max_new_tokens: int = 40
    temperature: float = 1