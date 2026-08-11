from pydantic import BaseModel, Field
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
    top_k: int = Field(default = None)
    top_p: float = Field(default = None)
    temperature: float = Field(default = 1, ge = 0)

class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default = 50, ge = 0)
    use_cache: bool = Field(default = True)
    do_sample: bool = Field(default = False)
    sampling_config: SamplingConfig = Field(default = None)

class GenerateRequestDTO(BaseModel):
    prompt: str
    generation_config: GenerationConfig = Field(default_factory = GenerationConfig)