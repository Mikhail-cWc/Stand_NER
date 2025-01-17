from pydantic import BaseModel
from typing import List


class NERRequest(BaseModel):
    text: str
    framework: str = "spacy"
    model_name: str = "ru_core_news_sm"


class EntityResponse(BaseModel):
    entity: str
    start_offset: int
    end_offset: int
    text: str


class NERResponse(BaseModel):
    entities: List[EntityResponse]
