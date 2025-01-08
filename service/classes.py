from pydantic import BaseModel
from typing import List


class NERRequest(BaseModel):
    text: str
    model_name: str = "spacy"


class EntityResponse(BaseModel):
    entity: str
    start_offset: int
    end_offset: int
    text: str


class NERResponse(BaseModel):
    entities: List[EntityResponse]
