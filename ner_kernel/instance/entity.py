from dataclasses import dataclass


@dataclass
class Entity:
    entity: str
    start_offset: int
    end_offset: int
    text: str
