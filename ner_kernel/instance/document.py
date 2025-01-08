from dataclasses import dataclass, field
from typing import Optional, Dict, List
from .entity import Entity


@dataclass
class Document:
    name: Optional[str] = None

    text: str = ""
    plaintext: str = ""

    gold_markup: List[Entity] = field(default_factory=list)
    pred_markup: List[Entity] = field(default_factory=list)

    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text or not self.plaintext:
            raise ValueError("Пустой текст недопустим.")
