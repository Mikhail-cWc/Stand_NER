from typing import List, Optional

from .base_parser import FileParser
from ..instance import Entity, Document


class TextParser(FileParser):
    @staticmethod
    def parse_file(filename: str, content: str, gold_markup: Optional[List[Entity]] = None) -> Document:
        return Document(
            name=filename,
            text=content,
            plaintext=content,
            gold_markup=gold_markup or [],
            metadata={"source_type": "txt"}
        )
