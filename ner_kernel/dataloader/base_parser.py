from typing import List, Optional

from ..instance import Entity, Document


class FileParser:
    @staticmethod
    def parse_file(filename: str, content: str, gold_markup: Optional[List[Entity]] = None) -> Document:
        raise NotImplementedError("Subclasses must implement parse_file method")
