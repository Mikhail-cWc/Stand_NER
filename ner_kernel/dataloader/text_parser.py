from typing import List, Optional, Any

from .base_parser import FileParser
from ..instance import Entity, Document


class TextParser(FileParser):

    def __init__(self, stop_words: Optional[List[str]] = None, language: Optional[str] = None):
        self.stop_words = stop_words
        self.language = language

    def parse_file(self, filename: str, content: str, gold_markup: Optional[List[Entity]] = None) -> Document:
        plaintext = content.strip()
        if self.stop_words:
            filtered_words = [
                word for word in plaintext.split() if word.lower()
                not in self.stop_words
            ]
            plaintext = " ".join(filtered_words)
        return Document(
            name=filename,
            text=content,
            plaintext=plaintext,
            gold_markup=gold_markup or [],
            metadata={"source_type": "txt"}
        )
