from typing import List, Optional

from .base_parser import FileParser
from ..instance import Entity, Document


class HtmlParser(FileParser):
    @staticmethod
    def parse_file(filename: str, content: str, gold_markup: Optional[List[Entity]] = None) -> Document:
        soup = BeautifulSoup(content, "html.parser")
        clean_text = soup.get_text()

        return Document(
            name=filename,
            text=content,
            plaintext=clean_text,
            gold_markup=gold_markup or [],
            metadata={"source_type": "html"}
        )
