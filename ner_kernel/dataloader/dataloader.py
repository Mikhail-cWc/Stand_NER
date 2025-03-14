import os
import json
from typing import List, Optional, Dict

from bs4 import BeautifulSoup

from .base_parser import FileParser
from .text_parser import TextParser
from .html_parser import HtmlParser
from ..instance import Entity, Document


class DataLoader:

    def __init__(
        self,
        path_to_files: Optional[str] = None,
        path_to_markups: Optional[str] = None
    ):
        self.path_to_files = path_to_files
        self.path_to_markups = path_to_markups

        self.parsers = {
            '.txt': TextParser,
            '.html': HtmlParser
        }

    def run(self, texts: Optional[List[str]] = None, gold_markups: Optional[List[List[Entity]]] = None) -> List[Document]:
        documents = []

        # Case 1: Process in-memory texts
        if texts:
            if gold_markups and len(texts) != len(gold_markups):
                raise ValueError("Количество текстов и количество gold_markups не совпадают.")

            if not gold_markups:
                gold_markups = [None] * len(texts)

            for idx, (raw_text, gold_markup) in enumerate(zip(texts, gold_markups)):
                doc = self._create_document(
                    name=f"in-memory-doc-{idx}",
                    text=raw_text,
                    gold_markup=gold_markup
                )
                documents.append(doc)

        # Case 2: Process files from directory
        elif self.path_to_files:
            for filename in os.listdir(self.path_to_files):
                full_path = os.path.join(self.path_to_files, filename)

                if not os.path.isfile(full_path):
                    continue

                _, ext = os.path.splitext(filename)
                ext = ext.lower()

                if ext not in self.parsers:
                    continue

                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                gold_markup = []
                if self.path_to_markups:
                    gold_markup = self._load_gold_markup(filename)

                doc = self.parsers[ext].parse_file(filename, content, gold_markup)
                documents.append(doc)
        return documents

    def _load_gold_markup(self, filename: str) -> List[Entity]:
        base_name, _ = os.path.splitext(filename)
        json_name = base_name + ".json"
        json_path = os.path.join(self.path_to_markups, json_name)

        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entities = []
                for item in data:
                    entity_obj = Entity(
                        entity=item["entity"],
                        start_offset=item["start_offset"],
                        end_offset=item["end_offset"],
                        text=item["text"]
                    )
                    entities.append(entity_obj)
                return entities
        else:
            return []

    @staticmethod
    def _create_document(
        text: str,
        plaintext: Optional[str] = None,
        name: str = "",
        gold_markup: Optional[List[Entity]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Document:
        if gold_markup is None:
            gold_markup = []

        if metadata is None:
            metadata = {}

        if plaintext is None:
            plaintext = text

        doc = Document(
            name=name,
            text=text,
            plaintext=plaintext,
            gold_markup=gold_markup,
            metadata=metadata,
        )
        return doc
