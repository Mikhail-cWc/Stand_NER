from typing import List, Optional, Dict
import os
import json
from bs4 import BeautifulSoup
from ..instance import Entity, Document


class DataLoader:
    def __init__(
        self,
        path_to_files: Optional[str] = None,
        path_to_markups: Optional[str] = None
    ):
        self.path_to_files = path_to_files
        self.path_to_markups = path_to_markups

    def run(self, texts: Optional[List[str]] = None, gold_markups: Optional[List[List[Entity]]] = None) -> List[Document]:
        documents = []

        if texts:
            if texts and gold_markups and len(texts) != len(gold_markups):
                raise ValueError("Количество текстов и количество gold_markups не совпадают.")
            elif texts and not gold_markups:
                gold_markups = [None]*len(texts)
            for idx, (raw_text, gold_markup) in enumerate(zip(texts, gold_markups)):
                doc = self._create_document(
                    name=f"in-memory-doc-{idx}",
                    text=raw_text,
                    gold_markup=gold_markup
                )
                documents.append(doc)

        elif self.path_to_files:
            for filename in os.listdir(self.path_to_files):
                full_path = os.path.join(self.path_to_files, filename)

                if os.path.isfile(full_path):
                    if filename.lower().endswith('.html'):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            raw_html = f.read()

                        gold_markup = []
                        soup = BeautifulSoup(raw_html, "html.parser")
                        clean_text = soup.get_text()

                        if self.path_to_markups:
                            gold = self._load_gold_markup(filename)
                            gold_markup = gold

                        doc = self._create_document(
                            name=filename,
                            text=raw_html,
                            plaintext=clean_text,
                            gold_markup=gold_markup,
                            metadata={"source_type": "html"}
                        )
                        documents.append(doc)

                    elif filename.lower().endswith('.txt'):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                            gold_markup = []
                            if self.path_to_markups:
                                gold = self._load_gold_markup(filename)
                                gold_markup = gold
                            doc = self._create_document(
                                name=filename,
                                text=raw_text,
                                gold_markup=gold_markup,
                                metadata={"source_type": "txt"}
                            )

                        documents.append(doc)
                    else:
                        continue

        return documents

    def _load_gold_markup(self, filename: str) -> List[Entity]:
        base_name, _ = os.path.splitext(filename)
        json_name = base_name + ".json"
        json_path = os.path.join(self.path_to_markups, json_name)

        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data ожидается списком словарей вида:
                # [
                #   {"entity": "PERSON", "start_offset": 0, "end_offset": 9, "text": "Илон Маск"},
                #   ...
                # ]
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
