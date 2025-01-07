from typing import Dict, List
from ..instance import Document
from ..models import BaseNERModel


class Pipeline:
    def __init__(self, models: Dict[str, BaseNERModel]):
        self.models = models

    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        results = {}

        for model_name, model in self.models.items():
            docs_copy = []
            for doc in documents:
                new_doc = Document(
                    name=doc.name,
                    text=doc.text,
                    plaintext=doc.plaintext,
                    gold_markup=doc.gold_markup[:],
                    metadata=dict(doc.metadata)
                )
                model.predict_document(new_doc)
                docs_copy.append(new_doc)
            results[model_name] = docs_copy

        return results
