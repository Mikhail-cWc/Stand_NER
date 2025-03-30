from abc import ABC, abstractmethod
from typing import List

from ..instance import Document, Entity


class BaseNERModel(ABC):
    # Base class for all NER models
    def __init__(self):
        self.model_name = "Base"
        self.entity_label = "MISC"

    def predict_entities(self, text: str) -> List[Entity]:
        result_entities = []
        for entity_span in [word for word in text.split() if word.istitle()]:
            start_position = text.find(entity_span)
            e = Entity(
                entity=self.entity_label,
                start_offset=start_position,
                end_offset=start_position + len(entity_span),
                text=entity_span
            )
            result_entities.append(e)
        return result_entities

    def predict_document(self, doc: Document) -> Document:
        predicted = self.predict_entities(doc.plaintext)
        doc.pred_markup = predicted
        return doc

    def change_model(self, model_name: str):
        pass
