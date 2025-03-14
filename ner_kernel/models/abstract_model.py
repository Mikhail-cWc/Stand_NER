from abc import ABC, abstractmethod
from typing import List

from ..instance import Document, Entity


class BaseNERModel(ABC):
    @abstractmethod
    def predict_entities(self, text: str) -> List[Entity]:
        pass

    def predict_document(self, doc: Document) -> Document:
        predicted = self.predict_entities(doc.plaintext)
        doc.pred_markup = predicted
        return doc

    @abstractmethod
    def change_model(self, model_name: str):
        pass
