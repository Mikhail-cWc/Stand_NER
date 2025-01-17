from typing import Dict, List, Optional
from ..instance import Document
from ..models import BaseNERModel
from ..validator import NERValidator


class Pipeline:
    def __init__(self, model: BaseNERModel, validator: Optional[NERValidator] = None):
        self.model = model
        self.validator = validator

    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        for doc in documents:
            self.model.predict_document(doc)

    def validate(self, documents: List[Document]) -> Dict[str, float]:
        if self.validator is None:
            raise ValueError("Валидатор не определен.")
        return self.validator.evaluate(documents)
