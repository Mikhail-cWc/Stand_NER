from typing import Dict, List, Optional
from ..instance import Document, Entity
from ..models import BaseNERModel
from ..validator import NERValidator
from ..standardizer import LabelStandardizer

class Pipeline:
    def __init__(
        self,
        model: BaseNERModel,
        standardizer: LabelStandardizer,
        validator: Optional[NERValidator] = None,
        dataset_name: Optional[str] = None
    ):
        self.model = model
        self.validator = validator
        self.standardizer = standardizer
        self.dataset_name = dataset_name

    def run(self, documents: List[Document]) -> List[Document]:
        if self.dataset_name:
            documents = self._standardize_input(documents)
        
        processed_docs = [self.model.predict_document(doc) for doc in documents]
        if self.standardizer.model_mappings[self.model.model_name]:
            processed_docs = self._standardize_output(processed_docs)
        return processed_docs

    def _standardize_input(self, documents: List[Document]) -> List[Document]:
        return [
            Document(
                name=doc.name,
                text=doc.text,
                plaintext=doc.plaintext,
                gold_markup=[
                    Entity(
                        entity=self.standardizer.map_dataset_label(
                            dataset_name=self.dataset_name,
                            original_label=ent.entity
                        ),
                        start_offset=ent.start_offset,
                        end_offset=ent.end_offset,
                        text=ent.text
                    ) for ent in doc.gold_markup
                ],
                pred_markup=doc.pred_markup,
                metadata=doc.metadata
            ) for doc in documents
        ]

    def _standardize_output(self, documents: List[Document]) -> List[Document]:
        return [
            Document(
                name=doc.name,
                text=doc.text,
                plaintext=doc.plaintext,
                gold_markup=doc.gold_markup,
                pred_markup=[
                    Entity(
                        entity=self.standardizer.map_model_label(
                            model_name=self.model.model_name,
                            original_label=ent.entity
                        ),
                        start_offset=ent.start_offset,
                        end_offset=ent.end_offset,
                        text=ent.text
                    ) for ent in doc.pred_markup
                ],
                metadata=doc.metadata
            ) for doc in documents
        ]

    def validate(self, documents: List[Document]) -> Dict[str, float]:
        if not self.validator:
            raise ValueError("Валидатор не определен")   
        return self.validator.evaluate(documents)

    def update_mappings(
        self,
        model_mapping: Optional[Dict] = None,
        dataset_mapping: Optional[Dict] = None
    ):
        if model_mapping:
            self.standardizer.add_model_mapping(
                model_name=self.model.model_name,
                mapping=model_mapping
            )
        if dataset_mapping and self.dataset_name:
            self.standardizer.add_dataset_mapping(
                dataset_name=self.dataset_name,
                mapping=dataset_mapping
            )