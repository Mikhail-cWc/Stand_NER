from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List
from ..instance import Document, Entity
from .base_model import BaseNERModel


class HFNERModel(BaseNERModel):
    # Transformer model from Hugging Face
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name)
        self.ner_pipeline = pipeline(
            "ner", model=self.model,
            tokenizer=self.tokenizer, aggregation_strategy="simple"
        )

    def predict_entities(self, text: str) -> List[Entity]:
        ner_result = self.ner_pipeline(text)
        # ner_result — список словарей вида
        # [{'entity_group': 'PER', 'score': 0.999, 'word': 'Илон', 'start': 0, 'end': 4}, ...]
        result_entities = []
        for item in ner_result:
            e = Entity(
                entity=item["entity_group"],
                start_offset=item["start"],
                end_offset=item["end"],
                text=text[item["start"]:item["end"]]
            )
            result_entities.append(e)
        return result_entities

    def change_model(self, model_name: str):
        if self.model_name != model_name:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name)
            self.ner_pipeline = pipeline(
                "ner", model=self.model,
                tokenizer=self.tokenizer, aggregation_strategy="simple"
            )
