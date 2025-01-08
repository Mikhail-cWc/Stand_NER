import spacy
from typing import List
from ..instance import Document, Entity
from .abstract_model import BaseNERModel


class SpacyNERModel(BaseNERModel):
    def __init__(self, model_name: str = "ru_core_news_sm"):
        self.nlp = spacy.load(model_name)

    def predict_entities(self, text: str) -> List[Entity]:
        doc_spacy = self.nlp(text)
        result_entities = []
        for ent in doc_spacy.ents:
            # В spaCy у нас есть ent.label_, ent.start_char, ent.end_char, ent.text
            e = Entity(
                entity=ent.label_,
                start_offset=ent.start_char,
                end_offset=ent.end_char,
                text=ent.text
            )
            result_entities.append(e)
        return result_entities
