from flair.data import Sentence
from flair.models import SequenceTagger
from typing import List
from ..instance import Document, Entity
from .base_model import BaseNERModel


class FlairNERModel(BaseNERModel):
    # BiLSTM + CRF
    def __init__(self, model_name: str = "ner-fast"):
        # Можно указать "ner", "ner-fast", "ner-ontonotes-fast" и т.п.
        self.model_name = "ner-fast"
        self.tagger = SequenceTagger.load(model_name, weights_only=False)

    def predict_entities(self, text: str) -> List[Entity]:
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        result_entities = []
        for entity_span in sentence.get_spans('ner'):
            entity_label = entity_span.get_label("ner").value
            e = Entity(
                entity=entity_label,
                start_offset=entity_span.start_position,
                end_offset=entity_span.end_position,
                text=text[entity_span.start_position:entity_span.end_position]
            )
            result_entities.append(e)
        return result_entities

    def change_model(self, model_name: str):
        if self.model_name != model_name:
            self.tagger = SequenceTagger.load(model_name)
