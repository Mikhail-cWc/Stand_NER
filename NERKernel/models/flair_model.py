from flair.data import Sentence
from flair.models import SequenceTagger
from typing import List
from ..instance import Document, Entity
from .abstract_model import BaseNERModel


class FlairNERModel(BaseNERModel):
    def __init__(self, model_path: str = "ner-fast"):
        # Можно указать "ner", "ner-fast", "ner-ontonotes-fast" и т.п.
        self.tagger = SequenceTagger.load(model_path)

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
