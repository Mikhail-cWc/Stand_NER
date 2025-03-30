import spacy
import subprocess
import sys
import logging

from typing import List
from ..instance import Document, Entity
from .base_model import BaseNERModel
from ..logger import logger


class SpacyNERModel(BaseNERModel):
    # CNN or LSTM
    def __init__(self, model_name: str = "ru_core_news_sm", **kwargs):
        self.model_name = model_name
        self._ensure_model_installed(model_name)
        self.nlp = spacy.load(model_name, **kwargs)

    def _ensure_model_installed(self, model_name: str):
        try:
            spacy.util.get_package_path(model_name)
        except (OSError, ModuleNotFoundError):
            logger.info(f"Модель {model_name} не установлена. Устанавливаю...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True
            )
            logger.info(f"Модель {model_name} успешно установлена.")

    def predict_entities(self, text: str) -> List[Entity]:
        doc_spacy = self.nlp(text)
        result_entities = []
        for ent in doc_spacy.ents:
            e = Entity(
                entity=ent.label_,
                start_offset=ent.start_char,
                end_offset=ent.end_char,
                text=ent.text
            )
            result_entities.append(e)
        return result_entities

    def change_model(self, model_name: str):
        if self.model_name != model_name:
            self.model_name = model_name
            self._ensure_model_installed(model_name)
            self.nlp = spacy.load(model_name)
