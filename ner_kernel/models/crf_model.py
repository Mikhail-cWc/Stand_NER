import logging
from typing import List, Tuple
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from ..instance import Document, Entity
from .base_model import BaseNERModel
from ..logger import logger


class CRFNERModel(BaseNERModel):
    # CRF model
    def __init__(self, model_name: str = "crf_ner", **kwargs):
        self.model_name = model_name
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            **kwargs
        )
        self.is_trained = False

    def _word2features(self, sent: List[str], i: int) -> dict:
        word = sent[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = sent[i-1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True

        return features

    def _sent2features(self, sent: List[str]) -> List[dict]:
        return [self._word2features(sent, i) for i in range(len(sent))]

    def train(self, X_train: List[List[str]], y_train: List[List[str]]):
        """
        Обучение CRF модели
        :param X_train: список предложений (каждое предложение - список слов)
        :param y_train: список меток для каждого предложения
        """
        X = [self._sent2features(s) for s in X_train]
        self.crf.fit(X, y_train)
        self.is_trained = True
        logger.info("CRF модель успешно обучена")

    def predict_entities(self, text: str) -> List[Entity]:
        if not self.is_trained:
            logger.warning(
                "CRF модель не обучена. Возвращаю пустой список сущностей.")
            return []

        words = text.split()
        X = self._sent2features(words)
        y_pred = self.crf.predict([X])[0]

        result_entities = []
        current_entity = None
        start_pos = 0

        for i, (word, label) in enumerate(zip(words, y_pred)):
            if label.startswith('B-'):
                if current_entity:
                    result_entities.append(current_entity)
                current_entity = Entity(
                    entity=label[2:],
                    start_offset=start_pos,
                    end_offset=start_pos + len(word),
                    text=word
                )
            elif label.startswith('I-') and current_entity and current_entity.entity == label[2:]:
                current_entity.text += ' ' + word
                current_entity.end_offset = start_pos + len(word)
            else:
                if current_entity:
                    result_entities.append(current_entity)
                    current_entity = None

            start_pos += len(word) + 1  # +1 for space

        if current_entity:
            result_entities.append(current_entity)

        return result_entities

    def change_model(self, model_name: str):
        if self.model_name != model_name:
            self.model_name = model_name
            self.crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            self.is_trained = False
