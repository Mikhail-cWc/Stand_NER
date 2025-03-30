from typing import List, Dict, Set
import numpy as np
from hmmlearn.hmm import CategoricalHMM
from ..instance import Document, Entity
from .base_model import BaseNERModel
from ..logger import logger


class HMMNERModel(BaseNERModel):
    # HMM model
    def __init__(self, model_name: str = "hmm_ner"):
        self.model_name = model_name
        self.model = None
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.tag_to_idx: Dict[str, int] = {}
        self.idx_to_tag: Dict[int, str] = {}
        self.is_trained = False

    def _prepare_features(self, text: str) -> List[str]:
        """Подготовка признаков для слова"""
        words = text.split()
        features = []
        for i, word in enumerate(words):
            word_features = [
                word.lower(),
                word[-3:] if len(word) >= 3 else word,
                word[-2:] if len(word) >= 2 else word,
                '1' if word[0].isupper() else '0',
                '1' if word.isdigit() else '0',
                '1' if word.isalpha() else '0'
            ]
            features.append(word_features)
        return features

    def _create_vocabulary(self, X_train: List[List[str]], y_train: List[List[str]]):
        """Создание словарей для слов и меток"""
        # Создаем словарь слов
        words = set()
        for sentence in X_train:
            words.update(sentence)
        self.word_to_idx = {word: idx for idx, word in enumerate(words)}
        self.idx_to_word = {idx: word for word,
                            idx in self.word_to_idx.items()}

        # Создаем словарь меток
        tags = set()
        for sentence in y_train:
            tags.update(sentence)
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}

    def _prepare_sequences(self, X_train: List[List[str]], y_train: List[List[str]]) -> tuple:
        """Подготовка последовательностей для обучения"""
        # Объединяем все последовательности в одну
        X = []
        y = []
        lengths = []  # длины каждой последовательности

        for sentence, labels in zip(X_train, y_train):
            # Преобразуем слова в индексы
            sentence_indices = [self.word_to_idx[word] for word in sentence]
            # Преобразуем метки в индексы
            label_indices = [self.tag_to_idx[label] for label in labels]

            # Добавляем последовательности
            X.extend(sentence_indices)
            y.extend(label_indices)
            lengths.append(len(sentence_indices))

        # Преобразуем в numpy массивы
        # reshape для соответствия формату hmmlearn
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        lengths = np.array(lengths)

        return X, y, lengths

    def train(self, X_train: List[List[str]], y_train: List[List[str]]):
        """
        Обучение HMM модели
        :param X_train: список предложений (каждое предложение - список слов)
        :param y_train: список меток для каждого предложения
        """
        self._create_vocabulary(X_train, y_train)
        X, y, lengths = self._prepare_sequences(X_train, y_train)

        # Создаем и обучаем HMM модель
        n_states = len(self.tag_to_idx)
        self.model = CategoricalHMM(n_components=n_states, random_state=42)

        # Обучаем модель
        self.model.fit(X, lengths=lengths)
        self.is_trained = True
        logger.info("HMM модель успешно обучена")

    def predict_entities(self, text: str) -> List[Entity]:
        if not self.is_trained:
            logger.warning(
                "HMM модель не обучена. Возвращаю пустой список сущностей.")
            return []

        # Подготовка входных данных
        words = text.split()
        X = [self.word_to_idx.get(word, len(self.word_to_idx) - 1)
             for word in words]
        X = np.array(X).reshape(-1, 1)

        # Предсказание меток
        y_pred = self.model.predict(X)

        # Преобразование предсказаний в сущности
        result_entities = []
        current_entity = None
        start_pos = 0

        for i, (word, label_idx) in enumerate(zip(words, y_pred)):
            label = self.idx_to_tag[label_idx]

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
            self.model = None
            self.word_to_idx = {}
            self.idx_to_word = {}
            self.tag_to_idx = {}
            self.idx_to_tag = {}
            self.is_trained = False
