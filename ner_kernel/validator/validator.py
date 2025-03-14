from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from ..instance import Document, Entity


class NERValidator:
    def __init__(self, tolerance: int = 5):
        self.tolerance = tolerance

    def _compare_entities(self, gold_ent: Entity, pred_ent: Entity) -> bool:
        if gold_ent.entity != pred_ent.entity:
            return False
        if abs(gold_ent.start_offset - pred_ent.start_offset) > self.tolerance:
            return False
        return True

    def _match_entities(self, gold_entities: List[Entity], pred_entities: List[Entity]) -> List[Tuple[Optional[Entity], Optional[Entity]]]:
        all_pairs = []
        matched_pred_indices = set()

        for gold in gold_entities:
            matched_index = None
            for i, pred in enumerate(pred_entities):
                if i not in matched_pred_indices and self._compare_entities(gold, pred):
                    matched_index = i
                    break

            if matched_index is not None:
                matched_pred_indices.add(matched_index)
                all_pairs.append((gold, pred_entities[matched_index]))
            else:
                all_pairs.append((gold, None))  # Ложноотрицательный результат

        for i, pred in enumerate(pred_entities):
            if i not in matched_pred_indices:
                all_pairs.append((None, pred))  # Ложноположительный результат

        return all_pairs

    def _calculate_binary_labels(self, entity_pairs: List[Tuple[Optional[Entity], Optional[Entity]]]) -> Tuple[List[int], List[int], Dict[str, Dict[str, List[int]]]]:
        y_true = []
        y_pred = []
        label_map = defaultdict(lambda: {"y_true": [], "y_pred": []})

        for gold, pred in entity_pairs:
            y_true_val = 1 if gold is not None else 0
            y_pred_val = 1 if pred is not None else 0

            y_true.append(y_true_val)
            y_pred.append(y_pred_val)

            label = None
            if gold is not None:
                label = gold.entity
            elif pred is not None:
                label = pred.entity
            if label:
                label_map[label]["y_true"].append(y_true_val)
                label_map[label]["y_pred"].append(y_pred_val)

        return y_true, y_pred, label_map

    def _calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(y_true)
        }

    def _calculate_label_metrics(self, label_map: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, float]]:
        label_metrics = {}
        for label, data in label_map.items():
            label_metrics[label] = self._calculate_metrics(data["y_true"], data["y_pred"])

        return label_metrics

    def evaluate(self, docs: List[Document]) -> Dict[str, any]:
        doc_metrics = []
        global_y_true = []
        global_y_pred = []
        global_label_map = defaultdict(lambda: {"y_true": [], "y_pred": []})

        for doc in docs:
            entity_pairs = self._match_entities(doc.gold_markup, doc.pred_markup)

            y_true_doc, y_pred_doc, doc_label_map = self._calculate_binary_labels(entity_pairs)

            global_y_true.extend(y_true_doc)
            global_y_pred.extend(y_pred_doc)

            for label, data in doc_label_map.items():
                global_label_map[label]["y_true"].extend(data["y_true"])
                global_label_map[label]["y_pred"].extend(data["y_pred"])

            doc_metrics_dict = self._calculate_metrics(y_true_doc, y_pred_doc)

            doc_label_metrics = self._calculate_label_metrics(doc_label_map)

            doc_metrics.append({
                "document_id": doc.name,
                **doc_metrics_dict,
                "label_metrics": doc_label_metrics
            })

        micro_avg = self._calculate_metrics(global_y_true, global_y_pred)
        macro_avg = {
            "precision": np.mean([m["precision"] for m in doc_metrics]),
            "recall": np.mean([m["recall"] for m in doc_metrics]),
            "f1": np.mean([m["f1"] for m in doc_metrics])
        }

        label_metrics = self._calculate_label_metrics(global_label_map)
        return {
            "documents": doc_metrics,
            "micro_avg": micro_avg,
            "macro_avg": macro_avg,
            "label_metrics": label_metrics
        }
