from typing import List, Dict
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
        if abs(gold_ent.end_offset - pred_ent.end_offset) > self.tolerance:
            return False
        return True

    def evaluate(self, docs: List[Document]) -> Dict[str, any]:
        doc_metrics = []
        global_y_true = []
        global_y_pred = []

        per_label_map = defaultdict(lambda: {"y_true": [], "y_pred": []})

        for doc in docs:
            gold_entities = doc.gold_markup
            pred_entities = doc.pred_markup

            y_true_doc = []
            y_pred_doc = []

            doc_label_map = defaultdict(lambda: {"y_true": [], "y_pred": []})

            all_pairs = []
            matched_pred_indices = set()

            for g in gold_entities:
                matched_index = None
                for i, p in enumerate(pred_entities):
                    if i not in matched_pred_indices and self._compare_entities(g, p):
                        matched_index = i
                        break
                if matched_index is not None:
                    matched_pred_indices.add(matched_index)
                    all_pairs.append((g, pred_entities[matched_index]))
                else:
                    all_pairs.append((g, None))

            for i, p in enumerate(pred_entities):
                if i not in matched_pred_indices:
                    all_pairs.append((None, p))

            for g, p in all_pairs:
                y_true_val = 1 if g is not None else 0
                y_pred_val = 1 if p is not None else 0

                y_true_doc.append(y_true_val)
                y_pred_doc.append(y_pred_val)
                global_y_true.append(y_true_val)
                global_y_pred.append(y_pred_val)

                label = None
                if g is not None:
                    label = g.entity
                elif p is not None:
                    label = p.entity

                if label:
                    # Запись в doc_label_map
                    doc_label_map[label]["y_true"].append(y_true_val)
                    doc_label_map[label]["y_pred"].append(y_pred_val)
                    # Запись в per_label_map (весь корпус)
                    per_label_map[label]["y_true"].append(y_true_val)
                    per_label_map[label]["y_pred"].append(y_pred_val)
            precision_doc, recall_doc, f1_doc, _ = precision_recall_fscore_support(
                y_true_doc, y_pred_doc, average='binary', zero_division=0
            )

            doc_label_metrics = {}
            for label, data in doc_label_map.items():
                lab_precision, lab_recall, lab_f1, _ = precision_recall_fscore_support(
                    data["y_true"], data["y_pred"], average='binary', zero_division=0
                )
                doc_label_metrics[label] = {
                    "precision": lab_precision,
                    "recall": lab_recall,
                    "f1": lab_f1,
                    "support": len(data["y_true"])
                }

            doc_metrics.append({
                "document_id": doc.name,
                "precision": precision_doc,
                "recall": recall_doc,
                "f1": f1_doc,
                "support": len(y_true_doc),
                "label_metrics": doc_label_metrics
            })

        precision_global, recall_global, f1_global, _ = precision_recall_fscore_support(
            global_y_true, global_y_pred, average='binary', zero_division=0
        )

        macro_precision = np.mean([m["precision"] for m in doc_metrics])
        macro_recall = np.mean([m["recall"] for m in doc_metrics])
        macro_f1 = np.mean([m["f1"] for m in doc_metrics])

        label_metrics = {}
        for label, data in per_label_map.items():
            lab_precision, lab_recall, lab_f1, _ = precision_recall_fscore_support(
                data["y_true"], data["y_pred"], average='binary', zero_division=0
            )
            label_metrics[label] = {
                "precision": lab_precision,
                "recall": lab_recall,
                "f1": lab_f1,
                "support": len(data["y_true"])
            }

        return {
            "documents": doc_metrics,
            "micro_avg": {
                "precision": precision_global,
                "recall": recall_global,
                "f1": f1_global,
                "support": len(global_y_true)
            },
            "macro_avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1
            },
            "label_metrics": label_metrics
        }
