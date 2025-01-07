from sklearn.metrics import precision_recall_fscore_support


class NERValidator:
    def evaluate(self, docs: List[Document]) -> Dict[str, float]:
        y_true = []
        y_pred = []

        for doc in docs:
            gold_set = {(ent.entity, ent.start_offset, ent.end_offset) for ent in doc.gold_markup}
            pred_set = {(ent.entity, ent.start_offset, ent.end_offset) for ent in doc.pred_markup}

            all_entities = gold_set.union(pred_set)
            for e in all_entities:
                y_true.append(1 if e in gold_set else 0)
                y_pred.append(1 if e in pred_set else 0)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return {"precision": precision, "recall": recall, "f1": f1}
