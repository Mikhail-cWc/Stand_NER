from typing import Dict
from collections import defaultdict


class LabelStandardizer:
    def __init__(self):
        self.model_mappings = defaultdict(dict)  # {"model_name": {"ORG": "ORG", ...}}
        self.dataset_mappings = defaultdict(dict)

        self.default_label = "MISC"

    def map_model_label(self, model_name: str, original_label: str) -> str:
        return self.model_mappings[model_name].get(
            original_label,
            self.default_label
        )

    def map_dataset_label(self, dataset_name: str, original_label: str) -> str:
        return self.dataset_mappings[dataset_name].get(
            original_label,
            self.default_label
        )

    def add_model_mapping(self, model_name: str, mapping: Dict[str, str]):
        self.model_mappings[model_name].update(mapping)

    def add_dataset_mapping(self, dataset_name: str, mapping: Dict[str, str]):
        self.dataset_mappings[dataset_name].update(mapping)
