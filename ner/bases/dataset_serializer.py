from typing import Callable
from base.dataset.dataset import Dataset
from base.dataset.dataset_mapping import DatasetMapping
from base.training.example import Instance
from ner.bases.entity import NamedEntity


class NerDatasetSerializer(DatasetMapping[str, list[dict], str, list[NamedEntity]]):
    def __init__(self, inner_dataset: Dataset) -> None:
        def serialize(raw: dict) -> NamedEntity:
            return NamedEntity(
                start=raw["start"],
                end=raw["end"],
                text=raw["text"],
                type=raw["type"],
                children=[serialize(c) for c in raw["children"]],
            )

        def deserialize(ent: NamedEntity) -> dict:
            return {
                "start": ent.start,
                "end": ent.end,
                "text": ent.text,
                "type": ent.type,
                "children": [deserialize(c) for c in ent.children],
            }

        super().__init__(
            inner_dataset=inner_dataset,
            mapper_out=lambda x: Instance(
                input=x.input, label=[serialize(e) for e in x.label]
            ),
            mapper_in=lambda x: Instance(
                input=x.input, label=[deserialize(e) for e in x.label]
            ),
        )