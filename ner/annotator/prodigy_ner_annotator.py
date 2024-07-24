from base.dataset.dataset import Dataset, ListDataset
from base.training.example import Instance
from ner.bases.entity import NamedEntity
import prodigy
from prodigy.core import Controller
from prodigy.components.db import connect
from prodigy.app import server
import os
from typing import Any, Callable


class ProdigyNerAnnotator:
    dataset_name: str
    config: dict
    original_dataset: Dataset[str, list[NamedEntity]]
    annotated_dataset: Dataset[str, list[NamedEntity]]
    on_saved: Callable[[list[Instance]], None] | None

    def __init__(
        self,
        dataset: Dataset[str, list[NamedEntity]],
        dataset_name: str,
        config: dict = {},
        on_saved: Callable[[list[Instance]], None] | None = None,
        buffer: Dataset[str, list[NamedEntity]] = ListDataset([]),
    ) -> None:
        self.original_dataset = dataset
        self.dataset_name = dataset_name
        self.config = config
        self.on_saved = on_saved
        self.annotated_dataset = buffer

    def start(self, host: str = "localhost", port: int = 8888):
        db = connect()
        old_data = db.get_dataset_examples(self.dataset_name)
        if old_data is not None:
            self._update_annotated_set(old_data)

        component = self._recipe()
        controller = Controller.from_components(self.dataset_name, component)
        server(controller=controller, config=self.config | {"host": host, "port": port})

    def reset(self):
        db = connect()
        if db.get_dataset_examples(self.dataset_name) is not None:
            db.drop_dataset(self.dataset_name)
        self.annotated_dataset.clear()

    def _recipe(self):
        def get_stream():
            for x in self.original_dataset:
                data = {
                    "text": x.input,
                    "spans": [
                        {
                            "start": e.start,
                            "end": e.end,
                            "token_start": e.start,
                            "token_end": e.end - 1,
                            "label": e.type,
                        }
                        for e in x.label
                    ],
                    "tokens": [
                        {"text": t, "start": i, "end": i + 1, "id": i}
                        for i, t in enumerate(x.input)
                    ],
                }

                yield data

        def update(inputs):
            new_instances = self._update_annotated_set(inputs)
            if self.on_saved is not None:
                self.on_saved(new_instances)

        stream = get_stream()

        return {
            "dataset": self.dataset_name,
            "update": update,
            "view_id": "ner_manual",
            "stream": stream,
        }

    def _update_annotated_set(self, inputs: Any) -> list[Instance[str, NamedEntity]]:
        new_instances = []
        for raw_instance in inputs:
            if raw_instance['answer'] != 'accept':
                continue
            instance = Instance(
                raw_instance["text"],
                [
                    NamedEntity(
                        type=e["label"],
                        start=e["start"],
                        end=e["end"],
                        text=raw_instance["text"][e["start"] : e["end"]],
                    )
                    for e in raw_instance["spans"]
                ],
            )
            new_instances.append(instance)
        self.annotated_dataset.extend(instances=new_instances)
        return new_instances