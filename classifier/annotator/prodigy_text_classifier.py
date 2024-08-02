from base.dataset.dataset import Dataset, ListDataset
from base.training.example import Instance
import prodigy
from prodigy.core import Controller
from prodigy.components.db import connect
from prodigy.app import server
import os
from typing import Any, Callable


class ProdigyTextClassificationAnnotator:
    dataset_name: str
    config: dict
    original_dataset: Dataset[str, list]
    annotated_dataset: Dataset[str, list]
    on_saved: Callable[[list[Instance]], None] | None
    
    def __init__(
        self, 
        dataset: Dataset[str, list],
        dataset_name: str,
        config: dict = {},
        on_saved: Callable[[list[Instance]], None] | None = None,
        buffer: Dataset[str, list] = ListDataset([]),
    ) -> None:
        self.dataset_name = dataset_name
        self.config = config
        self.original_dataset = dataset
        self.annotated_dataset = buffer
        self.on_saved = on_saved

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
                    "options": self.config['options'],
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
            "view_id": "choice",
            "stream": stream,
        }
    
    def _update_annotated_set(self, inputs: Any) -> list[Instance[str, list]]:
        new_instances = []
        for raw_instance in inputs:
            if raw_instance['answer'] != 'accept':
                continue
            instance = Instance(
                raw_instance["text"],
                raw_instance["accept"],
            )
            new_instances.append(instance)
        self.annotated_dataset.extend(instances=new_instances)
        return new_instances