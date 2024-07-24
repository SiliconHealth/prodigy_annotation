from typing import Any, Callable, Generic, TypeVar
import datasets

from base.dataset.dataset import Dataset
from base.training.example import Instance

TInput = TypeVar("TInput")
TLabel = TypeVar("TLabel")


class HuggingfaceDataset(Dataset[TInput, TLabel], Generic[TInput, TLabel]):
    hf_dataset: datasets.Dataset
    current_index: int = 0
    input_key: str
    label_key: str | None

    def __init__(
        self,
        dataset: datasets.Dataset = datasets.Dataset.from_list([]),
        input_key: str = "text",
        label_key: str | None = "label",
    ) -> None:
        super().__init__()
        self.hf_dataset = dataset
        self.input_key = input_key
        self.label_key = label_key

    def __getitem__(
        self, items
    ) -> list[Instance[TInput, TLabel]] | Instance[TInput, TLabel]:
        if isinstance(items, slice):
            return [
                Instance(
                    x[self.input_key],
                    x[self.label_key] if self.label_key is not None else None,
                )
                for x in self.hf_dataset.select(
                    range(
                        items.start or 0,
                        min(len(self.hf_dataset), items.stop or len(self.hf_dataset)),
                        items.step or 1,
                    )
                )
            ]
        elif isinstance(items, int):
            x = self.hf_dataset[items]
            return Instance(
                x[self.input_key],
                x[self.label_key] if self.label_key is not None else None,
            )

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> Instance[TInput, TLabel]:
        if self.current_index < len(self):
            x = self[self.current_index]
            self.current_index += 1
            return x
        raise StopIteration

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def shuffle(self):
        self.hf_dataset = self.hf_dataset.shuffle()

    def append(self, instance: Instance[TInput, TLabel]):
        self.hf_dataset = self.hf_dataset.add_item(
            {self.input_key: instance.input}
            | ({self.label_key: instance.label} if self.label_key is not None else {})
        )

    def extend(self, instances: list[Instance[TInput, TLabel]]):
        for ins in instances:
            self.append(ins)

    def clear(self):
        self.hf_dataset = datasets.Dataset.from_list([])