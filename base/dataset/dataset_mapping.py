from typing import Any, Callable, Generic, TypeVar
from base.dataset.dataset import Dataset
from base.training.example import Instance

TInnerInput = TypeVar("TInnerInput")
TInnerLabel = TypeVar("TInnerLabel")
TOuterInput = TypeVar("TOuterInput")
TOuterLabel = TypeVar("TOuterLabel")


class DatasetMapping(
    Dataset[TOuterInput, TOuterLabel],
    Generic[TInnerInput, TInnerLabel, TOuterInput, TOuterLabel],
):
    inner_dataset: Dataset[TInnerInput, TInnerLabel]
    mapper_out: Callable[
        [Instance[TInnerInput, TInnerLabel]], Instance[TOuterInput, TOuterLabel]
    ]
    mapper_in: Callable[
        [Instance[TOuterInput, TOuterLabel]], Instance[TInnerInput, TInnerLabel]
    ]

    _current_index: int = 0

    def __init__(
        self,
        inner_dataset: Dataset,
        mapper_out: Callable[
            [Instance[TInnerInput, TInnerLabel]], Instance[TOuterInput, TOuterLabel]
        ],
        mapper_in: Callable[
            [Instance[TOuterInput, TOuterLabel]], Instance[TInnerInput, TInnerLabel]
        ],
    ) -> None:
        super().__init__()
        self.inner_dataset = inner_dataset
        self.mapper_out = mapper_out
        self.mapper_in = mapper_in

    def __getitem__(
        self, item
    ) -> list[Instance[TOuterInput, TOuterLabel]] | Instance[TOuterInput, TOuterLabel]:
        result = self.inner_dataset[item]
        if isinstance(result, list):
            return list(map(self.mapper_out, result))
        else:
            return self.mapper_out(result)

    def __iter__(self):
        self._current_index = 0
        return self

    def __len__(self) -> int:
        return self.inner_dataset.__len__()

    def __next__(self) -> Instance[TOuterInput, TOuterLabel]:
        if self._current_index < len(self):
            x = self[self._current_index]
            self._current_index += 1
            return x
        raise StopIteration

    def shuffle(self):
        self.inner_dataset.shuffle()

    def append(self, instance: Instance[TOuterInput, TOuterLabel]):
        self.inner_dataset.append(self.mapper_in(instance))

    def extend(self, instances: list[Instance[TOuterInput, TOuterLabel]]):
        self.inner_dataset.extend([self.mapper_in(ins) for ins in instances])

    def clear(self):
        self.inner_dataset.clear()