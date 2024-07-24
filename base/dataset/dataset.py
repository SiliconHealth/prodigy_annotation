from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, TypeVar
import random

from base.training.example import Instance

TInput = TypeVar("TInput")
TLabel = TypeVar("TLabel")


class Dataset(ABC, Generic[TInput, TLabel]):
    @abstractmethod
    def __getitem__(
        self, item
    ) -> list[Instance[TInput, TLabel]] | Instance[TInput, TLabel]:
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[Instance[TInput, TLabel]]:
        pass

    @abstractmethod
    def __next__(self) -> Instance[TInput, TLabel]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def shuffle(self):
        pass

    @abstractmethod
    def append(self, instance: Instance[TInput, TLabel]):
        pass

    @abstractmethod
    def extend(self, instances: list[Instance[TInput, TLabel]]):
        pass

    @abstractmethod
    def clear(self):
        pass


class ListDataset(Dataset[TInput, TLabel]):
    instances: list[Instance[TInput, TLabel]]

    def __init__(self, instances: list[Instance[TInput, TLabel]]):
        self.instances = instances

    def __getitem__(
        self, item
    ) -> list[Instance[TInput, TLabel]] | Instance[TInput, TLabel]:
        return self.instances[item]

    def __iter__(self):
        return self.instances.__iter__()

    def __next__(self):
        return self.instances.__next__()

    def __len__(self):
        return len(self.instances)

    def shuffle(self):
        random.shuffle(self.instances)

    def append(self, instance: Instance[TInput, TLabel]):
        self.instances.append(instance)

    def extend(self, instances: list[Instance[TInput, TLabel]]):
        self.instances.extend(instances)

    def clear(self):
        self.instances.clear()