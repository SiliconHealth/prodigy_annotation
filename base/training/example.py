from abc import ABC
from dataclasses import dataclass
import typing

TInput = typing.TypeVar("TInput")
TLabel = typing.TypeVar("TLabel")

@dataclass
class Instance(typing.Generic[TInput, TLabel]):
    input: TInput
    label: TLabel

    def __init__(self, input: TInput, label: TLabel) -> None:
        self.input = input
        self.label = label