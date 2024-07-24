from __future__ import annotations
from dataclasses import dataclass
import json
import re
import copy


@dataclass
class NamedEntity:
    type: str
    text: str
    start: int
    end: int
    children: list[NamedEntity]

    def __init__(
        self,
        type: str,
        text: str,
        start: int,
        end: int,
        children: list[NamedEntity] | None = None,
    ) -> None:
        self.type = type
        self.text = text
        self.start = start
        self.end = end
        self.children = children or []

    @staticmethod
    def serialize(dict: dict) -> NamedEntity:
        return NamedEntity(**dict)

    def trim_entities(self):
        e = copy.copy(self)

        startPattern = "^\s+"
        endPattern = "\s+$"

        matchStart = re.search(startPattern, e.text, re.DOTALL)
        matchEnd = re.search(endPattern, e.text, re.DOTALL)

        e.text = e.text.strip()
        if matchStart:
            e.start += len(matchStart.group(0))
        if matchEnd:
            e.end -= len(matchEnd.group(0))

        return e