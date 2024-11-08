import json
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, is_dataclass, asdict


class DatasetName(str, Enum):
    MULTIWOZ = "MultiWOZ"
    # TODO: Implement more datasets


@dataclass
class Arguments:
    model: str
    dataset: str
    no_summary_tree: bool
    no_dialogue_KG: bool
    exp_name: str


@dataclass
class Sample:
    document_ids: List[str]
    conversation: List[Dict[str, str]]


@dataclass
class Label:
    document_relevant: bool
    segments: Optional[List[str]]
    answer: Optional[str]

    time_taken: Optional[float] = None


class DataClassEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)
