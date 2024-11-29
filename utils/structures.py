import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class DatasetName(str, Enum):
    MULTIWOZ = "MultiWOZ"
    QUAC = "QuAC"
    COQA = "CoQA"


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

    # For multiple answers we use a delimiter of ANSWER_DELIM to join them.
    # the first answer (if separated b/w the delim) is considered the "best" answer
    answer: Optional[str]

    time_taken: Optional[float] = None  # How long it takes to answer the question


class DataClassEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)
