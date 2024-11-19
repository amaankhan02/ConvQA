import json
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, is_dataclass, asdict


class DatasetName(str, Enum):
    MULTIWOZ = "MultiWOZ"
    QUAC = "QuAC"
    COQA = "CoQA"
    # TODO: @Amaan Implement more datasets


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
    
    # TODO: why don't we add a question field? or do we just assume the last question in the conversation is the question?
    
    # QuAC and CoQA specific fields
    id: Optional[str]   # TODO: what should be the datatype for the sample id?


@dataclass
class Label:
    document_relevant: bool
    segments: Optional[List[str]]    #
    answer: Optional[str]
    
    # QuAC and CoQA specific fields
    q_id: Optional[str]     # Question ID / Turn ID
    span_start: Optional[int]
    span_end: Optional[int]

    time_taken: Optional[float] = None


class DataClassEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)
