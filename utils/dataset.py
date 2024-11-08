import os
import json
from typing import Any

from .structures import *
from .data import multiwoz_utils
from .scorer import Scorer


class Dataset:
    def __init__(self, fp: str) -> None:
        should_preprocess = not (
            os.path.exists(os.path.join(fp, "docs.json"))
            and os.path.exists(os.path.join(fp, "train_X.json"))
            and os.path.exists(os.path.join(fp, "train_Y.json"))
            and os.path.exists(os.path.join(fp, "test_X.json"))
            and os.path.exists(os.path.join(fp, "test_Y.json"))
        )
        if should_preprocess:
            # TODO: Implement more datasets
            if os.path.basename(fp) == DatasetName.MULTIWOZ:
                # Load original data
                with open(os.path.join(fp, "train", "logs.json"), "r") as f:
                    train_logs = json.load(f)
                with open(os.path.join(fp, "train", "labels.json"), "r") as f:
                    train_labels = multiwoz_utils.preprocess_labels(json.load(f))
                with open(os.path.join(fp, "val", "logs.json"), "r") as f:
                    val_logs = json.load(f)
                with open(os.path.join(fp, "val", "labels.json"), "r") as f:
                    val_labels = multiwoz_utils.preprocess_labels(json.load(f))
                with open(os.path.join(fp, "knowledge.json"), "r") as f:
                    knowledge = json.load(f)

                self._docs = multiwoz_utils.get_docs(knowledge)

                self._train_X, self._train_Y = multiwoz_utils.get_XY(
                    train_logs, train_labels, knowledge, self._docs
                )
                self._test_X, self._test_Y = multiwoz_utils.get_XY(
                    val_logs, val_labels, knowledge, self._docs
                )

                # Save preprocessed files
                with open(os.path.join(fp, "docs.json"), "w") as f:
                    json.dump(self._docs, f, cls=DataClassEncoder, indent=4)
                with open(os.path.join(fp, "train_X.json"), "w") as f:
                    json.dump(self._train_X, f, cls=DataClassEncoder, indent=4)
                with open(os.path.join(fp, "train_Y.json"), "w") as f:
                    json.dump(self._train_Y, f, cls=DataClassEncoder, indent=4)
                with open(os.path.join(fp, "test_X.json"), "w") as f:
                    json.dump(self._test_X, f, cls=DataClassEncoder, indent=4)
                with open(os.path.join(fp, "test_Y.json"), "w") as f:
                    json.dump(self._test_Y, f, cls=DataClassEncoder, indent=4)
            else:
                raise NotImplementedError("Dataset not supported.")
        else:
            with open(os.path.join(fp, "docs.json"), "r") as f:
                self._docs = json.load(f)
            with open(os.path.join(fp, "train_X.json"), "r") as f:
                self._train_X = [Sample(**v) for v in json.load(f)]
            with open(os.path.join(fp, "train_Y.json"), "r") as f:
                self._train_Y = [Label(**v) for v in json.load(f)]
            with open(os.path.join(fp, "test_X.json"), "r") as f:
                self._test_X = [Sample(**v) for v in json.load(f)]
            with open(os.path.join(fp, "test_Y.json"), "r") as f:
                self._test_Y = [Label(**v) for v in json.load(f)]

    @property
    def docs(self) -> Dict[str, str]:
        return self._docs

    @property
    def train_X(self) -> List[Sample]:
        return self._train_X

    @property
    def train_Y(self) -> List[Label]:
        return self._train_Y

    @property
    def test_X(self) -> List[Sample]:
        return self._test_X

    @property
    def test_Y(self) -> List[Label]:
        return self._test_Y
