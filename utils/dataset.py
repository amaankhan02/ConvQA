import json
import os
from typing import Any, Dict, List

from .data import coqa_utils, multiwoz_utils, quac_utils
from .scorer import Scorer
from .structures import DataClassEncoder, DatasetName, Label, Sample


# TODO: put all the filenames in a config file instead of it being literal strings here
class Dataset:
    def __init__(self, fp: str) -> None:
        """Initialize the dataset.

        Args:
            fp (str): path to the dataset folder

        Raises:
            NotImplementedError: raises NotImplementedError if the dataset is not supported
        """

        # Check if the preprocessed files exist. if so, no need to preprocess
        should_preprocess = not (
            os.path.exists(os.path.join(fp, "docs.json"))
            and os.path.exists(os.path.join(fp, "train_X.json"))
            and os.path.exists(os.path.join(fp, "train_Y.json"))
            and os.path.exists(os.path.join(fp, "test_X.json"))
            and os.path.exists(os.path.join(fp, "test_Y.json"))
        )
        if should_preprocess:
            if os.path.basename(fp) == DatasetName.MULTIWOZ:
                self._setup_multiwoz(fp)
            elif os.path.basename(fp) == DatasetName.QUAC:
                self._setup_quac(fp)
            elif os.path.basename(fp) == DatasetName.COQA:
                self._setup_coqa(fp)
            else:
                raise NotImplementedError("Dataset not supported.")
        else:
            self._load_preprocessed_data(fp)

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

    def _load_preprocessed_data(self, fp: str) -> None:
        """Helper function to load the preprocessed data files."""
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

    def _setup_multiwoz(self, fp: str) -> None:
        """Helper function to load the original MultiWOZ dataset, preprocess it and save the preprocessed files."""
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
        self._save_preprocessed_files(
            fp, self._docs, self._train_X, self._train_Y, self._test_X, self._test_Y
        )

    def _setup_quac(self, fp: str) -> None:
        """Helper function to load the original QuAC dataset, preprocess it and save the preprocessed files."""
        # Load original data
        with open(os.path.join(fp, "train_v0.2.json"), "r") as f:
            train_data = json.load(f)
        with open(os.path.join(fp, "val_v0.2.json"), "r") as f:
            val_data = json.load(f)

        self._docs = quac_utils.get_docs(train_data)

        self._train_X, self._train_Y = quac_utils.get_XY(train_data)
        self._test_X, self._test_Y = quac_utils.get_XY(val_data)

        self._save_preprocessed_files(
            fp, self._docs, self._train_X, self._train_Y, self._test_X, self._test_Y
        )

    def _setup_coqa(self, fp: str) -> None:
        """Helper function to load the original CoQA dataset, preprocess it and save the preprocessed files."""
        # Load original data
        with open(os.path.join(fp, "coqa-train-v1.0.json"), "r") as f:
            train_data = json.load(f)
        with open(os.path.join(fp, "coqa-dev-v1.0.json"), "r") as f:
            val_data = json.load(f)

        self._docs = coqa_utils.get_docs(train_data)

        self._train_X, self._train_Y = coqa_utils.get_XY(train_data)
        self._test_X, self._test_Y = coqa_utils.get_XY(val_data)

        # Save preprocessed files
        self._save_preprocessed_files(
            fp, self._docs, self._train_X, self._train_Y, self._test_X, self._test_Y
        )

    def _save_preprocessed_files(
        self,
        fp: str,
        docs: Dict[str, str],
        train_X: List[Sample],
        train_Y: List[Label],
        test_X: List[Sample],
        test_Y: List[Label],
    ) -> None:
        """Helper function to save the preprocessed files."""
        with open(os.path.join(fp, "docs.json"), "w") as f:
            json.dump(docs, f, cls=DataClassEncoder, indent=4)
        with open(os.path.join(fp, "train_X.json"), "w") as f:
            json.dump(train_X, f, cls=DataClassEncoder, indent=4)
        with open(os.path.join(fp, "train_Y.json"), "w") as f:
            json.dump(train_Y, f, cls=DataClassEncoder, indent=4)
        with open(os.path.join(fp, "test_X.json"), "w") as f:
            json.dump(test_X, f, cls=DataClassEncoder, indent=4)
        with open(os.path.join(fp, "test_Y.json"), "w") as f:
            json.dump(test_Y, f, cls=DataClassEncoder, indent=4)
