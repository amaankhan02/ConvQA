import torch
import os
import numpy as np
from typing import Dict, List, Any
from transformers import pipeline

from .structures import *


class Scorer:
    def __init__(
        self,
        fp: str,
        evaluator: str = "meta-llama/Llama-3.2-1B-Instruct",
    ) -> None:
        self.evaluator = pipeline(
            "text-generation",
            model=evaluator,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.evaluator.model.generation_config.pad_token_id = (
            self.evaluator.tokenizer.eos_token_id
        )

        self.fp = fp
        os.makedirs(self.fp, exist_ok=True)

    def relevance(self, Y_hat: List[Label], Y: List[Label]) -> Dict[str, Any]:
        raise NotImplementedError("Not implemented yet.")

    def retrieval(self, Y_hat: List[Label], Y: List[Label]) -> Dict[str, Any]:
        raise NotImplementedError("Not implemented yet.")

    def answer(self, Y_hat: List[Label], Y: List[Label]) -> Dict[str, Any]:
        answers = []

        for y_hat, y in zip(Y_hat, Y):
            prompt = [
                {
                    "role": "user",
                    "content": '1. {y_hat.answer}\n2.{retrieval.answer}\n\nDo 1 & 2 give the same answer? Answer "YES" or "NO" only.',
                }
            ]

            outputs = self.evaluator(prompt, max_new_tokens=10)
            response = outputs[0]["generated_text"][-1]["content"].upper()

            if "NO" not in response and "YES" in response:
                answers.append(1)
            else:
                # Consider as answered incorrectly
                answers.append(0)

        return {
            "accuracy": np.mean(answers),
            "values": answers,
        }

    def __call__(self, Y_hat: List[Label], Y: List[Label], save="score.json") -> None:
        time = [y_hat.time_taken for y_hat in Y_hat]
        scores = {
            "relevance": None,  # TODO: self.relevance(),
            "retrieval": None,  # TODO: self.retrieval(),
            "answer": self.answer(Y_hat, Y),
            "time": {
                "average": np.mean(time),
                "standard_deviation": np.std(time),
            }
        }

        with open(os.path.join(self.fp, save), "w") as f:
            json.dump(scores, f, indent=4)
