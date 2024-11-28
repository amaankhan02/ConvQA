import torch
import os
import numpy as np
from typing import Dict, List, Any
from transformers import pipeline

from utils.structures import *
from utils.data.squad_eval import compute_f1
from utils.constants import ANSWER_DELIM

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
        """
        Calculate word-level F1 scores between predicted and ground truth answers
        when documents are marked as relevant.
        """
        f1_scores = []
        
        for y_hat, y in zip(Y_hat, Y):
            # Only evaluate F1 when both predict document is relevant
            # If either predicts document not relevant, F1 score is 0
            # If both predict document relevant but no answer for either, then F1 score is 0
            both_docs_relevant = y_hat.document_relevant and y.document_relevant
            both_have_answer = y_hat.answer is not None and y.answer is not None
            curr_f1_score = 0.0
            
            if both_docs_relevant and both_have_answer:
                # TODO: do i only use the first answer or all of them with the delimiter splitted? for now, we use the first answer only
                y_ans = y.answer.split(ANSWER_DELIM)[0]
                y_hat_ans = y_hat.answer.split(ANSWER_DELIM)[0]
                curr_f1_score = float(compute_f1(y_ans, y_hat_ans))
            
            f1_scores.append(curr_f1_score)
                
        return {
            "f1": np.mean(f1_scores),
            "values": f1_scores
        }

    def retrieval(self, Y_hat: List[Label], Y: List[Label]) -> Dict[str, Any]:
        # Compare if the model correctly decided whether to retrieve or not
        retrieval_scores = []
        
        for y_hat, y in zip(Y_hat, Y):
            if y_hat.document_relevant == y.document_relevant:
                retrieval_scores.append(1)
            else:
                retrieval_scores.append(0)

        return {
            "accuracy": np.mean(retrieval_scores),
            "values": retrieval_scores
        }

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
        time = [y_hat["time_taken"] for y_hat in Y_hat]
        scores = {
            "relevance": self.relevance(Y_hat, Y),
            "retrieval": self.retrieval(Y_hat, Y),
            "answer": self.answer(Y_hat, Y),
            "time": {
                "average": np.mean(time),
                "standard_deviation": np.std(time),
            }
        }

        with open(os.path.join(self.fp, save), "w") as f:
            json.dump(scores, f, indent=4)
