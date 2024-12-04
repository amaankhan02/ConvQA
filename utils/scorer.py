import torch
import os
import numpy as np
from tqdm import tqdm
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
                y_ans_list = y.answer.split(ANSWER_DELIM)
                y_hat_ans = y_hat.answer
                curr_f1_score = max([float(compute_f1(y_ans, y_hat_ans)) for y_ans in y_ans_list])
            
            f1_scores.append(curr_f1_score)
                
        return {
            "f1": np.mean(f1_scores),
            "values": f1_scores
        }

    def retrieval(self, Y_hat: List[Label], Y: List[Label]) -> Dict[str, Any]:
        # Compare if the model correctly decided whether to retrieve or not
        retrieval_scores = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for y_hat, y in zip(Y_hat, Y):
            if y_hat.document_relevant == y.document_relevant:
                if y.document_relevant == False:
                    TN += 1
                else:
                    TP += 1
                retrieval_scores.append(1)
            else:
                if y.document_relevant == False:
                    FP += 1
                else:
                    FN += 1
                retrieval_scores.append(0)

        return {
            "accuracy": np.mean(retrieval_scores),
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "values": retrieval_scores
        }

    def answer(self, X: List[Sample], Y_hat: List[Label], Y: List[Label]) -> Dict[str, Any]:
        answers = []

        for x, y_hat, y in tqdm(zip(X, Y_hat, Y)):
            if (y.answer == None and y_hat.answer != None) or (y.answer != None and y_hat.answer == None):
                # Consider as answered incorrectly
                answers.append(0)
            elif y.answer == None and y_hat.answer == None:
                answers.append(1)
            else:
                query = x.conversation[-1]["content"]
                prompt = [
                    {
                        "role": "user",
                        "content": f'[QUESTION] {query}'
                    },
                    {
                        "role": "user",
                        "content": f'[ANSWER 1] {y_hat.answer}'
                    },
                    {
                        "role": "user",
                        "content": f'[ANSWER 2] {y.answer}'
                    },
                    {
                        "role": "user",
                        "content": f'Are Answers 1 & 2 consistent with each other and convey roughly the same idea? Answer only \"YES, THEY ARE CONSISTENT.\" or \"NO, THEY ARE NOT CONSISTENT.\"',
                    }
                ]

                outputs = self.evaluator(prompt, max_new_tokens=10)
                response = outputs[0]["generated_text"][-1]["content"]
                print(prompt, response)

                if response.startswith("YES"):
                    answers.append(1)
                else:
                    # Consider as answered incorrectly
                    answers.append(0)

        return {
            "accuracy": np.mean(answers),
            "values": answers,
        }

    def __call__(self, X: List[Sample], Y_hat: List[Label], Y: List[Label], save="score.json") -> None:
        time = [y_hat.time_taken for y_hat in Y_hat]
        scores = {
            "relevance": self.relevance(Y_hat, Y),
            "retrieval": self.retrieval(Y_hat, Y),
            "answer": self.answer(X, Y_hat, Y),
            "time": {
                "average": np.mean(time),
                "standard_deviation": np.std(time),
            }
        }
        for k, v in scores.items():
            if k == "time":
                print(k, round(v['average'], 3), f"(± {round(v['standard_deviation'], 3)})")
            else:
                metric_name = 'f1'
                metric = v.get(metric_name, None)
                if not metric:
                    metric_name = 'accuracy'
                    metric = v.get(metric_name, None) * 100
                print(k, metric_name, metric)

        with open(os.path.join(save), "w") as f:
            json.dump(scores, f, indent=4)
