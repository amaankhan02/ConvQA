import os
import json
from typing import List, Dict
from tqdm import tqdm

from .structures import Sample, Label, DataClassEncoder
from .method import ConvRef
from .scorer import Scorer

def run_inference_and_evaluate(
    prefix: str, 
    X: List[Sample], 
    Y: List[Label], 
    docs: Dict[str, str], 
    method: ConvRef, 
    scorer: Scorer, 
    fp: str
) -> None:
    """
    Evaluate model predictions and save results
    
    Args:
        prefix: Prefix for output files
        X: List of input samples
        Y: List of ground truth labels  
        docs: Dictionary of documents
        method: Model/method to generate predictions
        scorer: Scorer object for evaluation
        fp: Output file path
    """
    Y_hat = []
    yhat_fp = os.path.join(fp, f"{prefix}Y_hat.json")
    if os.path.exists(yhat_fp):
        Y_hat = json.load(open(yhat_fp, "r"))
    
    for i, x in tqdm(enumerate(X)):
        if i < len(Y_hat):
            continue
        Y_hat.append(method(x, docs))
        print(Y_hat[-1])

        # Save generated output
        with open(yhat_fp, "w") as f:
            json.dump(Y_hat, f, cls=DataClassEncoder, indent=4)

    scorer(Y_hat, Y, save=os.path.join(fp, f"{prefix}eval.json"))