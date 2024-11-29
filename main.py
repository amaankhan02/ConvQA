import os
import argparse
from tqdm import tqdm

from utils.structures import *
from utils.dataset import Dataset
from utils.method import ConvRef
from utils.scorer import Scorer


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", type=str)

    parser.add_argument(
        "--dataset",
        required=True,
        # format would generally be `data/*` e.g. data/MultiWOZ
        help="Provide filepath to dataset",
    )

    # Ablation
    parser.add_argument("--no_summary_tree", action="store_true")
    parser.add_argument("--no_dialogue_KG", action="store_true")

    # Results
    parser.add_argument("--exp_name", default="", type=str)

    return Arguments(**vars(parser.parse_args()))


def eval(prefix: str, X: List[Sample], Y: List[Label], docs: Dict[str, str], method: ConvRef, scorer: Scorer, fp: str) -> None:
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


if __name__ == "__main__":
    args = parse_args()

    # fp = "/home/ecchan2/ConvQA/results/Llama-3.1-8B-Instruct_llm_only/"
    # prefix = ""

    # dataset = Dataset(args.dataset)
    # yhat_fp = os.path.join(fp, f"{prefix}Y_hat.json")
    # if os.path.exists(yhat_fp):
    #     Y_hat = json.load(open(yhat_fp, "r"))
    # Y = dataset.train_Y + dataset.test_Y
    # scorer = Scorer(fp)
    # scorer(Y_hat, Y, save=os.path.join(fp, f"{prefix}eval.json"))
    # exit()

    fp = "results"
    if args.exp_name:
        fp = os.path.join(fp, args.exp_name)

    dataset = Dataset(args.dataset)

    method = ConvRef(args.model, not args.no_dialogue_KG)

    scorer = Scorer(fp)

    if not args.no_summary_tree:
        summary_trees_fp = os.path.join(args.dataset, f"summary_trees.json")
        if not os.path.exists(summary_trees_fp):
            from transformers import AutoModel
            model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
            method.generate_summary_trees(summary_trees_fp, dataset.docs, model)
        else:
            method.load_summary_trees(summary_trees_fp)
    eval("", dataset.train_X + dataset.test_X, dataset.train_Y + dataset.test_Y, dataset.docs, method, scorer, fp)

    print("Finished!")
