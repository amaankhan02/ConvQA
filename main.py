import os
import argparse
from tqdm import tqdm

from utils.structures import *
from utils.dataset import Dataset
from utils.method import ConvRef
from utils.scorer import Scorer


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", type=str)

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


def eval(prefix, X, Y, docs, method, scorer, fp) -> None:
    Y_hat = []
    for x in tqdm(X):
        Y_hat.append(method(x, docs))
        print(Y_hat[-1])

    # Save generated output
    with open(os.path.join(fp, f"{prefix}_Y_hat.json"), "w") as f:
        json.dump(Y_hat, f, indent=4)

    scorer(Y_hat, Y, save=os.path.join(fp, f"{prefix}_eval.json"))


if __name__ == "__main__":
    args = parse_args()

    fp = "results"
    if args.exp_name:
        fp = os.path.join(fp, args.exp_name)

    method = ConvRef(
        args.model,
        no_summary_tree=args.no_summary_tree,
        no_dialogue_KG=args.no_dialogue_KG,
    )
    scorer = Scorer(fp)

    dataset = Dataset(args.dataset)

    eval("train", dataset.train_X, dataset.train_Y, dataset.docs, method, scorer, fp)
    eval("test", dataset.test_X, dataset.test_Y, dataset.docs, method, scorer, fp)

    print("Finished!")
