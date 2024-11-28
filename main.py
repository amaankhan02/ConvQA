import os
import argparse
from tqdm import tqdm

from utils.structures import *
from utils.dataset import Dataset
from utils.method import ConvRef
from utils.scorer import Scorer
from utils.evaluate import run_inference_and_evaluate


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


if __name__ == "__main__":
    args = parse_args()

    fp = "results"
    if args.exp_name:
        fp = os.path.join(fp, args.exp_name)

    method = ConvRef(args.model, not args.no_dialogue_KG)

    scorer = Scorer(fp)

    dataset = Dataset(args.dataset)

    if not args.no_summary_tree:
        summary_trees_fp = os.path.join(args.dataset, f"summary_trees.json")
        if not os.path.exists(summary_trees_fp):
            from transformers import AutoModel
            model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
            method.generate_summary_trees(summary_trees_fp, dataset.docs, model)
        else:
            method.load_summary_trees(summary_trees_fp)
            
    print("Running inference and evaluation...")
    run_inference_and_evaluate("", dataset.train_X + dataset.test_X, dataset.train_Y + dataset.test_Y, dataset.docs, method, scorer, fp)

    print("Finished!")
