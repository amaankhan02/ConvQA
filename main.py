import os
import argparse
from tqdm import tqdm

from utils.structures import *
from utils.dataset import Dataset
from utils.method import ConvRef
from utils.scorer import Scorer
from utils.evaluate import run_inference_and_evaluate
"""
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset data/CoQA --llm_only --exp_name CoQA/llm_only
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset data/QuAC --llm_only --exp_name QuAC/llm_only
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset data/MultiWOZ --llm_only --exp_name MultiWOZ/llm_only
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset data/CoQA --no_summary_tree --exp_name CoQA/ours
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset data/QuAC --no_summary_tree --exp_name QuAC/ours
"""
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
    parser.add_argument("--llm_only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no_summary_tree", action="store_true")
    # parser.add_argument("--no_dialogue_KG", action="store_true")
    
    # Stage 1 and Stage 2 ablation flags
    parser.add_argument("--use_gt_segments", action="store_true", 
                        help="Flag for Stage 1 ablation. Use GT segments instead of searching for it.")
    parser.add_argument("--use_gt_doc_relevancy", action="store_true", 
                        help="Flag for Stage 2 ablation. Use GT document relevancy instead of computing it.")

    # Results
    parser.add_argument("--exp_name", default="", type=str)

    return Arguments(**vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_args()

    fp = "results"
    if args.exp_name:
        fp = os.path.join(fp, args.exp_name)

    dataset = Dataset(args.dataset)

    method = ConvRef(args.model, args.llm_only, args.strict, args.use_gt_segments, args.use_gt_doc_relevancy)

    scorer = Scorer(fp)

    if not args.no_summary_tree:
        summary_trees_fp = os.path.join(args.dataset, f"summary_trees.json")
        if not os.path.exists(summary_trees_fp):
            from transformers import AutoModel
            model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
            method.generate_summary_trees(summary_trees_fp, dataset.docs, model)
        else:
            method.load_summary_trees(summary_trees_fp)
            
    print("Running inference and evaluation...")
    X = dataset.test_X[:100]# dataset.train_X + dataset.test_X
    Y = dataset.test_Y[:100]# dataset.train_Y + dataset.test_Y
    run_inference_and_evaluate("", X, Y, dataset.docs, method, scorer, fp)

    print("Finished!")
