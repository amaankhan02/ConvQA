# ConvQA
Final Project for Conversational AI Project


**Members**: 
- Amaan Khan (amaanmk2)
- Eunice Chan (ecchan2)
- Yi-Chia Chang (yichia3)

## Download the datasets
```bash
$ bash download_datasets.sh (MAC/Linux)
$ .\download_datasets.ps1 (Windows)
```
Running the download script will download the QuAC, CoQA, and Augmented MultiWOZ dataset. However, these are the original versions of the dataset. We later preprocesses and standardize the three datasets. When you `main.py` (see section below on further explanation), it will automatically preprocess the datasets to a standardized version (as described in the paper) if it has not already been done so. That is, the first time that main.py is ran, it will preprocess the datasets. After which it wouldn't need to since those files already exist (unless they are deleted).

## Setup environment
```
$ conda create -n convqa python=3.10
$ conda activate convqa
$ pip install -r requirements.txt
```
## Run experiments
To run the experiments, you can run the following `main.py` file as such
```bash
$ python3 main.py --model meta-llama/Llama-3.2-3B-Instruct --dataset data/CoQA --no_summary_tree --exp_name llama-coqa
```
*Llama access required: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Here is a breakdown of all the arguments you can pass into `main.py`
### Model (`--model`)
- **Type**: String
- **Default**: "meta-llama/Llama-3.2-3B-Instruct"
- **Required**: No
- **Purpose**: Specifies the large language model to be used

### Dataset (`--dataset`)
- **Type**: String
- **Default**: None
- **Required**: Yes
- **Purpose**: Filepath to the dataset being used

### LLM Only (`--llm_only`)
- **Type**: Boolean flag
- **Default**: False
- **Required**: No
- **Purpose**: Ablation flag to use only the language model and not our method (as described in the paper and presentation)

### Strict (`--strict`)
- **Type**: Boolean flag
- **Default**: False
- **Required**: No
- **Purpose**: flag to enable strict mode (`Ours_strict` model as discussed in the paper/presentation)
- **Example**: `python script.py --strict`

### No Summary Tree (`--no_summary_tree`)
- **Type**: Boolean flag
- **Default**: False
- **Required**: No
- **Purpose**: flag to disable summary tree functionality
- **Example**: `python script.py --no_summary_tree`

### Use Ground Truth Segments (`--use_gt_segments`)
- **Type**: Boolean flag
- **Default**: False
- **Required**: No
- **Purpose**: Stage 1 ablation flag to use ground truth segments instead of searching for them
- **Example**: `python script.py --use_gt_segments`

### Use Ground Truth Document Relevancy (`--use_gt_doc_relevancy`)
- **Type**: Boolean flag
- **Default**: False
- **Required**: No
- **Purpose**: Stage 2 ablation flag to use ground truth document relevancy instead of computing it
- **Example**: `python script.py --use_gt_doc_relevancy`

### Experiment Name (`--exp_name`)
- **Type**: String
- **Default**: "" (empty string)
- **Required**: No
- **Purpose**: Name of the experiment you are running
- **Example**: `python script.py --exp_name "experiment_1"`

A full example combining multiple arguments might look like:

```bash
python script.py --dataset data/MultiWOZ --model "meta-llama/Llama-7B" --exp_name "ablation_test" --use_gt_segments --strict
