# ConvQA
Final Project for Conversational AI Project


**Members**: 
- Amaan Khan (amaanmk2)
- Eunice Chan (ecchan2)
- Yi-Chia Chang (yichia3)

## Download the datasets
```
$ bash download_datasets.sh
```
## Setup environment
```
$ conda create -n convqa python=3.10
$ conda activate convqa
$ pip install -r requirements.txt
```
## Run experiments
```
$ python3 main.py --model meta-llama/Llama-3.2-3B-Instruct --dataset data/CoQA --no_summary_tree --exp_name llama-coqa
```
*Llama access required: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
