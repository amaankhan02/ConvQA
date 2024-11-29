# ConvQA
Final Project for Conversational AI Project

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
$ python3 main.py --model meta-llama/Llama-3.1-8B-Instruct --dataset data/CoQA/ --exp_name llama-coqa
```
*Llama access required: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
