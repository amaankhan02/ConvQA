#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ./data/CoQA
mkdir -p ./data/QuAC
# TODO: add implementation for downloading MultiWOZ

# Download the training and dev files for CoQA
coqa_train_path="./data/CoQA/coqa-train-v1.0.json"
coqa_dev_path="./data/CoQA/coqa-dev-v1.0.json"
quac_train_path="./data/QuAC/train_v0.2.json"
quac_dev_path="./data/QuAC/val_v0.2.json"

if [ ! -f "$coqa_train_path" ]; then
    echo "Downloading CoQA training set..."
    curl -o "$coqa_train_path" https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json
else
    echo "CoQA training set already exists, skipping download."
fi

if [ ! -f "$coqa_dev_path" ]; then
    echo "Downloading CoQA dev set..."
    curl -o "$coqa_dev_path" https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
else
    echo "CoQA dev set already exists, skipping download."
fi

if [ ! -f "$quac_train_path" ]; then
    echo "Downloading QuAC training set..."
    curl -o "$quac_train_path" https://s3.amazonaws.com/my89public/quac/train_v0.2.json
else
    echo "QuAC training set already exists, skipping download."
fi

if [ ! -f "$quac_dev_path" ]; then
    echo "Downloading QuAC dev set..."
    curl -o "$quac_dev_path" https://s3.amazonaws.com/my89public/quac/val_v0.2.json
else
    echo "QuAC dev set already exists, skipping download."
fi

echo "Download complete!"

# NOTE: make sure to run chmod +x download_datasets.sh to make the script executable