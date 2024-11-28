#!/bin/bash

# Create data directories if they don't exist
mkdir -p ./data/CoQA
mkdir -p ./data/QuAC
mkdir -p ./data/MultiWOZ/train
mkdir -p ./data/MultiWOZ/val

# Define all dataset paths
coqa_train_path="./data/CoQA/coqa-train-v1.0.json"
coqa_dev_path="./data/CoQA/coqa-dev-v1.0.json"
quac_train_path="./data/QuAC/train_v0.2.json"
quac_dev_path="./data/QuAC/val_v0.2.json"

# Define MultiWOZ files as an array of URL/path pairs
declare -A multiwoz_files
multiwoz_files=(
    ["./data/MultiWOZ/train/labels.json"]="https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/train/labels.json"
    ["./data/MultiWOZ/train/logs.json"]="https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/train/logs.json"
    ["./data/MultiWOZ/val/labels.json"]="https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/val/labels.json"
    ["./data/MultiWOZ/val/logs.json"]="https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/val/logs.json"
    ["./data/MultiWOZ/knowledge.json"]="https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/knowledge.json"
)

# Download CoQA and QuAC datasets
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

# Download MultiWOZ files
for filepath in "${!multiwoz_files[@]}"; do
    if [ ! -f "$filepath" ]; then
        echo "Downloading MultiWOZ file to $filepath..."
        curl -o "$filepath" "${multiwoz_files[$filepath]}"
        if [ -f "$filepath" ]; then
            echo "Successfully downloaded $filepath"
        else
            echo "Failed to download $filepath"
        fi
    else
        echo "MultiWOZ file $filepath already exists, skipping download."
    fi
done

echo "All downloads completed!"

# NOTE: make sure to run chmod +x download_datasets.sh to make the script executable