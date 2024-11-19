#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ./data/CoQA
mkdir -p ./data/QuAC
# TODO: add implementation for downloading MultiWOZ

# Download the training and dev files for CoQA
echo "Downloading CoQA training set..."
curl -o ./data/CoQA/coqa-train-v1.0.json https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json

echo "Downloading CoQA dev set..."
curl -o ./data/CoQA/coqa-dev-v1.0.json https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json

# Download the training and dev files for QuAC
echo "Downloading QuAC training set..."
curl -o ./data/QuAC/train_v0.2.json https://s3.amazonaws.com/my89public/quac/train_v0.2.json


echo "Downloading QuAC dev set..."
curl -o ./data/QuAC/val_v0.2.json https://s3.amazonaws.com/my89public/quac/val_v0.2.json

echo "Download complete!"

# NOTE: make sure to run chmod +x download_datasets.sh to make the script executable