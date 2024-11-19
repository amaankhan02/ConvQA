# Create data directories if they don't exist
New-Item -ItemType Directory -Force -Path ".\data\CoQA"
New-Item -ItemType Directory -Force -Path ".\data\QuAC"
# TODO: add implementation for downloading MultiWOZ

# Download the training and dev files for CoQA
Write-Host "Downloading CoQA training set..."
Invoke-WebRequest -Uri "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json" -OutFile ".\data\CoQA\coqa-train-v1.0.json"

Write-Host "Downloading CoQA dev set..."
Invoke-WebRequest -Uri "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json" -OutFile ".\data\CoQA\coqa-dev-v1.0.json"

# Download the training and dev files for QuAC
Write-Host "Downloading QuAC training set..."
Invoke-WebRequest -Uri "https://s3.amazonaws.com/my89public/quac/train_v0.2.json" -OutFile ".\data\QuAC\train_v0.2.json"

Write-Host "Downloading QuAC dev set..."
Invoke-WebRequest -Uri "https://s3.amazonaws.com/my89public/quac/val_v0.2.json" -OutFile ".\data\QuAC\val_v0.2.json"

Write-Host "Download complete!"