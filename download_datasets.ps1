# Create data directories if they don't exist
New-Item -ItemType Directory -Force -Path ".\data\CoQA"
New-Item -ItemType Directory -Force -Path ".\data\QuAC"
# TODO: add implementation for downloading MultiWOZ

# Download the training and dev files for CoQA
$coqaTrainPath = ".\data\CoQA\coqa-train-v1.0.json"
$coqaDevPath = ".\data\CoQA\coqa-dev-v1.0.json"
$quacTrainPath = ".\data\QuAC\train_v0.2.json"
$quacDevPath = ".\data\QuAC\val_v0.2.json"

if (!(Test-Path $coqaTrainPath)) {
    Write-Host "Downloading CoQA training set..."
    Invoke-WebRequest -Uri "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json" -OutFile $coqaTrainPath
} else {
    Write-Host "CoQA training set already exists, skipping download."
}

if (!(Test-Path $coqaDevPath)) {
    Write-Host "Downloading CoQA dev set..."
    Invoke-WebRequest -Uri "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json" -OutFile $coqaDevPath
} else {
    Write-Host "CoQA dev set already exists, skipping download."
}

if (!(Test-Path $quacTrainPath)) {
    Write-Host "Downloading QuAC training set..."
    Invoke-WebRequest -Uri "https://s3.amazonaws.com/my89public/quac/train_v0.2.json" -OutFile $quacTrainPath
} else {
    Write-Host "QuAC training set already exists, skipping download."
}

if (!(Test-Path $quacDevPath)) {
    Write-Host "Downloading QuAC dev set..."
    Invoke-WebRequest -Uri "https://s3.amazonaws.com/my89public/quac/val_v0.2.json" -OutFile $quacDevPath
} else {
    Write-Host "QuAC dev set already exists, skipping download."
}

Write-Host "Download complete!"