# Create data directories if they don't exist
New-Item -ItemType Directory -Force -Path ".\data\CoQA"
New-Item -ItemType Directory -Force -Path ".\data\QuAC"
New-Item -ItemType Directory -Force -Path ".\data\MultiWOZ\train"
New-Item -ItemType Directory -Force -Path ".\data\MultiWOZ\val"

# Define all dataset paths
$coqaTrainPath = ".\data\CoQA\coqa-train-v1.0.json"
$coqaDevPath = ".\data\CoQA\coqa-dev-v1.0.json"
$quacTrainPath = ".\data\QuAC\train_v0.2.json"
$quacDevPath = ".\data\QuAC\val_v0.2.json"

# Define MultiWOZ files
$multiWozFiles = @(
    @{
        Url = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/train/labels.json"
        Path = ".\data\MultiWOZ\train\labels.json"
    },
    @{
        Url = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/train/logs.json"
        Path = ".\data\MultiWOZ\train\logs.json"
    },
    @{
        Url = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/val/labels.json"
        Path = ".\data\MultiWOZ\val\labels.json"
    },
    @{
        Url = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/val/logs.json"
        Path = ".\data\MultiWOZ\val\logs.json"
    },
    @{
        Url = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master/data/knowledge.json"
        Path = ".\data\MultiWOZ\knowledge.json"
    }
)

# Download CoQA and QuAC datasets
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

# Download MultiWOZ files
foreach ($file in $multiWozFiles) {
    if (!(Test-Path $file.Path)) {
        Write-Host "Downloading MultiWOZ file to $($file.Path)..."
        Invoke-WebRequest -Uri $file.Url -OutFile $file.Path
        if (Test-Path $file.Path) {
            Write-Host "Successfully downloaded $($file.Path)"
        } else {
            Write-Host "Failed to download $($file.Path)"
        }
    } else {
        Write-Host "MultiWOZ file $($file.Path) already exists, skipping download."
    }
}

Write-Host "All downloads completed!"