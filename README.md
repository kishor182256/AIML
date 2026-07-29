# LoRA and QLoRA Fine-Tuning Project

Production-oriented starter project for preparing chat data, fine-tuning instruction models with LoRA or QLoRA, evaluating adapters, merging adapters into base models, and exporting models for local inference.

## Project Structure

```text
configs/
  lora.yaml
  qlora.yaml

data/
  raw/
  processed/
  train.jsonl
  validation.jsonl
  test.jsonl

models/
adapters/
merged_models/
logs/

scripts/
  01_download_dataset.py
  02_prepare_dataset.py
  03_train_lora.py
  04_train_qlora.py
  05_evaluate.py
  06_merge_adapter.py
  07_export_gguf.py

utils/
  config.py
  dataset.py
  modeling.py
  tokenization.py

requirements.txt
README.md
.env.example
.gitignore
```

## Setup

Use Python 3.10 or 3.11.

```powershell
cd C:\AIML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version before installing the rest of the requirements.

## Environment

Create your local `.env` from the example file:

```powershell
Copy-Item .env.example .env
```

Set `HF_TOKEN` if you need access to gated Hugging Face models or datasets.

## Data Preparation

Download and inspect the dataset:

```powershell
python scripts/01_download_dataset.py
```

Prepare the full UltraChat SFT JSONL files:

```powershell
python scripts/02_prepare_dataset.py
```

Use `--max-samples 50` only when you explicitly want a tiny smoke test.

Generated files under `data/raw/` and `data/processed/` are ignored by Git because they can be large.

On a fresh RunPod or Linux clone, run:

```bash
cd /workspace/AIML
pip install -r requirements.txt
python scripts/02_prepare_dataset.py
python scripts/03_train_lora.py
```

## LoRA Training

Check `configs/lora.yaml`, then run:

```powershell
python scripts/03_train_lora.py
```

The script trains on the full processed files configured in `configs/lora.yaml`.

## QLoRA

`scripts/04_train_qlora.py` is reserved for QLoRA training. QLoRA commonly depends on `bitsandbytes`, which is best supported on Linux or WSL with CUDA.

## GitHub Workflow

This repository is prepared to keep source code and lightweight placeholders in Git while ignoring local secrets, datasets, logs, model caches, adapters, and merged model outputs.

```powershell
git status
git add .
git commit -m "Prepare production LoRA fine-tuning project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Do not commit `.env`, downloaded datasets, model weights, adapters, or merged model artifacts.
