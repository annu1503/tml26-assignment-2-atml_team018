# tml26-assignment-2-team_XIX
# Stolen Model Detection — TML26 Task 2
**Team:** atml_team018  
**Best leaderboard score:** 0.6111 (TPR@5%FPR)

## Task
Detect which of 360 suspect models were stolen (copied, fine-tuned, or distilled) from a target ResNet-18 trained on a CIFAR-100 subset.

## How to Reproduce the Best Result

### 1. Requirements
```bash
pip install torch torchvision safetensors pandas numpy tqdm
```

### 2. Directory Structure
Your project should look like this:
```
TML26_TASK2/
├── detect_stolen_models.py
├── submission.py
├── task_template.py
├── train_main_idx.json
├── target_model/
│   └── weights.safetensors
├── suspect_models/
│   ├── suspect_000.safetensors
│   └── ... (up to suspect_359.safetensors)
└── data/
    └── cifar-100-python/   ← already present
```

### 3. Download Suspect Models
Download the suspect models from HuggingFace:
```bash
git clone https://huggingface.co/datasets/tml26/tml26-task2-suspect-models suspect_models/
```

### 4. Run Detection
```bash
python detect_stolen_models.py
```

This will:
- Use CIFAR-100 from `./data/` (already downloaded)
- Score all 360 suspect models using weight, output, and activation signals
- Save progress to `checkpoint_v5.csv` (safe to interrupt and resume)
- Write final scores to `submission.csv`


### 5. Submit
Update `FILE_PATH` in `submission.py` to point to your `submission.csv`, then run:
```bash
python submission.py
```
