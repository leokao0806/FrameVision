# SemEval-2027 Task 9: Multimodal Framing in News (Subtask 2)

**Team FrameVision**
* **Leo Kao (Lead Multimodal & Deep Learning Architect)**: End-to-end PyTorch pipeline, visual entity extraction (SAM), cross-modal attention fusion, and Asymmetric Loss.
* **En-Shuo Zhang (Data Engineering & Baseline Specialist)**: Text-only baseline (DeBERTa), VLM prompt benchmark (LLaVA/Qwen2-VL), and evaluation harness.

---

## Architecture Overview
The system models multimodal framing dynamics by combining:
1. **News Text**: Headlines and article paragraphs.
2. **Global Visual Context**: Full news photography.
3. **Local Salient Entities**: Segmented regions (protest banners, facial expressions, law enforcement insignia) extracted via **Segment Anything (SAM) / Grounding-DINO**.
4. **Cross-Attention Fusion Layer**: Text actively queries both global and local visual features to resolve cross-modal framing alignment or contradiction.
5. **Asymmetric Loss (ASL)**: Dynamically downweights prevalent negative labels to handle extreme class sparsity across the 14 policy framing dimensions.

---

## Directory Structure
```text
task9/
├── data/
│   ├── labels.txt                             # 14 framing dimension categories
│   └── EN/
│       ├── raw-documents/                     # Raw paired news texts and images
│       ├── subtask-2-annotations.txt          # Multi-label annotations for Subtask 2
│       └── subtask-1-annotations.txt          # Paragraph-level annotations for Subtask 1
├── crops/                                     # Pre-extracted salient entity crops (SAM)
├── dataset.py                                 # PyTorch FramingMultimodalDataset
├── sam_extractor.py                           # Offline salient crop extractor (SAM/Saliency)
├── model.py                                   # Cross-modal Attention Network + ASL Loss
├── metrics.py                                 # Macro-F1, Micro-F1, and Hamming Loss
├── train.py                                   # Training and evaluation loop (MPS/CUDA/CPU)
├── demo_dryrun.py                             # Verification unit test
└── requirements.txt                           # Dependencies
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
cd /Users/leokao/Documents/Python/task9
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Extract Salient Crops (Offline Preprocessing)
Run this once to pre-extract localized visual crops from news photos:
```bash
python sam_extractor.py --input_dir data/EN/raw-documents --output_dir crops
```

### 3. Verify Pipeline (Dry-Run)
Verify dataset ingestion, forward pass, loss calculation, and backpropagation:
```bash
python demo_dryrun.py
```

### 4. Train the Model
Trains with automatic device selection (`mps` for Apple Silicon, `cuda` for NVIDIA, or `cpu`):
```bash
python train.py --epochs 15 --batch_size 4 --lr 1e-4 --use_asl
```
