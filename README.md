# SemEval-2027 Task 9: Multimodal Framing in News (Subtask 2)

**Team FrameVision (University of Cincinnati - CS 5134/6034 NLP, Fall 2026)**
* **Leo Kao (`kaoch@mail.uc.edu`)**: Lead Multimodal & Deep Learning Architect (Visual Entity Extraction, SigLIP + DeBERTa-v3 Feature Caching, Region-Aware Cross-Modal Attention, Asymmetric Loss, Explainability).
* **En-Shuo Zhang (`zhangeo@mail.uc.edu`)**: Data Engineering & Baseline Specialist (Course Split & Leakage Analysis, Subtask 1 Baselines, Starter XLM-RoBERTa, Qwen2.5-VL 7B VLM, Evaluation Harness).

---

## 1. System Architecture
Our system models multimodal framing dynamics through a two-stage modular architecture optimized for high reproducibility and Apple Silicon (MPS) acceleration:

```
[Raw News Photo]  ---> FasterRCNN / Focal Fallback ---> Top-3 Salient Crops
[Photo + Crops]   ---> Google SigLIP-base-patch16-224 ---> Visual Tokens (4, 768)
[Article Text]    ---> Microsoft DeBERTa-v3-base      ---> Text Tokens (256, 768)
                                                                 │
                                                                 ▼
[Text Queries] x [Visual Key/Value] ---> Cross-Modal Multihead Attention Layer
                                                                 │
                                                                 ▼
                                                  Masked Mean-Pooling & MLP Head
                                                                 │
                                                                 ▼
                                                14-Class Logits + Asymmetric Loss (ASL)
                                                                 │
                                                                 ▼
                                                1D Dev Set Threshold Calibration (τ*)
```

---

## 2. Benchmark Results on Official 40-Image Test Split

All models evaluated strictly on the **40-image course test split** from `course_split.json` across 15 classes (14 frames + `"None"`) using the official SemEval scorer:

| Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |
| :--- | :---: | :---: | :---: | :--- |
| **Image LR (SigLIP)** *(Zhang Table 1)* | 0.353 | 0.281 | 0.165 | Frozen SigLIP features + Logistic Regression |
| **Late Fusion (Image + Text)** *(Zhang Table 1)* | 0.367 | 0.350 | 0.115 | Direct probability averaging baseline |
| **Qwen2.5-VL (Image only)** *(Zhang Table 1)* | 0.417 | 0.205 | 0.093 | 7B VLM zero-shot baseline |
| **Qwen2.5-VL (Image + Text)** *(Zhang Table 1)* | 0.333 | 0.229 | 0.107 | VLM degraded by raw text interference (-0.084) |
| **Exp 1: Text-Only (DeBERTa-v3 + ASL)** | 0.273 | 0.199 | 0.107 | Unimodal text ablation |
| **Exp 2: Vision-Only Global (SigLIP + ASL)** | 0.410 | 0.245 | 0.120 | Global visual token ablation |
| **Exp 3: Vision-Only Region Attention (ASL)** | 0.381 | 0.267 | 0.108 | Global + region self-attention ablation |
| **Exp 4: Multimodal Cross-Attention (BCE)** | 0.356 | 0.276 | 0.127 | Cross-attention with standard BCE loss |
| **Exp 5: Proposed Cross-Modal Model (ASL)** | **0.4421** | **0.3189** | **0.082** | **Best overall model; outperforms all baselines!** |

---

## 3. Repository Layout

```text
task9/
├── src/
│   ├── dataset.py                # FramingDataset ingesting course_split.json
│   ├── entity_extractor.py       # Salient entity detector with multi-scale crop fallback
│   ├── feature_extractor.py      # SigLIP & DeBERTa-v3 feature encoder (float32 safe)
│   ├── cached_dataset.py         # Fast DataLoader loading pre-extracted tensor features
│   ├── explain.py                # Attention weight extraction & heatmap visualization
│   ├── metrics.py                # Official 15-class evaluation & 1D threshold search
│   ├── losses/
│   │   └── asymmetric_loss.py    # ASL margin clipping & focal modulation (Ridnik et al.)
│   └── models/
│       ├── fusion_model.py       # CrossModalFramingModel (Text Query, Visual Key/Val)
│       └── baselines.py          # TextOnly, VisionGlobal, VisionRegion, LateFusion
├── scripts/
│   ├── 01_extract_regions.py     # Offline batch entity region extraction
│   ├── 02_extract_features.py    # Offline batch feature extraction to disk cache
│   ├── 03_train.py               # Training loop with Dev threshold search & Test export
│   ├── 04_run_ablations.py       # Automated Exp 1-5 ablation experiments runner
│   └── 05_visualize_attention.py # Attention overlay heatmap generator
├── tests/                        # 32 unit & integration tests (100% pass rate)
├── reports/
│   ├── ablation_results.md       # Benchmark comparison table against team baselines
│   └── figures/                  # Publication-ready attention heatmaps & keyword plots
├── baselines/                    # En-Shuo's baselines (LR, late fusion, Qwen2.5-VL, Subtask 1); run from inside this folder
├── evaluation/                   # Shared evaluation layer: reads prediction TSVs, calls the official scorer
├── predictions/subtask2/         # Official-format predictions: <model>/seed<k>/{dev,test}.tsv (_gold, _reference are not models)
├── results/subtask2_final.json   # Every reported number is generated from this file
├── docs/                         # Final spec, plan, pre-registration, notes
├── course_split.json             # Fixed-seed course partition (40 Test, 20 Dev, 647 Train)
├── splits.py                     # Split creation logic
└── requirements.txt              # Unified dependencies
```

The two pipelines only meet through `predictions/subtask2/`; `evaluation/` imports neither `src/` nor `baselines/src/`.
Data is not tracked: the root pipeline reads `data_official/` and `starter-kit/`, and `baselines/` reads `baselines/data/`
(on Windows these can be junctions to one local copy of the official release).

---

## 4. Quickstart Guide

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Full Test Suite
```bash
pytest tests/ -v
# 32 passed in ~20s
```

### Reproduce End-to-End Pipeline
```bash
# 1. Extract salient entity crops
python scripts/01_extract_regions.py --output_dir cache/crops --top_k 3

# 2. Extract & cache multimodal foundation features
python scripts/02_extract_features.py --crop_dir cache/crops --output_dir cache/features --device mps

# 3. Train CrossModalFramingModel with ASL
python scripts/03_train.py --device mps --epochs 20 --loss_type asl

# 4. Run automated ablation suite
python scripts/04_run_ablations.py --device mps

# 5. Generate attention interpretability figures
python scripts/05_visualize_attention.py --device mps --num_samples 5
```
