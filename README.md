# SemEval-2027 Task 9: Analysis of Multimodal Framing in the News (Subtask 2a)

**Team 1 — FrameVision (University of Cincinnati · CS 5134/6034 NLP, Fall 2026)**
* **Leo Kao (`kaoch@mail.uc.edu`)**: Multimodal & Deep Learning Architect (Frame-Neutral Grounding-DINO Regions, Unified SigLIP Feature Caching, Cross-Modal Attention, Asymmetric Loss, Interpretability & Slides).
* **En-Shuo Zhang (`zhangeo@mail.uc.edu`)**: Data & Evaluation Lead (Anti-Leakage Course Splits, Subtask 1 Baselines, Zero-Shot Qwen2.5-VL E10/E12, Unified Official Scorer & C1–C4 Analyses).

---

## 1. Task Overview & Official Sources

This repository implements Team 1 (FrameVision)'s solution for **SemEval-2027 Task 9: Analysis of Multimodal Framing in the News** ([Official Shared Task Website](https://propaganda.math.unipd.it/semeval2027task9/)). We target **Subtask 2a (Unguided Image Framing)**: given a news article and its accompanying photograph, predict which of the **14 generic Media Frames Corpus categories** (Card et al., 2015)—or **`None`** (15th class)—are conveyed by the **image itself**.

<p align="center">
  <img src="assets/semeval2027_task9_example.jpg" width="460" alt="SemEval-2027 Task 9: Migration Text and Image Framing Example">
</p>
<p align="center">
  <em>Official task illustration from the <a href="https://propaganda.math.unipd.it/semeval2027task9/">SemEval-2027 Task 9 homepage</a>: Image 1 reinforces a textual frame (<code>Economic</code>), whereas Image 2 introduces a frame absent from the text (<code>Public_opinion</code>).</em>
</p>

### Official Sources, Models & Key Literature

| Resource / Citation | Role | Source / Link |
| :--- | :--- | :--- |
| **SemEval-2027 Task 9** | Official task definition, training data, and unmodified scorers | [propaganda.math.unipd.it/semeval2027task9](https://propaganda.math.unipd.it/semeval2027task9/) |
| **mm-framing-baselines** | Source of Batch 1 (400 EN) image-text pairs | [github.com/YovaKem/mm-framing-baselines](https://github.com/YovaKem/mm-framing-baselines) |
| **Grounding-DINO Tiny** | Frame-neutral open-vocabulary region detector | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| **SigLIP Base (patch16-224)** | Unified 768-d Vision + Text backbone (`max_text_len=64`) | [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) |
| **Qwen2.5-VL 7B** | Zero-shot multimodal VLM baseline via Ollama (`temp=0`) | [ollama.com/library/qwen2.5vl](https://ollama.com/library/qwen2.5vl) |
| **Entman (1993)** | Framing salience theory (*Journal of Communication*) | — |
| **Rodriguez & Dimitrova (2011)** | Four levels of visual framing (*Journal of Visual Literacy*) | — |
| **Card et al. (ACL 2015)** | Media Frames Corpus (14+1 generic framing taxonomy) | [ACL Anthology](https://aclanthology.org/P15-2072/) |
| **Ridnik et al. (ICCV 2021)** | Asymmetric Loss (ASL) for sparse multi-label learning | [arXiv:2009.14119](https://arxiv.org/abs/2009.14119) |
| **Piskorski et al. (SemEval 2023)** | SemEval-2023 Task 3 (predecessor text framing task) | [ACL Anthology](https://aclanthology.org/2023.semeval-1.317/) |
| **Arora et al. (EMNLP 2025)** | *Multi-Modal Framing Analysis of News* | [doi:10.18653/v1/2025.emnlp-main.1606](https://doi.org/10.18653/v1/2025.emnlp-main.1606) |

---

## 2. System Architecture

Our system replaces heterogeneous 4-model assembly with a minimal **2-backbone + lightweight fusion** architecture optimized for Apple Silicon (MPS) and small-sample generalization:

```text
[Raw News Photo]  ---> Grounding-DINO Tiny (frame-neutral physical prompt, 93.3% real detections) ---> Top-3 Distinct Crops
[Full Photo + 3 Crops] ---> SigLIP Vision (google/siglip-base-patch16-224) ---> Visual Tokens (4, 768)
[Article Text (64 tok)] ---> SigLIP Text (shared 768-d space)              ---> Text Tokens  (64, 768)
                                                                                       │
                                                                                       ▼
[Text Queries] x [Visual Key/Value] ---> Cross-Modal Multi-Head Attention + Masked Pooling + MLP
                                                                                       │
                                                                                       ▼
                                                      14-Class Logits + Asymmetric Loss (ASL)
                                                                                       │
                                                                                       ▼
                                                      Dev Threshold τ* (all < τ* => "None")
```

---

## 3. Benchmark Results on Official 40-Image Test Split

All models are evaluated via `evaluation.run_eval` using the **unmodified official SemEval scorer** (`starter-kit/subtask2a_2b/scorer-subtask-2.py`, `0` mismatches) on the fixed **437 Train (400 Batch 1 + 37 Batch 2 EN) / 20 Dev / 40 Test** partition in `course_split.json` across 15 classes (14 frames + `"None"`). Trained models report `mean ± std` over 3 seeds (`[13, 42, 2026]`) with 2,000-resample article-level bootstrap 95% confidence intervals (source of truth: `results/subtask2_final.json`).

| Model / Setting | Dev Micro | Test Micro (Primary) | Test 95% CI | Test Macro | `None` F1 | Notes / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Most Frequent (`Political`)** | 0.238 | 0.295 | [0.169, 0.422] | 0.166 | 0.000 | Trivial majority baseline |
| **Image LR (SigLIP)** | 0.419 | 0.353 | [0.269, 0.444] | 0.281 | 0.000 | Frozen SigLIP + Logistic Regression |
| **Late Fusion 2a (LR)** | 0.444 | 0.367 | [0.248, 0.486] | 0.350 | 0.308 | Direct probability averaging baseline |
| **Qwen2.5-VL (Image-Only)** | 0.478 | 0.417 | [0.292, 0.553] | 0.205 | 0.000 | 7B VLM zero-shot baseline |
| **Qwen2.5-VL (+Title / +1500c / +Full)** | 0.458 / 0.440 / 0.480 | 0.370 / 0.333 / 0.340 | — | 0.235 / 0.229 / 0.313 | 0.000 | E12: VLM degraded by raw text |
| **Exp 1: SigLIP Text-Only (ASL)** | 0.397 | 0.303 ± 0.035 | [0.214, 0.393] | 0.261 | 0.108 | Unimodal text ablation |
| **Exp 2: SigLIP Vision Global (ASL)** | 0.546 | 0.389 ± 0.070 | [0.290, 0.497] | 0.317 | 0.258 | Global visual token ablation |
| **Exp 3: SigLIP + GDINO Regions (ASL)** | 0.527 | **0.446 ± 0.041** | [0.322, 0.571] | 0.316 | 0.229 | Top Test Micro F1 (+0.057 over Global) |
| **Exp 4: Global + Text Cross-Attn (ASL)** | **0.565** | **0.430 ± 0.015** | [0.321, 0.541] | **0.341** | 0.236 | **Best on Dev; sig. beats `Most Frequent`** |
| **Exp 5: SigLIP Late Fusion (ASL)** | 0.528 | 0.393 ± 0.034 | [0.289, 0.505] | 0.287 | 0.130 | Unified SigLIP probability averaging |
| **Exp 6: Full GDINO Cross-Modal (BCE)** | 0.514 | 0.401 ± 0.012 | [0.287, 0.518] | 0.316 | 0.228 | Full architecture with standard BCE |
| **Exp 7: Full GDINO Cross-Modal (ASL)** | 0.513 | **0.422 ± 0.033** | [0.313, 0.531] | 0.310 | 0.147 | Full proposed architecture (+0.021 vs. BCE) |

---

## 4. Repository Layout

```text
task9/
├── assets/                       # README visual assets (official Task 9 illustration)
├── src/
│   ├── dataset.py                # FramingDataset ingesting course_split.json
│   ├── entity_extractor.py       # Frame-neutral Grounding-DINO detector + deterministic fallback
│   ├── feature_extractor.py      # Unified SigLIP Vision + Text encoder (float32 safe)
│   ├── cached_dataset.py         # Fast DataLoader loading pre-extracted .pt tensors
│   ├── explain.py                # Cross-attention extraction & heatmap visualization
│   ├── metrics.py                # 15-class evaluation, threshold search & TSV export
│   ├── losses/
│   │   └── asymmetric_loss.py    # ASL margin clipping & focal modulation (Ridnik et al.)
│   └── models/
│       ├── fusion_model.py       # CrossModalFramingModel (Text Query, Visual Key/Val)
│       └── baselines.py          # TextOnly, VisionGlobal, VisionRegion, LateFusion
├── scripts/
│   ├── 01_extract_regions.py     # Offline Grounding-DINO crop extraction (cache/crops_gdino)
│   ├── 02_extract_features.py    # Offline unified SigLIP feature caching (cache/features_gdino_siglip)
│   ├── 03_train.py               # Single-run training loop with Dev threshold search
│   ├── 04_run_ablations.py       # 7-experiment 3-seed ablation runner & TSV/probs exporter
│   └── 05_visualize_attention.py # Test-set attention overlay heatmap generator
├── baselines/                    # En-Shuo's baselines (LR, late fusion, Qwen2.5-VL, Subtask 1)
├── evaluation/                   # Shared evaluation layer calling the unmodified official scorer
├── predictions/subtask2/         # Official-format predictions: <model>/seed<k>/{dev,test}.{tsv,probs.json}
├── results/                      # Single source of truth: subtask2_final.json, c1/c2 analyses
├── reports/
│   ├── ablation_results_gdino_siglip.md  # Grounding-DINO + unified SigLIP ablation summary
│   └── figures_gdino_siglip/     # Test-set Grounding-DINO + SigLIP attention heatmaps
├── docs/                         # Design specs, implementation plans, pre-registration, notes
├── tests/                        # Unit & integration test suite
├── course_split.json             # Fixed-seed course partition (437 Train, 20 Dev, 40 Test)
├── splits.py                     # Split creation logic
└── requirements.txt              # Dependencies
```

Data is not tracked: the root pipeline reads `data_official/` and `starter-kit/`, and `baselines/` reads `baselines/data/`.

---

## 5. Quickstart Guide

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Run Test Suite
```bash
venv/bin/python -m pytest tests -q --deselect tests/test_losses_and_metrics.py::test_evaluate_predictions_matches_official_scorer
```

### Reproduce End-to-End Pipeline
```bash
# 1. Extract frame-neutral Grounding-DINO region crops
venv/bin/python scripts/01_extract_regions.py --output_dir cache/crops_gdino --top_k 3 --conf_thresh 0.25

# 2. Extract & cache unified SigLIP Vision + Text features
venv/bin/python scripts/02_extract_features.py --crop_dir cache/crops_gdino --output_dir cache/features_gdino_siglip

# 3. Run 7-experiment 3-seed ablation suite & export predictions
venv/bin/python scripts/04_run_ablations.py --feature_dir cache/features_gdino_siglip --epochs 15

# 4. Run unified official scorer & bootstrap CIs
venv/bin/python -m evaluation.run_eval --pred_root predictions/subtask2 --scorer starter-kit/subtask2a_2b/scorer-subtask-2.py --out results/subtask2_final.json

# 5. Generate test-set attention interpretability figures
venv/bin/python scripts/05_visualize_attention.py --checkpoint checkpoints/ablations_gdino_siglip/sig_e7_fusion_gdino_asl/best_fusion_asl_seed42.pt --feature_dir cache/features_gdino_siglip --crop_dir cache/crops_gdino --output_dir reports/figures_gdino_siglip --split test --num_samples 5
```
