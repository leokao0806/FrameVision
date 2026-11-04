# Design Specification: Region-Aware Cross-Modal Framing Architecture (FrameVision)
**Project**: SemEval-2027 Task 9: Analysis of Multimodal Framing in News (Subtask 2)  
**Team**: FrameVision (Leo Kao - Lead Multimodal Architect)  
**Course**: CS 5134/6034 Natural Language Processing  
**Date**: 2026-11-04  
**Status**: Final Approved (Aligned with Team Intermediate Report)

---

## 1. Intent & The Why

### Background & Scientific Problem
In news journalism, photojournalism and text narratives interact to shape audience perception. While article text provides explicit exposition, images introduce subtle emotional or ideological frames that can **reinforce (align with)**, **complement**, or **contradict** the textual narrative.

Standard multimodal classification architectures typically apply global average pooling to visual representations. However, in news framing, critical narrative cues are localized and fine-grained (e.g., small protest signs, facial expressions, law enforcement insignia, or armbands). Global pooling dilutes these critical visual features against dominant background clutter.

### Core Research Hypothesis
Decomposing news photographs into localized semantic entities via **Segment Anything (SAM) / Salient Object Grounding**, combined with pre-trained foundation representations (**SigLIP** for vision, **DeBERTa-v3** for text), and fusing them through a **Cross-Modal Attention Network** (where text actively queries both global and local visual tokens), will significantly improve framing discriminability and cross-modal alignment/contradiction resolution. Furthermore, optimizing with **Multi-Label Asymmetric Loss (ASL)** directly addresses the extreme class imbalance and sparsity across the 14 generic framing categories.

### Need for Full Rebuild
The initial prototype in `task9` served only as a proof-of-concept pipeline:
- Entity extraction used a 3x3 grid edge-density filter instead of genuine semantic segmentation/grounding.
- The model defaulted to a hash-based tokenizer with BiGRU and a 4-layer CNN instead of pre-trained foundation backbones (SigLIP / DeBERTa).
- The pipeline lacked systematic ablation study mechanisms, threshold calibration, and interpretability heatmaps (Grad-CAM).
- It used an obsolete 100-sample random test split that has been officially withdrawn in the team's Intermediate Report.

This design completely replaces that prototype with a publication-grade research framework strictly synchronized with the team's Intermediate Report contract.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
1. **Authentic Semantic Entity Extraction**: Implement true salient entity segmentation/grounding (MobileSAM / FastSAM / OWL-ViT) to extract Top-K informative visual regions per image with spatial bounding boxes.
2. **Foundation Model Representations**: Integrate pre-trained `google/siglip-base-patch16-224` (vision) and `microsoft/deberta-v3-base` (text, using `float32` to prevent Adam NaN instabilities) to establish rigorous semantic embeddings.
3. **Region-Aware Cross-Modal Attention**: Build a modular PyTorch Transformer fusion layer where textual tokens query global visual context and local entity tokens to resolve cross-modal framing.
4. **Asymmetric Loss (ASL) Optimization**: Implement ASL to dynamically downweight easy negative samples and prevent majority classes from dominating gradient updates.
5. **Systematic Ablation Framework**: Provide a one-command harness to run controlled ablation experiments:
   - Text-Only vs. Vision-Only (Global) vs. Vision-Only (Global + Region) vs. Full Cross-Modal Fusion.
   - Standard BCE Loss vs. Asymmetric Loss (ASL).
6. **Official Team Benchmark Compliance**: Ingest `course_split.json` / `splits.py` directly (40 Test images, 20 Dev images, 647 Train images) and evaluate with `starter-kit/subtask2a_2b/scorer-subtask-2.py` (Micro-F1 primary, Macro-F1 secondary, `"None"` as 15th label).
7. **Beat Established Baselines**: Target outperforming the team's current best marks on the 40-test set:
   - Beat `Late fusion` Macro-F1 (0.350) and Micro-F1 (0.367).
   - Beat `Qwen2.5-VL` Micro-F1 (0.417) and resolve its text-incongruence degradation.
   - Resolve zero-predictions on rare frames (`None` and `Morality`).
8. **Interpretability & Visualization**: Implement cross-attention map extraction to generate overlay heatmaps for qualitative case studies in the final report.
9. **Apple Silicon (M1) Optimization**: Enable a two-stage feature caching mechanism so full training and hyperparameter search run locally on Mac M1 (MPS) in seconds per run.

### Non-Goals
1. **End-to-End Backbone Pre-training**: We will not pre-train SigLIP or DeBERTa from scratch; 647 samples are insufficient and would guarantee catastrophic overfitting.
2. **Subtask 1 Implementation**: Paragraph-level text framing (Subtask 1) is a separate track handled by team member En-Shuo Zhang; this codebase focuses strictly on Subtask 2 (image framing with optional text guidance).
3. **Ad-Hoc Data Splits**: We will NOT generate independent data splits; all evaluation is anchored to `course_split.json`.

### Anti-Goals
1. **No Toy Placeholders**: No hash tokenizers, toy CNNs, or heuristic edge filters disguised as SAM.
2. **No Data Leakage**: Threshold calibration and checkpoint selection occur strictly on the 20-sample Dev set (`course_split.json["dev"]`), never touching the 40-sample Test set.

---

## 3. Constraints & Operating Environment

- **Hardware**: Mac M1 (Apple Silicon, MPS acceleration). Memory and compute must be respected.
- **Data Partitions (`course_split.json`)**:
  - **Train**: 647 images (400 first-batch EN, 37 second-batch EN, plus multilingual 100 BG, 105 PT, 5 RU).
  - **Course Dev**: 20 images (derived from second-batch English articles).
  - **Course Test**: 40 images (isolated for team benchmark, includes 5 `"None"` images).
  - Linked articles and trial articles (`EN_TRAIL_1` through `5`, including `EN_TRAIL_2` with the sole `Legality` label) are guaranteed in Train.
- **Evaluation Metric**: Micro-averaged F1 as Primary metric (following official SemEval task contract), Macro-averaged F1 as complementary metric across 15 classes (14 frames + `"None"`).

---

## 4. Evaluated Approaches & Selected Path

Three architectural approaches were considered:
- **Approach A: Two-Stage Modular Feature & Region Extraction Pipeline** (Selected):
  - Phase 1: Offline extraction of Top-K semantic entity crops and pre-computed SigLIP/DeBERTa embeddings.
  - Phase 2: Lightweight trainable Cross-Modal Attention Fusion Network + ASL.
  - *Trade-offs*: Fast local execution on M1 (seconds per epoch), zero OOM risk, robust against small-sample overfitting, easy ablation runner. Backbones are frozen during fusion training.
- **Approach B: End-to-End Fine-Tuning with LoRA**:
  - Full end-to-end gradient backprop through LoRA adapters in SigLIP and DeBERTa.
  - *Trade-offs*: Highly computationally intensive on M1, batch size constrained to 1-2, cannot cleanly incorporate SAM mask extraction in the gradient graph, high risk of overfitting on small data.
- **Approach C: Multimodal LLM Prompted Feature Extraction**:
  - Using an open-source VLM (Qwen2-VL) to produce textual framing descriptions of images, then encoding text.
  - *Trade-offs*: Diverges from Leo's role (custom deep learning / SAM vision architect) into prompt engineering.

**Decision**: Approach A is approved.

---

## 5. Detailed System Architecture

### Component 1: Semantic Entity Extraction (`entity_extractor.py`)
- **Backbone**: FastSAM / MobileSAM or OWL-ViT (zero-shot open-vocabulary object detector) with general news entity queries (`person, crowd, protest sign, banner, face, uniform, weapon, building, vehicle, emblem`).
- **Processing**:
  - Takes raw input images from `data_official`.
  - Extracts the top $K$ ($K=3$ default) most salient/confident candidate bounding boxes.
  - Crops and saves them into `cache/crops/{image_name}_crop_{k}.jpg` along with a coordinate metadata JSON `cache/crops/metadata.json`.
  - Fallback: If no confident entity is detected, extracts center and multi-scale crops to guarantee $K$ regions.

### Component 2: Feature Caching Pipeline (`feature_extractor.py`)
- **Vision Features**:
  - Global image: passed through `google/siglip-base-patch16-224` vision encoder $\rightarrow$ global visual embedding $v_{\text{global}} \in \mathbb{R}^{D}$ ($D=768$).
  - Top-$K$ regional crops: each passed through SigLIP $\rightarrow$ region tokens $v_{\text{reg}, 1}, \dots, v_{\text{reg}, K} \in \mathbb{R}^{D}$.
  - Combined visual sequence: $V = [v_{\text{global}}; v_{\text{reg}, 1}; \dots; v_{\text{reg}, K}] \in \mathbb{R}^{(1+K) \times D}$.
- **Text Features**:
  - Raw article text (headline concatenated with body text) tokenized and encoded via `microsoft/deberta-v3-base` (in `float32`).
  - Contextual token sequence: $T \in \mathbb{R}^{L_t \times D_{\text{text}}}$ ($D_{\text{text}}=768$, max length $L_t=256$) along with its boolean attention mask.
- **Disk Caching**:
  - Pre-encoded features saved to `cache/features/{id}.pt`.
  - Ensures training iterations only train the fusion layer without repeatedly executing forward passes through massive backbones.

### Component 3: Region-Aware Cross-Modal Attention Model (`models/fusion_model.py`)
- **Projections**:
  - Linear projection $W_t: \mathbb{R}^{D_{\text{text}}} \rightarrow \mathbb{R}^{d_{\text{embed}}}$ ($d_{\text{embed}}=512$).
  - Linear projection $W_v: \mathbb{R}^{D} \rightarrow \mathbb{R}^{d_{\text{embed}}}$.
- **Cross-Attention Mechanism**:
  - Query: Text tokens $Q = T W_t \in \mathbb{R}^{L_t \times d_{\text{embed}}}$.
  - Key / Value: Concatenated visual tokens $K = V = V W_v \in \mathbb{R}^{(1+K) \times d_{\text{embed}}}$.
  - Attention:
    $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{embed}}}}\right) V$$
  - Includes LayerNorm, Multi-Head Attention (heads=8), Dropout (0.1), and a 2-layer Feed-Forward Network with GELU and residual connection.
- **Aggregation & Head**:
  - Masked mean-pooling over the attended text queries to obtain unified multimodal representation $h_{\text{fused}} \in \mathbb{R}^{d_{\text{embed}}}$.
  - Multi-label classification head: Linear($d_{\text{embed}} \rightarrow d_{\text{embed}} / 2$) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear($\rightarrow 14$ classes).

### Component 4: Asymmetric Loss (`losses/asymmetric_loss.py`)
- Formulated as:
  $$L = - \sum_{c=1}^{C} \left[ y_c (1-p_c)^{\gamma_+} \log(p_c) + (1-y_c) (p_{m,c})^{\gamma_-} \log(1 - p_{m,c}) \right]$$
  where $p_{m,c} = \max(p_c - m, 0)$ is the probability shifting margin.
- Hyperparameters: $\gamma_- = 4.0$, $\gamma_+ = 1.0$, margin $m = 0.05$.
- Suppresses gradients from overwhelming easy negative frames while focusing attention on rare positive frames.

### Component 5: Experiment Harness & Ablations (`experiments/`)
- **Partition Integration**:
  - Ingests `course_split.json`: 647 Train images, 20 Dev images, 40 Test images.
- **Threshold Calibration**:
  - Evaluates decision thresholds $\tau \in [0.1, 0.9]$ on the 20-image Dev split.
  - Applies official SemEval scoring logic (`starter-kit/subtask2a_2b/scorer-subtask-2.py`) mapping zero-predictions to `"None"` to select $\tau^*$ that maximizes Micro-F1 / Macro-F1.
- **Ablation Benchmark Suite (`04_run_ablations.py`)**:
  - Experiment 1: Text-Only (DeBERTa pooled + ASL).
  - Experiment 2: Vision-Only Global (SigLIP Global pooled + ASL).
  - Experiment 3: Vision-Only Region Attention (Transformer Self-Attention across $(1+K)$ visual tokens + ASL).
  - Experiment 4: Multimodal Cross-Attention with BCE Loss.
  - Experiment 5: Multimodal Cross-Attention with ASL (Proposed Full Model).
  *(Note: Experiments 1, 2, 3, and 5 all share ASL to strictly isolate modality contributions, while Experiment 4 vs. 5 isolates the loss function contribution).*
- Outputs summary Markdown table of Micro-F1, Macro-F1, and Hamming Loss directly comparable with Table 1 of the Intermediate Report.

### Component 6: Interpretability & Heatmaps (`visualization/explain.py`)
- Extracts cross-attention weights between text tokens and visual tokens (global and entity crops).
- Generates high-resolution visualization figures:
  - Input image with overlaid bounding boxes and attention weights.
  - Top attended keywords from news text.
  - Explanation figures saved to `reports/figures/` for direct inclusion in the final report and presentation slides.

---

## 6. Project Directory Layout

```text
task9/
├── docs/
│   └── superpowers/specs/2026-11-04-multimodal-framing-research-design.md
├── course_split.json                      # Official team split contract (40 Test, 20 Dev, 647 Train)
├── splits.py                              # Official split generator script
├── data_official/                         # Official raw training datasets
│   ├── initial-training-data/
│   ├── training-second-batch-oct/
│   └── labels.txt
├── cache/
│   ├── crops/                             # Top-K extracted entity crops & bounding boxes
│   │   └── metadata.json
│   └── features/                          # Cached SigLIP & DeBERTa embeddings (.pt)
├── src/
│   ├── __init__.py
│   ├── dataset.py                         # Dataset loader for raw pairs & cached features
│   ├── entity_extractor.py                # Salient region extractor (FastSAM/OWL-ViT)
│   ├── feature_extractor.py               # Pre-computes SigLIP & DeBERTa embeddings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion_model.py                # Cross-Attention Network
│   │   └── baselines.py                   # Unimodal text & vision baselines
│   ├── losses/
│   │   └── asymmetric_loss.py             # ASL implementation
│   ├── metrics.py                         # Scorer wrapper & threshold optimizer
│   └── explain.py                         # Cross-attention visualizer
├── scripts/
│   ├── 01_extract_regions.py              # Run offline entity extraction
│   ├── 02_extract_features.py             # Run offline feature caching
│   ├── 03_train.py                        # Train fusion model or baselines
│   ├── 04_run_ablations.py                # Automated ablation benchmark suite
│   └── 05_visualize_attention.py          # Generate paper figures
├── starter-kit/                           # Official scorer
└── checkpoints/                           # Model weights & evaluation logs
```

---

## 7. What is Left to the Builder

The builder has discretion over:
- Concrete default learning rate (e.g. $1 \times 10^{-4}$ to $3 \times 10^{-4}$) and batch size (32 or 64 on cached features).
- Selection of specific lightweight salient detector package (e.g. `ultralytics` FastSAM or Hugging Face `google/owlvit-base-patch32`).
- Detailed progress bar formatting and logging messages.
- Specific color maps for attention heatmap rendering.

│   └── labels.txt
├── cache/
│   ├── crops/                             # Top-K extracted entity crops & bounding boxes
│   │   └── metadata.json
│   └── features/                          # Cached SigLIP & DeBERTa embeddings (.pt)
├── src/
│   ├── __init__.py
│   ├── dataset.py                         # Dataset loader for raw pairs & cached features
│   ├── entity_extractor.py                # Salient region extractor (FastSAM/OWL-ViT)
│   ├── feature_extractor.py               # Pre-computes SigLIP & DeBERTa embeddings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion_model.py                # Cross-Attention Network
│   │   └── baselines.py                   # Unimodal text & vision baselines
│   ├── losses/
│   │   └── asymmetric_loss.py             # ASL implementation
│   ├── metrics.py                         # Scorer wrapper & threshold optimizer
│   └── explain.py                         # Cross-attention / Grad-CAM visualizer
├── scripts/
│   ├── 01_extract_regions.py              # Run offline entity extraction
│   ├── 02_extract_features.py             # Run offline feature caching
│   ├── 03_train.py                        # Train fusion model or baselines
│   ├── 04_run_ablations.py                # Automated ablation benchmark suite
│   └── 05_visualize_attention.py          # Generate paper figures
├── starter-kit/                           # Official scorer
└── checkpoints/                           # Model weights & evaluation logs
```

---

## 7. What is Left to the Builder

The builder has discretion over:
- Concrete default learning rate (e.g. $1 \times 10^{-4}$ to $3 \times 10^{-4}$) and batch size (32 or 64 on cached features).
- Selection of specific lightweight salient detector package (e.g. `ultralytics` FastSAM or Hugging Face `google/owlvit-base-patch32`).
- Detailed progress bar formatting and logging messages.
- Specific color maps for attention heatmap rendering.
