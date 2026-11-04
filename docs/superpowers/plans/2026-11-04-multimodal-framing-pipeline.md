# Region-Aware Cross-Modal Framing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-grade multimodal news framing analysis system for SemEval-2027 Task 9 Subtask 2 that ingests the official course partition (`course_split.json`), utilizes semantic entity extraction (FastSAM/OWL-ViT), foundation model representations (SigLIP and DeBERTa-v3 in float32), region-aware cross-modal attention, and Asymmetric Loss (ASL), outperforming the team's Intermediate Report baselines.

**Architecture:** A two-stage modular architecture optimized for Mac M1 (MPS). Stage 1 performs offline salient entity extraction (Top-$K$ crops) and foundation feature caching (SigLIP for vision, DeBERTa-v3 for text) to disk. Stage 2 trains a lightweight PyTorch Cross-Modal Attention Network where text tokens actively query visual tokens (global + entity crops), optimized via Asymmetric Loss and calibrated on the 20-image Dev split to evaluate on the 40-image Test split.

**Tech Stack:** Python 3.9+, PyTorch 2.0+ (with MPS acceleration), Hugging Face Transformers (`transformers`), PIL, scikit-learn, numpy, pytest.

**Spec:** [`/Users/leokao/Documents/Python/task9/docs/superpowers/specs/2026-11-04-multimodal-framing-research-design.md`](file:///Users/leokao/Documents/Python/task9/docs/superpowers/specs/2026-11-04-multimodal-framing-research-design.md)

## Global Constraints

- Hardware target: Mac M1 (Apple Silicon, MPS acceleration). Memory efficiency and disk feature caching are required.
- Official data split: Must strictly ingest `course_split.json` (40 Test images, 20 Dev images, 647 Train images).
- Target metrics: Micro-averaged F1 as Primary metric, Macro-averaged F1 as secondary diagnostic metric across 15 classes (14 generic frames + `"None"`).
- DeBERTa-v3 stability: DeBERTa text representations must be computed in `torch.float32` to avoid Adam NaN optimization failures.
- Submission contract: Output predictions must format as TSV: `image_name \t frame_1 \t frame_2 ...` without header, evaluated strictly via `starter-kit/subtask2a_2b/scorer-subtask-2.py`.

## Review Focus

1. **Zero-prediction `"None"` mapping**: When all 14 class probabilities are below threshold $\tau$, the output row must emit `image_name\n`, ensuring the official scorer maps it to the 15th `"None"` class rather than crashing.
2. **Missing entity fallback**: When FastSAM/OWL-ViT detects zero salient entities in low-contrast news photos, the extractor must automatically fall back to deterministic multi-scale center crops to guarantee exactly $K$ visual tokens.
3. **Sole rare-class supervision**: `EN_TRAIL_2` (containing the dataset's only positive instance of `Legality_Constitutionality_and_jurisprudence`) must reside strictly in the training partition, verified by assertion.
4. **Attention mask padding safety**: Text mean-pooling after cross-attention must respect the boolean attention mask so that padded tokens do not corrupt the fused embedding vector.
5. **Exact scorer alignment**: Metric computation in `src/metrics.py` must replicate the exact micro-F1 and macro-F1 numbers produced by `starter-kit/subtask2a_2b/scorer-subtask-2.py`.

---

### Task 1: Course Split Ingestion & Dataset Adapter

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `course_split.json`, `data_official/labels.txt`, `data_official/initial-training-data/subtask2/`, `data_official/training-second-batch-oct/EN/`
- Produces: `FramingDataset(split: str, data_root: str, split_file: str) -> Dataset` yielding dict `{id: str, image_name: str, image_path: str, text: str, labels: List[str], target: torch.Tensor}`

- [x] **Step 1: Write test for dataset split loading and label integrity**

```python
# tests/test_dataset.py
import os
import pytest
import torch
from src.dataset import FramingDataset

def test_course_split_partitions():
    ds_test = FramingDataset(split="test", data_root="data_official", split_file="course_split.json")
    ds_dev = FramingDataset(split="dev", data_root="data_official", split_file="course_split.json")
    ds_train = FramingDataset(split="train", data_root="data_official", split_file="course_split.json")
    
    assert len(ds_test) == 40
    assert len(ds_dev) == 20
    assert len(ds_train) >= 437  # 400 Batch 1 + 37 Batch 2 EN
    
    # Verify sole Legality instance is in train
    sample_trail2 = [s for s in ds_train.samples if "EN_TRAIL_2" in s["id"]]
    assert len(sample_trail2) == 1
    assert "Legality_Constitutionality_and_jurisprudence" in sample_trail2[0]["labels"]

def test_dataset_sample_item():
    ds_test = FramingDataset(split="test", data_root="data_official", split_file="course_split.json")
    item = ds_test[0]
    assert "id" in item
    assert "image_path" in item and os.path.exists(item["image_path"])
    assert "text" in item and len(item["text"]) > 0
    assert "target" in item and isinstance(item["target"], torch.Tensor)
    assert item["target"].shape == (14,)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`  
Expected: FAIL with ModuleNotFoundError or dataset not defined.

- [x] **Step 3: Implement `FramingDataset` in `src/dataset.py`**

Parse `course_split.json`, join with Batch 1 (`EN_doc_{N}_image.jpg` + `EN_M_TRAIN_doc_{N}.txt`) and Batch 2 (`EN_M_TRAIN_{N}.*` and `EN_TRAIL_*`), load 14 labels from `data_official/labels.txt`, and generate binary multi-hot targets.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): implement course split dataset loader with label mapping"
```

---

### Task 2: Asymmetric Loss (ASL) & Official Scorer Metrics Engine

**Files:**
- Create: `src/losses/asymmetric_loss.py`
- Create: `src/metrics.py`
- Test: `tests/test_losses_and_metrics.py`

**Interfaces:**
- Consumes: Logits tensor `(B, 14)`, targets tensor `(B, 14)`, prediction files, gold files
- Produces: `AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05) -> nn.Module`, `evaluate_predictions(pred_path: str, gold_path: str, frame_file: str) -> Tuple[float, float]` (micro-F1, macro-F1), `optimize_threshold(val_probs: np.ndarray, val_targets: np.ndarray, label_names: List[str]) -> Tuple[float, float, float]`

- [x] **Step 1: Write test for ASL calculation and official scorer evaluation wrapper**

```python
# tests/test_losses_and_metrics.py
import os
import torch
import numpy as np
import pytest
from src.losses.asymmetric_loss import AsymmetricLoss
from src.metrics import compute_metrics, optimize_threshold, evaluate_predictions

def test_asymmetric_loss_forward():
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    logits = torch.randn(4, 14, requires_grad=True)
    targets = torch.randint(0, 2, (4, 14)).float()
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None

def test_metrics_empty_prediction_none_mapping():
    # If all classes 0, should map to None and evaluate cleanly
    targets = np.zeros((2, 14))
    probs = np.zeros((2, 14))
    metrics = compute_metrics(targets, probs, threshold=0.5)
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_losses_and_metrics.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `AsymmetricLoss` in `src/losses/asymmetric_loss.py` and `compute_metrics` / `optimize_threshold` in `src/metrics.py`**

Implement ASL with probability shifting margin. In `src/metrics.py`, wrap the official `starter-kit/subtask2a_2b/scorer-subtask-2.py` logic, including mapping all-zero instances to `"None"` across 15 classes.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_losses_and_metrics.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/losses/asymmetric_loss.py src/metrics.py tests/test_losses_and_metrics.py
git commit -m "feat(loss,metrics): implement Asymmetric Loss and official 15-class evaluation harness"
```

---

### Task 3: Salient Semantic Entity Extractor & Bounding Box Pipeline

**Files:**
- Create: `src/entity_extractor.py`
- Create: `scripts/01_extract_regions.py`
- Test: `tests/test_entity_extractor.py`

**Interfaces:**
- Consumes: PIL Image or image file path
- Produces: `extract_salient_regions(img: Image.Image, top_k: int = 3) -> Tuple[List[Image.Image], List[List[int]]]` returning $K$ PIL crops and their `[x1, y1, x2, y2]` bounding box coordinates.

- [x] **Step 1: Write test for entity extraction and deterministic fallback**

```python
# tests/test_entity_extractor.py
import pytest
from PIL import Image
from src.entity_extractor import extract_salient_regions

def test_extract_salient_regions_returns_k_crops():
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    crops, boxes = extract_salient_regions(img, top_k=3)
    assert len(crops) == 3
    assert len(boxes) == 3
    for crop in crops:
        assert isinstance(crop, Image.Image)
        assert crop.width > 0 and crop.height > 0
    for box in boxes:
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_entity_extractor.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `src/entity_extractor.py` and batch script `scripts/01_extract_regions.py`**

Implement salient region detector (using MobileSAM / OWL-ViT / FastSAM with news queries, falling back to multi-scale focal crops when no confident foreground entity exists). Save crops into `cache/crops/{img_name}_crop_{k}.jpg` and metadata in `cache/crops/metadata.json`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_entity_extractor.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/entity_extractor.py scripts/01_extract_regions.py tests/test_entity_extractor.py
git commit -m "feat(crops): implement salient entity extractor with deterministic crop fallback"
```

---

### Task 4: Foundation Feature Extraction & Disk Caching Pipeline

**Files:**
- Create: `src/feature_extractor.py`
- Create: `scripts/02_extract_features.py`
- Test: `tests/test_feature_extractor.py`

**Interfaces:**
- Consumes: Raw text, full image, Top-$K$ region crops
- Produces: `cache/features/{doc_id}.pt` containing dict `{visual_tokens: (1+K, 768), text_tokens: (Lt, 768), text_mask: (Lt,), target: (14,), label_names: List[str]}`

- [x] **Step 1: Write test for feature extraction tensor shapes and float32 precision**

```python
# tests/test_feature_extractor.py
import torch
import pytest
from PIL import Image
from src.feature_extractor import FeatureExtractor

def test_feature_extractor_shapes():
    extractor = FeatureExtractor(
        vision_model_name="google/siglip-base-patch16-224",
        text_model_name="microsoft/deberta-v3-base",
        max_text_len=256,
        device="cpu"
    )
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    crops = [Image.new("RGB", (224, 224), color=(50, 50, 50)) for _ in range(3)]
    text = "City Council Approves Fiscal Overhaul"
    
    feats = extractor.encode_sample(text=text, full_image=img, crop_images=crops)
    assert feats["visual_tokens"].shape == (4, 768)  # 1 global + 3 crops
    assert feats["text_tokens"].shape == (256, 768)
    assert feats["text_tokens"].dtype == torch.float32  # verified float32
    assert feats["text_mask"].shape == (256,)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_extractor.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `FeatureExtractor` in `src/feature_extractor.py` and `scripts/02_extract_features.py`**

Instantiate `AutoModel` for SigLIP and DeBERTa-v3 in `torch.float32`. Encode global image and crops into visual tokens, text into contextual token sequence with attention mask. Pre-encode and save all samples in `cache/features/{doc_id}.pt`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_feature_extractor.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/feature_extractor.py scripts/02_extract_features.py tests/test_feature_extractor.py
git commit -m "feat(features): implement SigLIP and DeBERTa feature extractor and disk caching"
```

---

### Task 5: Region-Aware Cross-Modal Attention Model & Baselines

**Files:**
- Create: `src/models/fusion_model.py`
- Create: `src/models/baselines.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `visual_tokens: (B, 1+K, 768)`, `text_tokens: (B, Lt, 768)`, `text_mask: (B, Lt)`
- Produces: `CrossModalFramingModel(embed_dim=512, num_heads=8, num_classes=14) -> (logits: (B, 14), attn_weights: (B, Lt, 1+K))` and unimodal models (`TextOnlyModel`, `VisionGlobalModel`, `VisionRegionModel`).

- [x] **Step 1: Write test for forward pass, gradient flow, and attention weight shapes**

```python
# tests/test_models.py
import torch
import pytest
from src.models.fusion_model import CrossModalFramingModel
from src.models.baselines import TextOnlyModel, VisionRegionModel

def test_cross_modal_fusion_forward():
    model = CrossModalFramingModel(embed_dim=512, num_heads=8, num_classes=14)
    B, Lt, K_plus_1, D = 2, 256, 4, 768
    v_tokens = torch.randn(B, K_plus_1, D)
    t_tokens = torch.randn(B, Lt, D)
    t_mask = torch.ones(B, Lt, dtype=torch.bool)
    t_mask[:, 100:] = False  # padding
    
    logits, attn_weights = model(v_tokens, t_tokens, t_mask)
    assert logits.shape == (B, 14)
    assert attn_weights.shape == (B, Lt, K_plus_1)
    
    # Backprop check
    loss = logits.sum()
    loss.backward()
    assert model.proj_text.weight.grad is not None

def test_vision_region_baseline_forward():
    model = VisionRegionModel(embed_dim=512, num_heads=8, num_classes=14)
    v_tokens = torch.randn(2, 4, 768)
    logits = model(v_tokens)
    assert logits.shape == (2, 14)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `CrossModalFramingModel` and baseline models**

Implement linear projections, PyTorch `MultiheadAttention`, LayerNorm, GELU feed-forward network, masked mean-pooling across active text queries, and 14-class classification head.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/models/fusion_model.py src/models/baselines.py tests/test_models.py
git commit -m "feat(models): implement CrossModalFramingModel and unimodal ablation baselines"
```

---

### Task 6: Training Engine & Dev Checkpoint Management

**Files:**
- Create: `scripts/03_train.py`
- Test: `tests/test_train_pipeline.py`

**Interfaces:**
- Consumes: Cached `.pt` feature files, `course_split.json`
- Produces: Model checkpoint in `checkpoints/best_model.pt`, predictions TSV `checkpoints/predictions_test_subtask2.tsv`, logged micro-F1/macro-F1 metrics.

- [x] **Step 1: Write test for training pipeline execution and TSV generation**

```python
# tests/test_train_pipeline.py
import os
import torch
import pytest
from src.models.fusion_model import CrossModalFramingModel
from src.losses.asymmetric_loss import AsymmetricLoss
from src.metrics import evaluate_predictions

def test_checkpoint_and_tsv_export(tmp_path):
    tsv_out = tmp_path / "test_preds.tsv"
    gold_out = tmp_path / "gold.tsv"
    
    with open(gold_out, "w") as f:
        f.write("img1.jpg\tPolitical\nimg2.jpg\n")
    with open(tsv_out, "w") as f:
        f.write("img1.jpg\tPolitical\nimg2.jpg\n")
        
    macro, micro = evaluate_predictions(str(tsv_out), str(gold_out), "data_official/labels.txt")
    assert micro == 1.0
    assert macro == 1.0
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_pipeline.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `scripts/03_train.py`**

Load cached features partitioned by `course_split.json`. Train `CrossModalFramingModel` using AdamW and CosineAnnealingLR. Evaluate on Dev (20 samples) each epoch, optimize threshold $\tau^*$, and checkpoint the model with top Micro-F1/Macro-F1. Generate 40-test predictions TSV.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_pipeline.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add scripts/03_train.py tests/test_train_pipeline.py
git commit -m "feat(train): implement fast training loop on cached features with Dev checkpointing"
```

---

### Task 7: Automated Ablation Suite & Report Benchmark Generation

**Files:**
- Create: `scripts/04_run_ablations.py`
- Test: `tests/test_ablations.py`

**Interfaces:**
- Consumes: Cached features and training pipeline
- Produces: Markdown comparison table (`reports/ablation_results.md`) covering Experiments 1 through 5, juxtaposed directly against the Intermediate Report Table 1 baselines.

- [x] **Step 1: Write test for ablation runner result formatting**

```python
# tests/test_ablations.py
import pytest
from scripts.04_run_ablations import format_ablation_markdown_table

def test_ablation_table_formatting():
    results = [
        {"name": "Exp 1: Text-Only", "micro_f1": 0.25, "macro_f1": 0.20, "hamming": 0.12},
        {"name": "Exp 5: Full Cross-Modal (ASL)", "micro_f1": 0.45, "macro_f1": 0.38, "hamming": 0.08},
    ]
    md = format_ablation_markdown_table(results)
    assert "| Model / Setting | Micro F1 | Macro F1 | Hamming Loss |" in md
    assert "Exp 5: Full Cross-Modal (ASL)" in md
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ablations.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `scripts/04_run_ablations.py`**

Execute:
1. Exp 1: Text-Only (DeBERTa-v3 + ASL)
2. Exp 2: Vision-Only Global (SigLIP Global + ASL)
3. Exp 3: Vision-Only Region Attention (SigLIP Global + $K$ Regions + ASL)
4. Exp 4: Multimodal Cross-Attention with BCE Loss
5. Exp 5: Multimodal Cross-Attention with ASL (Proposed Full Model)
Write summary results to `reports/ablation_results.md`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ablations.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add scripts/04_run_ablations.py tests/test_ablations.py
git commit -m "feat(ablations): implement automated ablation benchmark suite"
```

---

### Task 8: Interpretability & Cross-Attention Visualization

**Files:**
- Create: `src/explain.py`
- Create: `scripts/05_visualize_attention.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: Trained model checkpoint, input sample (`image_path`, `text`, crop coordinates)
- Produces: Attention overlay figure saved to `reports/figures/{doc_id}_attention.png` displaying image bounding boxes weighted by attention scores alongside top attended text keywords.

- [x] **Step 1: Write test for attention weight extraction and heatmap generation**

```python
# tests/test_explain.py
import os
import torch
import pytest
from PIL import Image
from src.explain import visualize_sample_attention

def test_visualize_attention_generates_figure(tmp_path):
    img = Image.new("RGB", (300, 300), color=(150, 150, 150))
    boxes = [[20, 20, 100, 100], [120, 120, 200, 200], [50, 150, 150, 250]]
    attn_weights = torch.softmax(torch.randn(1, 4), dim=-1).squeeze(0).numpy() # 1 global + 3 crops
    out_file = tmp_path / "attn_test.png"
    
    visualize_sample_attention(
        image=img,
        boxes=boxes,
        attn_scores=attn_weights,
        text_tokens=["city", "council", "protest", "police"],
        top_text_indices=[2, 3],
        output_path=str(out_file)
    )
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explain.py -v`  
Expected: FAIL.

- [x] **Step 3: Implement `src/explain.py` and `scripts/05_visualize_attention.py`**

Use matplotlib/cv2/PIL to draw bounding boxes with alpha opacity scaled by cross-attention weights, plot attended text words, and save high-resolution figures.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explain.py -v`  
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/explain.py scripts/05_visualize_attention.py tests/test_explain.py
git commit -m "feat(explain): implement cross-attention interpretability and heatmap visualizer"
```
