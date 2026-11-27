# Grounding-DINO + Unified SigLIP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Faster R-CNN` with frame-neutral `Grounding-DINO` in `src/entity_extractor.py`, replace `DeBERTa-v3` with `SigLIP`'s own text encoder in `src/feature_extractor.py`, and run a 7-experiment, 3-seed (`[13, 42, 2026]`) ablation suite comparing the unified 2-backbone system against En-Shuo's `Qwen2.5-VL` baselines.

**Architecture:** Clean 3-part pipeline (`Grounding-DINO` → crops → `SigLIP Vision + Text` → `CrossModalFramingModel` + `ASL`). Vision and text share SigLIP's 768-d aligned embedding space (`max_text_len=64`). New crops and features are saved to `cache/crops_gdino/` and `cache/features_gdino_siglip/` so existing caches remain untouched. Ablation predictions are also exported in `predictions/subtask2/<model>/seed<k>/{dev,test}.{tsv,probs.json}` for unified evaluation.

**Tech Stack:** PyTorch `2.15.0`, Transformers `5.22.1` (`IDEA-Research/grounding-dino-tiny`, `google/siglip-base-patch16-224`), Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-11-27-grounding-dino-regions-design.md`

## Global Constraints
- Environment: `task9/venv` on Mac M1 (`mps` with automatic `cpu` fallback if any Grounding-DINO op fails on MPS).
- Grounding-DINO prompt MUST be frame-neutral: `"person. crowd. face. sign. text. building. vehicle. object."` (no framing category words).
- SigLIP text max length is `64` tokens (`google/siglip-base-patch16-224`); outputs must remain `torch.float32` (`visual_tokens`: `(1+K, 768)`, `text_tokens`: `(64, 768)`, `text_mask`: `(64,)` `int64`).
- Data splits strictly anchored to `course_split.json` (437 Train / 20 Dev / 40 Test).
- Ablations run across 3 fixed seeds `[13, 42, 2026]` and report `mean ± std`.

## Review Focus
- Blank/low-contrast or odd-aspect-ratio images returning 0 Grounding-DINO detections must still deterministically fill `top_k` using the 6 fallback ratios.
- Empty string `""` or `None` text passed to SigLIP tokenizer must yield valid `(64, 768)` `float32` tensors and valid `int64` `text_mask` (avoiding `0/0` NaN in masked mean pooling).
- Duplicate or heavily overlapping Grounding-DINO boxes for the same object across multiple prompt labels must be deduplicated (`iou > 0.7`) before selecting Top-K.
- MPS compatibility: if `grounding-dino-tiny` raises an unsupported MPS op error, gracefully fall back to CPU without crashing batch extraction.
- `top_k=0` returns `([], [])` cleanly.

---

### Task 1: Frame-Neutral Grounding-DINO in `src/entity_extractor.py`

**Files:**
- Modify: `src/entity_extractor.py:1-157`
- Test: `tests/test_entity_extractor.py:1-95`

**Interfaces:**
- Produces: `SalientEntityExtractor(model_name="IDEA-Research/grounding-dino-tiny", prompt="person. crowd. face. sign. text. building. vehicle. object.", device=None)` with `.extract_regions(img: Image.Image, top_k: int = 3, conf_thresh: float = 0.25) -> Tuple[List[Image.Image], List[List[int]]]`, plus `get_shared_extractor(...)` and `extract_salient_regions(...)`.

- [ ] **Step 1: Update `tests/test_entity_extractor.py` for Grounding-DINO defaults and IoU deduplication**

Add/update tests in `tests/test_entity_extractor.py`:
- `test_extract_salient_regions_returns_k_crops` (validates `top_k=3`, `top_k=1`, `top_k=0` -> `([], [])`, and `box` bounds)
- `test_fallback_mechanism` (blank black image with `conf_thresh=0.5` returns exact deterministic fallback boxes `fb1, fb2, fb3`)
- `test_salient_entity_extractor_prompt_is_frame_neutral` (asserts `extractor.prompt == "person. crowd. face. sign. text. building. vehicle. object."`)

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_entity_extractor.py -v`
Expected: FAIL (`AttributeError: 'SalientEntityExtractor' object has no attribute 'prompt'`).

- [ ] **Step 3: Implement Grounding-DINO in `src/entity_extractor.py`**

Replace `fasterrcnn_mobilenet_v3_large_320_fpn` with `AutoProcessor` + `AutoModelForZeroShotObjectDetection` (`IDEA-Research/grounding-dino-tiny`), keeping `_clamp_box`, the 6 deterministic `fallback_ratios`, `get_shared_extractor`, and `extract_salient_regions`.
- Run processor on `(images=img_rgb, text=self.prompt, return_tensors="pt")` and post-process with `threshold=conf_thresh, text_threshold=conf_thresh, target_sizes=[(h, w)]`.
- Sort candidates by `(score, area)` descending, deduplicate identical or high-IoU (`iou > 0.7`) boxes, then fill remaining slots with `fallback_ratios`.
- If MPS raises a runtime error during inference, fall back `self.device` to `cpu` once.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_entity_extractor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entity_extractor.py tests/test_entity_extractor.py
git commit -m "feat(regions): replace Faster R-CNN with frame-neutral Grounding-DINO"
```

---

### Task 2: Unified SigLIP Text + Vision in `src/feature_extractor.py`

**Files:**
- Modify: `src/feature_extractor.py:1-142`
- Test: `tests/test_feature_extractor.py:1-75`

**Interfaces:**
- Produces: `FeatureExtractor(vision_model_name="google/siglip-base-patch16-224", text_model_name="google/siglip-base-patch16-224", max_text_len=64, device=None)` returning `{"visual_tokens": (1+K, 768) float32, "text_tokens": (64, 768) float32, "text_mask": (64,) int64}`, and `get_shared_feature_extractor(...)`.

- [ ] **Step 1: Update `tests/test_feature_extractor.py` for SigLIP 64-token text output**

Update expected `text_tokens.shape` to `(64, 768)` and `text_mask.shape` to `(64,)` in `test_feature_extractor_shapes_and_types` and `test_feature_extractor_empty_text`, and assert no NaNs in `text_tokens`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_feature_extractor.py -v`
Expected: FAIL (`assert (256, 768) == (64, 768)`).

- [ ] **Step 3: Implement unified SigLIP Vision + Text in `src/feature_extractor.py`**

- Default `text_model_name: str = "google/siglip-base-patch16-224"` and `max_text_len: int = 64`.
- If `text_model_name == vision_model_name` and `hasattr(base_vision, "text_model")`, reuse `base_vision.text_model`.
- Tokenize with `padding="max_length", max_length=self.max_text_len, truncation=True, return_tensors="pt"`. Construct `text_mask` from `(input_ids != pad_id).to(torch.int64)` if `attention_mask` is absent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_feature_extractor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/feature_extractor.py tests/test_feature_extractor.py
git commit -m "feat(features): unify text and vision encoders on SigLIP"
```

---

### Task 3: Redesign `scripts/04_run_ablations.py` (7 Claims, 3 Seeds, Unified Export)

**Files:**
- Modify: `scripts/02_extract_features.py` (update defaults to `google/siglip-base-patch16-224` and `max_text_len=64`)
- Modify: `scripts/04_run_ablations.py`
- Test: `tests/test_ablations.py`

**Interfaces:**
- Produces 7-experiment suite in `run_ablation_suite(..., seeds=[13, 42, 2026])`:
  1. `Exp 1`: Text-Only (`SigLIP Text + ASL`)
  2. `Exp 2`: Vision-Only Global (`SigLIP Global + ASL`)
  3. `Exp 3`: Vision-Only Region (`SigLIP Global + Grounding-DINO + ASL`)
  4. `Exp 4`: Global + Text Cross-Attention (`No Regions + ASL`)
  5. `Exp 5`: Late Fusion (`SigLIP Global/Region + Text + ASL`)
  6. `Exp 6`: Full Cross-Modal (`Grounding-DINO + SigLIP + BCE`)
  7. `Exp 7`: Full Cross-Modal (`Grounding-DINO + SigLIP + ASL`)
- Exports per-seed `{dev,test}.tsv` and `{dev,test}.probs.json` into `predictions/subtask2/<exp_slug>/seed<k>/` alongside `reports/ablation_results_gdino_siglip.md`.

- [ ] **Step 1: Update `tests/test_ablations.py` for 7-experiment suite & multi-seed summary**

Verify `run_ablation_suite(..., seeds=[13, 42])` returns 7 rows with `micro_f1_mean`, `micro_f1_std`, `macro_f1_mean`, `macro_f1_std`, includes `Exp 4 (Global+Text)` and `Exp 5 (Late Fusion)`, and writes `{dev,test}.tsv` + `.probs.json`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_ablations.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement 7-experiment multi-seed ablation suite & update `02_extract_features.py`**

- In `04_run_ablations.py`, support `seeds: List[int] = [13, 42, 2026]` with `torch.manual_seed(seed)`, support `global_only` Cross-Attention (`visual_tokens[:, :1, :]`) for Exp 4 and `LateFusionModel` for Exp 5, export `{dev,test}.{tsv,probs.json}` per seed, and format `mean ± std` in `reports/ablation_results_gdino_siglip.md` alongside En-Shuo's `Qwen2.5-VL` reference rows (`0.417` image-only, `0.333` image+text).
- Update `02_extract_features.py` defaults to `google/siglip-base-patch16-224` and `max_text_len=64`.

- [ ] **Step 4: Run full test suite**

Run: `venv/bin/pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Re-extract 497 crops/features to new cache and run ablations**

```bash
venv/bin/python scripts/01_extract_regions.py --output_dir cache/crops_gdino --top_k 3 --conf_thresh 0.25 --overwrite
venv/bin/python scripts/02_extract_features.py --crop_dir cache/crops_gdino --output_dir cache/features_gdino_siglip --overwrite
venv/bin/python scripts/04_run_ablations.py --feature_dir cache/features_gdino_siglip --report_dir reports --epochs 15
```

- [ ] **Step 6: Commit**

```bash
git add scripts/02_extract_features.py scripts/04_run_ablations.py tests/test_ablations.py reports/
git commit -m "feat(ablations): add 7-experiment multi-seed ablation for Grounding-DINO + SigLIP"
```
