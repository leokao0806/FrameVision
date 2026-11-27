# Short Design Spec: Frame-Neutral Grounding-DINO + Unified SigLIP Pipeline

**Date**: 2026-11-27  
**Project**: SemEval-2027 Task 9 Subtask 2a (`task9/`)  
**Size**: Small Change (swapping Faster R-CNN -> Grounding-DINO and DeBERTa-v3 -> SigLIP Text Encoder)

---

## 1. Intent & Why
1. `Faster R-CNN` (COCO 91 classes) misses non-COCO news entities (signs, crowds, text, faces) and falls back to fixed crops (`Exp 3` dropped to `0.381` vs `Exp 2` `0.410`).
2. Mixing `SigLIP` (vision) with `DeBERTa-v3` (text) forces two unaligned embedding spaces to align from scratch on only 437 training samples.

Replacing both with **Frame-Neutral Grounding-DINO** + **Unified SigLIP (Vision & Text)** reduces the system to 2 pre-trained backbones + 1 lightweight fusion head, puts vision and text in the exact same 768-d semantic space, and fulfills the Proposal's `Grounding-DINO + SigLIP + Cross-Attention + ASL` design without injecting frame-label bias.

## 2. Goals, Non-Goals & Anti-Goals
- **Goals**:
  - Replace Faster R-CNN in `src/entity_extractor.py` with `IDEA-Research/grounding-dino-tiny` using a frame-neutral physical prompt: `"person. crowd. face. sign. text. building. vehicle. object."`.
  - Replace `microsoft/deberta-v3-base` in `src/feature_extractor.py` with `google/siglip-base-patch16-224` `text_model` (`max_text_len=64`), sharing one `AutoModel.from_pretrained("google/siglip-base-patch16-224")` load for both vision and text.
  - Re-extract crops and cached features (`--overwrite`), then re-run `04_run_ablations.py` to compare against the previous 4-model results.
- **Non-Goals**:
  - Do not add SAM segmentation masks (rectangular RGB crops preserve scene context for SigLIP).
  - Do not change `CrossModalFramingModel` architecture, `AsymmetricLoss`, or `course_split.json`.
- **Anti-Goals**:
  - No frame-specific keywords (e.g., `"protest"`, `"courtroom"`) in Grounding-DINO prompts.

## 3. Constraints
- **Environment**: `task9/venv` (`torch 2.15.0`, `transformers 5.22.1`) on Mac M1 (`mps` with CPU fallback if needed).
- **SigLIP Text Limit**: `google/siglip-base-patch16-224` supports up to `64` text tokens (focusing on headline + lead sentence).

## 4. Evaluated Approaches & Selection
1. **Frame-Neutral Grounding-DINO + Unified SigLIP (Selected)**: Clean 3-part pipeline (`Grounding-DINO` + `SigLIP Vision/Text` + `Cross-Modal Head`) with zero framing-label leakage and unified 768-d space.
2. **Keep DeBERTa-v3 with Grounding-DINO**: Isolates detector only, but retains 4 models and misaligned embedding spaces.
3. **No Detector (SigLIP 196 Patch Tokens)**: Zero detector overhead, but loses zoomed-in high-res crops for small signs/text.

## 5. What is Left to the Builder
- Verify `IDEA-Research/grounding-dino-tiny` on `mps` (with CPU fallback if needed) and tune box `conf_thresh` / deduplication.
- Handle `SiglipTextModel` tokenization & attention mask (`input_ids != pad_token_id` if tokenizer omits mask) cleanly in `src/feature_extractor.py`.
- Update `tests/test_entity_extractor.py` and `tests/test_feature_extractor.py`.
