# Pre-registered evaluation rules (Subtask 2, final report)

These rules are fixed before any new final-report experiment is run (E1–E12). The commit date of this file is the
reference point. Results that existed before it — the Intermediate Report baselines and Leo's intermediate ablations
(Exp1–Exp5) — were produced before these rules were written.

## Selection

- Epochs, thresholds, the E9 mixing weight α, and the E10 prompt version are chosen on **dev only**.
- Test is never used for any selection.

## Best model and best trained model

- **Best model**: among all models, the one with the highest dev Micro F1 averaged over its 3 seeds (a single-run
  model uses its one score).
- **Best trained model** (used by E9 and E11): among E1–E8, the one with the highest seed-averaged dev Micro F1.
- **Ties** go to the simpler model, in this order:
  trivial < logistic regression < zero-shot VLM < trained single-modality < trained fusion < combination.
  Two dev scores count as tied when they differ by at most 1e-12. The exact order used by
  `evaluation/run_eval.py` (`COMPLEXITY_ORDER`), simplest first:
  1. Trivial: `most_frequent`, `copy_text_frames_2b`
  2. Logistic regression: `image_lr`, `text_lr`, `late_fusion_2a`, `late_fusion_2b`
  3. Zero-shot VLM: `qwen_image_only`, `qwen_image_only_v2`, `qwen_2a_title`, `qwen_2a`, `qwen_2a_fulltext`, `qwen_2b`
  4. Trained single-modality: `e1_text_only`, `e2_vision_global`, `e3_vision_region_coco`, `e7_vision_region_gdino`
  5. Trained fusion: `e6_fusion_global_asl`, `e4_fusion_coco_bce`, `e5_fusion_coco_asl`, `e8_fusion_gdino_asl`
  6. Combination: `combo_qwen_best`

  Any other model name (for example the E11 data ablation `e11_*_no_batch1`) ranks after all of these, then by name.
- **E12 variants are candidates.** `qwen_2a_title` and `qwen_2a_fulltext` compete for best model under the same
  dev-only rule as every other model. "E12 selects nothing" (below) means that no text length is picked for
  reporting: all three are reported on dev and test.

## Primary comparisons (4, specified in advance)

1. Best model vs. always predicting the most frequent label.
2. Best model vs. Qwen image-only.
3. E8 vs. E5: does switching to open-vocabulary regions help (the core hypothesis).
4. E6 vs. E2: does text help.

All other comparisons are reported as **exploratory, uncorrected** for multiple comparisons.

## Limitations stated in advance

- Dev has only 20 images, so selection on dev is noisy. The report discusses this under Limitations.

## Test access before the numbers are locked

Test labels and test predictions are not used by C1, C2 or C4 before `results/LOCKED` exists.

**Exception (disclosed).** On 2026-11-10 an exploratory C2 look read the existing Qwen test predictions and test labels
to answer the instructor's comment on the Intermediate Report; see `docs/notes/2026-11-10-c2-prelock-test-look.md`.
E12 was specified after that look, so E12 selects nothing: all three text lengths (title only, title + first 1,500
characters, full text) run on dev and test and are all reported. The report's Experimental Setup discloses this.
