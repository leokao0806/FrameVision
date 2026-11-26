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
