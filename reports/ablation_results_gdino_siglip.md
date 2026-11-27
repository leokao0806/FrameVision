# SemEval-2027 Task 9 Subtask 2 - Grounding-DINO + SigLIP Ablations

Seven experiments, each answering one question; scores are mean ± std over seeds [13, 42, 2026].
Qwen2.5-VL rows are En-Shuo's zero-shot baselines for reference.

| Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |
| :--- | :--- | :--- | :--- | :--- |
| Image LR (SigLIP) | 0.353 | 0.281 | - | Intermediate Report baseline |
| Late fusion (Image + Text) | 0.367 | 0.350 | - | Intermediate Report baseline |
| Qwen2.5-VL (Image only) | 0.417 | 0.205 | - | Intermediate Report zero-shot |
| Qwen2.5-VL (Image + Text) | 0.333 | 0.229 | - | Intermediate Report zero-shot |
| Exp 1: Text-Only (SigLIP text + ASL) | 0.303 ± 0.042 | 0.261 ± 0.007 | 0.150 | mean ± std over 3 seeds |
| Exp 2: Vision-Only Global (SigLIP + ASL) | 0.389 ± 0.086 | 0.317 ± 0.078 | 0.104 | mean ± std over 3 seeds |
| Exp 3: Vision-Only Global + Grounding-DINO Regions (ASL) | 0.446 ± 0.051 | 0.316 ± 0.018 | 0.089 | mean ± std over 3 seeds |
| Exp 4: Global + Text Cross-Attention, no regions (ASL) | 0.430 ± 0.018 | 0.341 ± 0.039 | 0.109 | mean ± std over 3 seeds |
| Exp 5: Late Fusion (global image + text, ASL) | 0.393 ± 0.042 | 0.287 ± 0.014 | 0.113 | mean ± std over 3 seeds |
| Exp 6: Full Cross-Modal, Grounding-DINO (BCE) | 0.401 ± 0.015 | 0.316 ± 0.027 | 0.104 | mean ± std over 3 seeds |
| Exp 7: Full Cross-Modal, Grounding-DINO (ASL) | 0.422 ± 0.041 | 0.310 ± 0.055 | 0.119 | mean ± std over 3 seeds |
