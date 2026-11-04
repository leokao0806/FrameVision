# SemEval-2027 Task 9 Subtask 2 - Ablation Benchmark Results

Comparison of proposed cross-modal framing architecture against unimodal baselines and team Intermediate Report Table 1 benchmarks.

| Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |
| :--- | :--- | :--- | :--- | :--- |
| Image LR (SigLIP) | 0.353 | 0.281 | - | Intermediate Report baseline |
| Late fusion (Image + Text) | 0.367 | 0.350 | - | Intermediate Report baseline |
| Qwen2.5-VL (Image only) | 0.417 | 0.205 | - | Intermediate Report zero-shot |
| Qwen2.5-VL (Image + Text) | 0.333 | 0.229 | - | Intermediate Report zero-shot |
| Exp 1: Text-Only (DeBERTa-v3 + ASL) | 0.273 | 0.199 | 0.107 | Completed |
| Exp 2: Vision-Only Global (SigLIP Global + ASL) | 0.410 | 0.245 | 0.120 | Completed |
| Exp 3: Vision-Only Region Attention (SigLIP Global + K Regions + ASL) | 0.381 | 0.267 | 0.108 | Completed |
| Exp 4: Multimodal Cross-Attention (BCE Loss) | 0.356 | 0.276 | 0.127 | Completed |
| Exp 5: Multimodal Cross-Attention (ASL Full Model) | 0.417 | 0.298 | 0.117 | Completed |
