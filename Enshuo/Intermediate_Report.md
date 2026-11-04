# Multimodal Framing in News: Intermediate Report
^**CS 5134/6034 Natural Language Processing, Fall 2026. Team FrameVision**
^Leo Kao (kaoch@mail.uc.edu), En-Shuo Zhang (zhangeo@mail.uc.edu), University of Cincinnati

## 1. Task and Data
We work on SemEval-2027 Task 9, Analysis of Multimodal Framing in the News, English track. Our target is Subtask 2: given a news article and its photograph, predict which of 14 generic frames (Card et al., 2015) the *image* conveys. Subtask 2a gives no gold text frames; 2b gives them. We also run Subtask 1 (paragraph frames), the text side of the problem. The official metric is multi-label **Micro F1**, with Macro F1 as a complementary score; "None" counts as a 15th label.

**Example from the data.** `EN_TRIAL_5` has gold text frames Legality, Policy, Crime, Security and Health, but its image is labeled only `Public_opinion`. An image can carry a frame the text does not, so copying text frames is not enough.

**Data used.** Official training data only; the official dev set is released without labels, so the proposal's plan to split it is not possible. We hold out our course dev and test sets from the labeled second-batch English data (97 articles, 1049 paragraphs, one image each).

| Part | Subtask 1 (paragraphs) | Subtask 2 (images) |
|---|---|---|
| Train | 37 EN articles (456), plus 252 EN SemEval-2023 paragraphs = 708. XLM-RoBERTa: all languages, 8568 | 647: the same 37, 400 first-batch EN, 100 BG, 105 PT, 5 RU |
| Course dev | 20 EN articles: 161 (48 None) | 20 images |
| Course test | 40 EN articles: 432 (123 None) | 40 images (5 None) |

**How the split was made.** A fixed-seed shuffle of article IDs that did not read labels (`src/splits.py`, `results/course_split.json`). The 5 trial articles are forced into train because we scored them in earlier work. Articles sharing an image (2 pairs) or text (1 pair) stay together. Trial articles repeated in the second batch are counted once.

**The test data was not used for training or tuning.** Thresholds, epochs and checkpoints are selected on dev only, and the VLM prompt was not changed. We saw test scores while running the models and report all of them, without choosing any configuration from them. Our earlier trial-only scores are superseded and not reported.

## 2. Resources
| Resource | Use | URL |
|---|---|---|
| Task data, starter kit, official scorers | Data, baseline, scoring | https://propaganda.math.unipd.it/semeval2027task9/ |
| mm-framing-baselines | Source of first-batch images | https://github.com/YovaKem/mm-framing-baselines |
| PyTorch 2.11 | Training | https://pytorch.org |
| Transformers 5.22 | Model loading | https://github.com/huggingface/transformers |
| scikit-learn 1.9 | LR, TF-IDF | https://scikit-learn.org |
| XLM-RoBERTa base | Starter baseline | https://huggingface.co/FacebookAI/xlm-roberta-base |
| DeBERTa-v3-large | Text encoder | https://huggingface.co/microsoft/deberta-v3-large |
| SigLIP base-patch16-224 | Image encoder | https://huggingface.co/google/siglip-base-patch16-224 |
| Qwen2.5-VL 7B via Ollama | Zero-shot VLM | https://ollama.com/library/qwen2.5vl |

Every test score comes from the unmodified official scorers, and the run stops if they disagree with our metrics code. The scorer gives F1 = 1 to a label absent from both gold and predictions, which raises Macro F1. The Subtask 2 scorer's comment swaps 2a and 2b; we follow the task website.

## 3. Technical Description
**Trivial baselines.** Random sampling from training label frequencies (mean of 1000 draws), always predicting the most frequent training label, and, for 2b, copying the gold text frames.

**Subtask 1 models.** (a) TF-IDF with one-vs-rest logistic regression (LR). (b) Frozen, mean-pooled DeBERTa-v3 with LR. (c) The official starter baseline, weighted XLM-RoBERTa (3 seeds). (d) Fine-tuned DeBERTa-v3 (lr 2e-5, up to 10 epochs, batch size 4, 256 tokens, 3 seeds), comparing plain binary cross-entropy (BCE) with Asymmetric Loss (ASL; Ridnik et al., 2021; gamma-neg 4, gamma-pos 1, clip 0.05).

**Subtask 2 models.** (a) Image LR on frozen SigLIP embeddings. (b) Text LR on the mean of an article's frozen DeBERTa-v3 paragraph vectors, trained on the 437 English articles. (c) Late fusion, averaging image and text probabilities; for 2b it also averages an LR on gold text frames (247 articles have them). (d) Zero-shot Qwen2.5-VL with three prompts: image only, image + text (2a), and image + text + gold frames (2b). The temperature is 0, and no reply failed to parse. The starter kit has no Subtask 2 baseline.

**Differences from previous work.** We reimplemented the starter baseline with its settings (lr 2e-5, 8 epochs, batch 8, accumulation 2, label weights, 256 tokens, inference threshold 0.5). Two changes: we keep None paragraphs, and we pick checkpoints on our by-article dev set rather than a random 10% of paragraphs. We cannot say whether we match its performance: the kit reports Micro 0.458 and Macro 0.435 on an undocumented test set without None, and ships no weights. The VLM prompts adapt the frame-definition, structured-output approach of Arora et al. (2025) on different data, so their scores are also not comparable. Unlike SemEval-2023 Task 3 (Piskorski et al., 2023), this task labels images as well as text.

**Changes since the proposal.** The Canvas comment on our proposal was "Nice proposal. Look forward to your progress!", with no further suggestions, so these changes are our own decisions.
- **Metric:** Micro F1 is now primary, following the official task definition, instead of Macro F1.
- **Split:** course dev and test come from labeled training data, because the official dev set has no labels.
- **Vision baseline:** frozen SigLIP with LR replaces fine-tuning, as a cheap first baseline.
- **Thresholds:** they are selected on dev, not on training data as proposed, since training scores would overstate performance.
- **VLM:** Qwen2.5-VL replaces the proposed LLaVA-1.6 / Qwen2-VL options.
- **Scope:** we added Subtask 1 to the proposed Subtask 2, as the text side of the problem.
- **Added baselines:** random, most frequent, copied text frames, TF-IDF, frozen DeBERTa-v3, official starter.
- **Text baseline:** fine-tuned DeBERTa-v3 runs on Subtask 1 only; Subtask 2's Text LR uses frozen features.
- **Proposed model:** the region-aware network exists only as a smoke-tested prototype (`src/models/cross_attention_model.py`) without SAM. It is not evaluated here, in line with the proposal's Nov 6-20 schedule.

## 4. Evaluation
| Subtask 2 (40 test images) | Micro F1 [95% CI] | Macro F1 | Hamming | Dev Micro |
|---|---|---|---|---|
| Random (training prior) | 0.132 | 0.128 | 0.164 | - |
| Most frequent label (`Political`) | 0.295 [0.169, 0.422] | 0.166 | 0.103 | - |
| Copy gold text frames (2b) | 0.238 [0.175, 0.302] | 0.159 | 0.320 | - |
| Image LR (SigLIP) | 0.353 [0.269, 0.444] | 0.281 | 0.165 | 0.419 |
| Text LR (DeBERTa-v3) | 0.152 [0.060, 0.255] | 0.189 | 0.148 | 0.408 |
| Late fusion, image + text (2a) | 0.367 [0.248, 0.486] | **0.350** | 0.115 | 0.444 |
| Late fusion + gold frames (2b) | 0.291 [0.191, 0.400] | 0.281 | 0.138 | 0.456 |
| Qwen2.5-VL, image only | **0.417** [0.292, 0.553] | 0.205 | 0.093 | **0.478** |
| Qwen2.5-VL, image + text (2a) | 0.333 [0.216, 0.463] | 0.229 | 0.107 | 0.440 |
| Qwen2.5-VL, + gold frames (2b) | 0.340 [0.222, 0.468] | 0.194 | 0.110 | 0.426 |

| Subtask 1 (432 test paragraphs) | Micro F1 [95% CI] | Macro F1 | Hamming | Dev Micro |
|---|---|---|---|---|
| Random (training prior) | 0.130 | 0.094 | 0.170 | - |
| Most frequent label | 0.144 [0.108, 0.181] | 0.019 | 0.136 | - |
| TF-IDF + LR | 0.253 [0.217, 0.289] | 0.186 | 0.190 | 0.289 |
| Frozen DeBERTa-v3 + LR | 0.214 [0.184, 0.245] | 0.106 | 0.128 | 0.234 |
| Starter XLM-RoBERTa | **0.320** ± 0.005 [0.294, 0.359] | **0.306** | 0.144 | 0.314 |
| Fine-tuned DeBERTa-v3, BCE | 0.297 ± 0.015 [0.241, 0.322] | 0.260 | 0.141 | **0.356** |
| Fine-tuned DeBERTa-v3, ASL | 0.278 ± 0.017 [0.230, 0.304] | 0.172 | **0.125** | 0.324 |

Intervals resample whole articles (2000 draws). Macro F1 is the official value, over all 15 labels. The last three rows are mean ± std over 3 seeds; their intervals and paired comparisons use the seed-42 run.

![Per-label F1](plots/figure3_subtask2_per_label_f1.png)
^*Figure 1. Subtask 2 per-label F1 on test images; n = gold images; labels with no test images omitted.*

**Interpretation.**
- **No image model clearly beats the baseline yet.** Qwen2.5-VL image only is best on both dev and test, but its gain over always predicting `Political` has a paired 95% CI of [-0.033, 0.281]. It also has no clear edge over Image LR ([-0.050, 0.173]). Forty images is too few to separate these models.
- **Adding text to the VLM did not help.** Image + text scores below image only on test (paired CI [-0.160, -0.016]) and dev (0.440 vs 0.478). This is one of several uncorrected comparisons, so we treat it as a lead to retest. Gold frames (2b) show no clear change ([-0.180, 0.023]).
- **Image and text frames differ.** Copying gold text frames has recall 0.625 but precision 0.147: most text frames are not image frames.
- **Metrics disagree.** Late fusion has the top Macro F1 (0.350), the VLM the top Micro F1. Two labels have no test images, and scorer-rule Macro is sensitive to them.
- **Rare frames and None are unsolved.** Figure 1 shows both models at zero for `None` (n=5) and `Morality` (n=3); the VLM always outputs a frame.
- **Small dev sets mislead.** Text LR selected threshold 0.9 with dev 0.408, then fell to 0.152 on test. We report this and did not change it.
- **Article leakage is real.** On the 617 non-test second-batch paragraphs, TF-IDF + LR gets Micro F1 0.292 with random paragraph folds but 0.203 with by-article folds.
- **Subtask 1: models beat the baseline, but fine-tuning DeBERTa-v3 adds little.** All five trained models beat the most frequent label (paired CIs exclude 0; the lowest is frozen DeBERTa-v3, [0.025, 0.119]). Dev selects BCE, but the starter is highest on test. On seed 42, BCE is below it ([-0.085, -0.006]); on seed 2026 they nearly tie (0.318 vs 0.321). The starter trains on 12 times more multilingual paragraphs, which may explain this, but we have not tested it.
- **ASL did not help and was unstable.** Its difference from BCE has a paired CI of [-0.071, 0.043]. In ASL seed 42, training loss rose from epoch 3 and dev stayed at 0.245, exactly the score of predicting None everywhere. We kept that run, using epoch 2 as selected on dev. The cause is unknown, and this failed run is the one used in the paired comparison.

**Challenges.** (1) The release mixes sources: trial articles repeat in the second batch, some images are duplicated, and "no frame" appears as None, `-` and `No framing`, so we normalized these. (2) First-batch data differs from the second batch: its most common image label is `Quality_of_life`, not `Political`, and its Subtask 1 articles have no None rows. We have not measured whether it helps. (3) DeBERTa-v3 first gave NaN losses: its float16 checkpoint breaks Adam, so we force float32 and stop on non-finite loss. (4) Fine-tuning is slow on our 8 GB GPU, about 2-5 hours per seed, so we use 3 seeds, not 5. (5) An earlier internal draft used stand-in models and a simulated VLM score; all of its numbers are withdrawn.

## 5. Plans
- **Next:** submit dev predictions to the leaderboard once it allows, test whether first-batch data helps or hurts, using dev only, and find out why ASL seed 42 failed.
- **Nov 6-20:** add SAM regions to the cross-attention prototype and compare it with Image LR and the VLM, with ablations of region input, text input and ASL versus BCE. Add a None decision for the VLM, analyze rare-frame errors, and use Grad-CAM as proposed.
- **Nov 19-Dec 3:** in-class presentation. **Dec 3:** final report, code and slides.

## 6. Contributions
We kept the division of work from the proposal.
- **Leo Kao (multimodal and deep learning):** PyTorch training pipeline, SigLIP features and image baseline, ASL, cross-attention prototype, late fusion (with En-Shuo). Next: SAM / Grounding-DINO regions and Grad-CAM.
- **En-Shuo Zhang (data and baselines):** data preprocessing, validation and splits; DeBERTa-v3, TF-IDF and starter baselines; VLM prompts and pipeline; evaluation code; late fusion (with Leo). Next: qualitative error analysis.

## References
- D. Card, A. E. Boydstun, J. H. Gross, P. Resnik, and N. A. Smith. The Media Frames Corpus: Annotations of frames across issues. In *Proc. of ACL*, 2015.
- A. Arora, S. Yadav, M. Antoniak, S. Belongie, and I. Augenstein. Multi-modal framing analysis of news. In *Proc. of EMNLP*, pages 31531-31553, 2025.
- J. Piskorski et al. SemEval-2023 Task 3: Detecting the category, the framing, and the persuasion techniques in online news in a multi-lingual setup. In *Proc. of SemEval*, 2023.
- T. Ridnik et al. Asymmetric loss for multi-label classification. In *Proc. of ICCV*, 2021.
