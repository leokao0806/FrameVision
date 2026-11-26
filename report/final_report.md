# Seeing the Frame: Image-Level Framing in News (SemEval-2027 Task 9)
^**CS 5134/6034 Natural Language Processing, Fall 2026. Team FrameVision (Team [[TBD: team number]])**
^Leo Kao (kaoch@mail.uc.edu), En-Shuo Zhang (zhangeo@mail.uc.edu), University of Cincinnati

<!--
DRAFT, 2026-11-26. Markers:
  [[TBD: ...]]  waits for the locked results (results/LOCKED, plan Task 12) or a team decision.
  [[LEO: ...]]  Leo writes or confirms this part.
  The table:main and table:ablation markers are filled by report/build_tables.py from results/subtask2_final.json.
Every 0.xxx number must come from a results JSON (report/check_numbers.py). Do not type numbers by hand.
Writing rule (instructor comment 3): paragraphs, conclusion first; at most 3 bulleted lists in the whole report.
-->

## 1. Introduction

**Framing is a choice about which side of an issue to show, and news photographs make that choice too.** A report on a border crossing can be about security, about the economy or about human suffering, depending on what it emphasizes. Computational work on framing has mostly studied text (Card et al., 2015; Piskorski et al., 2023), but editors also choose a photograph, and readers often see it first. SemEval-2027 Task 9 asks which of 14 generic frames a news *image* conveys. We work on its Subtask 2 in English: given an article and its photograph, predict the image's frames, either without the article's gold text frames (2a) or with them (2b).

**Four observations shaped our system, and we treat the last two as hypotheses to test.** First (M1), the image is not a copy of the text. In the labeled training and development articles, an article's paragraphs carry about five frames on average while its image carries about one, and copying the text frames onto the image reaches recall 0.787 but precision only 0.162 (Section 9.1). A model must therefore look at the image, and the text can only serve as context. Second (M2), the data is small: a few hundred labeled training images and 40 test images [[TBD: 39 or 40]]. We therefore start from trivial baselines, frozen encoders and a zero-shot vision-language model (VLM), and we attach a confidence interval to every comparison. Third (M3), framing cues are often local, such as a protest sign, a police badge, a hospital bed or a flag, and a single vector for the whole image may dilute them. We therefore add image regions and let the text act as the query in cross-attention, so that the article decides *where to look* instead of injecting its frames directly. Fourth (M4), the labels are sparse, since most images have a single frame out of 15 possible labels, so we train with Asymmetric Loss (ASL). Experiments E2–E8 test M3 and M4 (Section 8).

**This report makes three contributions, summarized as key ideas in Section 2.** [[TBD after lock: one sentence per key idea with the headline result, e.g. what the best model is under our pre-registered rule and whether regions or text helped.]]

## 2. Our Key Ideas

**K1. The image picks one of the article's frames, and showing a VLM more text pulls it toward the article.** [[TBD after lock; En-Shuo + Leo agree on wording.]] Draft: In 69% of labeled articles every image frame is also one of the article's text frames, and in another 18% the image shares none of them (Section 9.1). The difficulty is therefore not that images and text disagree, but that the image *selects* one or two of the article's five frames. When we give Qwen2.5-VL the article, half or more of the labels it adds are text frames and few of them are correct (Section 9.2), whether it reads the title, the first 1,500 characters or the full text.

**K2. Open-vocabulary regions derived from frame definitions, with text as the attention query.** [[LEO: 3–4 sentences. What the region-aware model is, why Grounding-DINO phrases come from the 14 frame definitions only, and the E8 vs E5 / E6 vs E2 outcome, reported whatever the sign.]]

**K3. With 20 development and 40 test images, evaluation discipline decides what we can claim.** Draft: We split by article and show that a paragraph-level split inflates a text baseline's Micro F1 from 0.203 to 0.292 (Section 9.6). We fixed the model-selection rule and four primary comparisons before running any new experiment, we disclose the one time we read test results early, and we report a bootstrap interval for every number. [[TBD after lock: the consequence. If the best model is still a model that ties Qwen image-only on dev but loses on test, say so here in one sentence: the rule picked it, and we kept the rule.]]

## 3. Task and Data

**Each example is a news article with one photograph, and the target is the set of frames the photograph conveys.** The 14 frames are the generic framing dimensions of Card et al. (2015), such as Economic, Security and defense, Health and safety, Quality of life and Political; an image with no frame is labeled None, which the official scorer treats as a 15th label. The official metric is multi-label Micro F1, which pools true and false positives over all labels; we also report Macro F1 and Hamming loss. Subtask 1, the text side of the task, labels each paragraph with the same frames; we include it as a secondary result (Section 8.3).

**A trial example shows why copying text frames is not enough.** Article `EN_TRIAL_5` has the gold text frames Legality, Policy, Crime, Security and Health, but its image is labeled only Public opinion. [[TBD: optional figure with this image and both label sets, if space allows.]]

**We hold out our course development and test sets from the labeled English training release, because the official development set has no labels.** The second-batch English data has 97 labeled articles, each with one image. Table 1 shows how we use it. The split shuffles article IDs with a fixed seed and never reads labels; the 5 public trial articles are forced into training because we had scored them in earlier work, and articles that share an image or a text stay on the same side. No paragraph, article or image is an exact copy across training and development or test; we did not check for paraphrases, translations or cropped images. [[TBD: the revised second batch drops `EN_M_TRAIN_4`, a duplicate of the image of `EN_M_TRAIN_11`, which is in our test set; state whether the test set has 39 or 40 images.]]

| Part | Subtask 2 (images) | Subtask 1 (paragraphs) |
|---|---|---|
| Train | 437 English (37 second-batch + 400 first-batch); Image LR also uses 100 BG, 105 PT, 5 RU = 647 | 708 English (456 from the 37 articles + 252 SemEval-2023); the starter model uses 8040 multilingual |
| Course dev | 20 images | 161 paragraphs in 20 articles |
| Course test | 40 images [[TBD: 39?]] | 432 paragraphs in 40 articles (123 None) |

^*Table 1. Data used. Training sets differ by model family; Section 6 says which model uses which. [[LEO: confirm 437 for E1–E8.]]*

**The models never saw the course test set during training or selection.** Thresholds, epochs, the combination weight and prompt versions were chosen on the development set only. The baselines' test scores were already published in our intermediate report; Section 7 describes the rules we fixed before every new experiment and the one time we looked at test predictions early.

## 4. Related Work

**Framing detection began as a text problem.** Card et al. (2015) defined 15 general-purpose framing dimensions and annotated U.S. news articles with them; the task's 14 frames come from this scheme. SemEval-2023 Task 3 (Piskorski et al., 2023) asked systems to find these frames in news articles in nine languages, again from text alone. Our Subtask 1 baselines and part of our Subtask 1 training data come from this line of work.

**Multimodal framing is newer, and the closest work uses large models as annotators rather than as evaluated classifiers.** Arora et al. (2025) prompt large language and vision-language models with frame definitions to label both the text and the images of news at scale, and compare the frames the two modalities use. We adapt their idea of definition-based prompts with structured output for our zero-shot VLM, but we measure it against human gold labels on a held-out set, and we compare it with supervised models.

**Our trained models combine standard components.** SigLIP (Zhai et al., 2023) and DeBERTa-v3 (He et al., 2023) provide frozen image and text features; Qwen2.5-VL (Bai et al., 2025) is our zero-shot VLM. Asymmetric Loss (Ridnik et al., 2021) down-weights the many easy negative labels in sparse multi-label problems. For regions, Faster R-CNN (Ren et al., 2015) detects a fixed list of everyday object classes, while Grounding DINO (Liu et al., 2024) detects boxes for free-text phrases, which lets us ask for frame-related objects such as a protest sign. [[LEO: one sentence if K2 builds on any specific cross-attention paper.]]

**Small test sets need significance testing that respects how the data was sampled.** We follow the paired bootstrap recommended for NLP comparisons (Berg-Kirkpatrick et al., 2012), but resample whole articles, because the paragraphs and image of one article are not independent.

## 5. Resources

| Resource | Use | URL |
|---|---|---|
| Task data, starter kit, official scorers | Data, Subtask 1 starter, scoring | https://propaganda.math.unipd.it/semeval2027task9/ |
| mm-framing-baselines | Source of first-batch images | https://github.com/YovaKem/mm-framing-baselines |
| PyTorch 2.11 | Training | https://pytorch.org |
| Transformers 5.22 | Model loading | https://github.com/huggingface/transformers |
| scikit-learn 1.9 | Logistic regression, TF-IDF | https://scikit-learn.org |
| SigLIP base-patch16-224 | Image encoder | https://huggingface.co/google/siglip-base-patch16-224 |
| DeBERTa-v3-large | Text encoder (baselines, Subtask 1) | https://huggingface.co/microsoft/deberta-v3-large |
| DeBERTa-v3-base | Text encoder (E1, E4–E6, E8) | https://huggingface.co/microsoft/deberta-v3-base |
| XLM-RoBERTa base | Subtask 1 starter | https://huggingface.co/FacebookAI/xlm-roberta-base |
| Faster R-CNN MobileNetV3 (torchvision) | COCO regions (E3–E5) | https://pytorch.org/vision/stable/models/faster_rcnn.html |
| Grounding DINO tiny | Open-vocabulary regions (E7, E8) | https://huggingface.co/IDEA-Research/grounding-dino-tiny |
| Qwen2.5-VL 7B via Ollama | Zero-shot VLM | https://ollama.com/library/qwen2.5vl |

^*Table 2. External resources. [[LEO: check the region rows and add anything the region-aware pipeline uses.]]*

## 6. Methods

**6.1 Baselines.** Three trivial baselines set the floor: random labels drawn from training frequencies (averaged over 1,000 draws, reference only), always predicting the most frequent training label (Political), and, for 2b, copying the gold text frames. Image LR trains one-vs-rest logistic regression on frozen SigLIP image embeddings. Text LR does the same on the mean of an article's frozen DeBERTa-v3-large paragraph vectors. Late fusion averages the image and text probabilities; for 2b it also averages a logistic regression on the gold text frames. Each thresholded model picks its threshold on the development set.

**6.2 Zero-shot VLM.** We run Qwen2.5-VL 7B locally with temperature 0. The prompt lists the 14 frame definitions, asks for the frames conveyed by the *image itself*, and requires a JSON answer; no reply failed to parse. Six prompt variants differ only in what they add: the image alone; the image alone with an explicit None option and its definition (E10, chosen on the development set before any test query); the image with the article title (E12a); with the title and the first 1,500 characters (2a); with the full article (E12b); and with the title, the first 1,500 characters and the gold text frames (2b). The context window is 8,192 tokens, and the longest full-text prompt fits within it, so no prompt was silently cut.

**6.3 Region-aware fusion model.** [[LEO: architecture in your own words, with the figure slides/figures/architecture.svg. Whole-image and region tokens from SigLIP, text tokens from DeBERTa-v3-base, cross-attention with text as the query, masked mean pooling, MLP, dev threshold, None rule, ASL. Name the E1–E8 variants by what they remove.]]

**6.4 Open-vocabulary regions.** [[LEO: Grounding DINO phrases derived from the frame definitions only (fixed before E7/E8 ran), box and text thresholds at the model defaults, top 3 boxes, fixed crops as fallback, and the coverage statistic.]]

**6.5 Combination with the VLM (E9).** E9 tests whether the VLM and the best trained model make different mistakes. For each image, the score of a label is α times the VLM's 0/1 decision plus (1 − α) times the trained model's probability; a label is predicted when the score passes a threshold. We choose α from {0.25, 0.5, 0.75} and the threshold from 0.05 to 0.95 in steps of 0.05 on the development set, separately for each of the trained model's three seeds, and break ties toward the smaller α and then the smaller threshold. As an exploratory check we ran the same procedure with Image LR in place of a trained model; the development set chose a setting that reproduces Qwen image only exactly, so Image LR added nothing there. [[TBD after lock: E9 result in one sentence.]]

**Models read different amounts of the article, which matters for the question of why text hurts.** Table 3 lists what each model sees.

| Model | Text it reads |
|---|---|
| Qwen 2a, 2b | Title + first 1,500 characters (2b also gets the gold text frames) |
| Qwen E12a / E12b | Title only / full article |
| Text LR, late fusion | Full article, each paragraph encoded separately, then averaged |
| E1, E4–E6, E8 | First 256 DeBERTa-v3-base tokens [[LEO: confirm, and say whether the title is included]] |

^*Table 3. Text input per model.*

## 7. Experimental Setup

**All models are scored by the unmodified official scorer through one shared evaluation layer.** Every model writes its development and test predictions in the official file format, and the evaluation code calls the official Subtask 2 scorer on them; our own metric code must agree with the scorer, or the run stops. This layer reproduces every Subtask 2 number of our intermediate report. Trained models run with seeds 13, 42 and 2026 and are reported as the mean over seeds; the baselines and the VLM (temperature 0) run once. An earlier number in our repository README (Micro F1 0.4421 for an intermediate fusion model) came from a single seed and a re-implemented scorer and is withdrawn. [[LEO: confirm why it differed from the 0.417 in the ablation notes.]]

**Confidence intervals resample whole articles.** We draw 2,000 bootstrap samples of test articles with a fixed seed. For a multi-seed model, each sample scores every seed and averages them, so the interval reflects both the test sample and seed variation. A paired comparison scores both models on the same samples and reports the 95% interval of the difference.

**We fixed the selection rule and the main comparisons before running any new experiment.** The rule and the comparisons were committed to the repository before E1–E12 ran. The best model is the one with the highest development Micro F1 (seed-averaged), with ties going to the simpler model; the E12 text-length variants are candidates like every other model. A later amendment, committed before those models ran, makes models added after registration exploratory: they are reported but cannot become the best model. Four comparisons are primary:

1. the best model against always predicting the most frequent label;
2. the best model against Qwen image only;
3. E8 against E5, open-vocabulary against COCO regions (the core hypothesis);
4. E6 against E2, whether text helps the trained model.

All other comparisons are exploratory and not corrected for multiple testing.

**We read test results early once, and we disclose it.** On 2026-11-10, to answer the instructor's question on why text hurts the VLM, we inspected the existing VLM test predictions image by image before our results were locked. We designed E12 after that look, so E12 selects nothing: all three text lengths are run and reported on both sets. Analyses C1, C2 and C4 otherwise used only training and development data until the results were locked on [[TBD: lock date]].

**Five changes to the proposal concern the models.**

- Regions come from Faster R-CNN with COCO classes and from Grounding DINO instead of SAM. [[LEO: the actual reason in one clause.]]
- The fusion model uses DeBERTa-v3-base instead of -large. [[LEO: the actual reason.]]
- The image encoder is frozen: the proposal fine-tuned SigLIP or ViT-Large, but Image LR and E2–E8 train only the layers above a frozen SigLIP. [[LEO/En-Shuo: the actual reason.]]
- The Subtask 2 text encoder is frozen (Text LR and E1); fine-tuning is used only for Subtask 1.
- Grad-CAM is replaced by attention maps (Section 9.5), which show where the model attends but are not explanations.

Three earlier changes concern the evaluation and were already reported in our intermediate report: Micro F1 is the primary metric, the course splits come from labeled training data, and Qwen2.5-VL replaces the proposed LLaVA-1.6 and Qwen2-VL.

## 8. Results

**8.1 Main results.** [[TBD after lock: lead with the primary comparisons, one paragraph each, conclusion first. Say which model the rule selected and why it matters (K3). Numbers only from the table and figure.]]

<!-- table:main -->

![Main results](../results/figures/main_results.png)
^*Figure 1. Test Micro F1 with 95% article-level bootstrap intervals. The model selected on the development set is highlighted.*

**8.2 Ablations of the trained model.** [[TBD after lock: E2, E3, E5, E6, E7, E8 dev and test (mean ± std); what regions, text and the loss change. Leo checks the wording.]]

<!-- table:ablation -->

**8.3 Subtask 1.** Only the official starter model clearly beats the strongest trivial baseline on paragraphs. Always predicting None scores 0.239 on the 432 test paragraphs, because more than a quarter of them have no frame. The starter XLM-RoBERTa reaches 0.320, fine-tuned DeBERTa-v3 reaches 0.297 with binary cross-entropy and 0.278 with ASL, and TF-IDF with logistic regression reaches 0.253. The development set selects DeBERTa-v3 with cross-entropy (0.356 on development), not the starter, but the two differ in training data (8040 multilingual against 708 English paragraphs) as well as in architecture, so the comparison does not isolate either factor. ASL did not deliver the gain we proposed it for and was unstable: in one seed, training collapsed at epoch 3 to an output that barely depends on the input, so that every threshold did worse than predicting None everywhere. We could not find the trigger because the run kept no per-step losses, gradient norms or checkpoints, and we report it as unresolved. [[TBD: paired intervals against Always None, if space allows.]]

## 9. Analysis

**9.1 How image frames relate to text frames (C1).** The image usually shows a subset of the article's frames rather than different ones. Over the 267 labeled articles with paragraph frames in training and development, all languages, we compare each image's frames with the union of its paragraphs' frames. In 69% of articles every image frame is also a text frame, in 4% the two sets partly overlap, in 18% they share nothing, and in 9% the image has no frame. Because an article carries about five text frames and its image about one, copying the text frames finds most image frames (recall 0.787) but is mostly wrong (precision 0.162). English articles alone show the same pattern with more divergence (26% share nothing). Figure 2 shows which image frames appear given each text frame. [[TBD after lock: add the test articles to the counts and the figure.]]

![C1 heatmap](../results/figures/c1_heatmap.png)
^*Figure 2. P(image frame | text frame) over labeled articles.*

**9.2 Why text makes the VLM worse, and whether length matters (C2, E12).** Text makes Qwen add labels that are article frames but not image frames. On the development set, the image-only prompt predicts 1.2 labels per image; with the title and first 1,500 characters it predicts 1.4, and of the 7 labels it adds, 4 are text frames and only 1 is correct, while it drops 3 labels, 1 of which was correct. Reading the full article instead adds 8 labels (2 correct, 4 from the text frames), and the title alone adds 10 (1 correct, 7 from the text frames). With the gold text frames in the prompt (2b), 88% of its labels are text frames. The truncated 2a prompt sees a median of 59% of an article (95% of development articles are longer than 1,500 characters), so the effect is not caused by truncation alone: every length moves the prediction toward the article. [[TBD after lock: replace with development and test numbers from results/c2_vlm_text.json, and add the E12 test scores from the main table. Cite Figure 3.]]

![C2](../results/figures/c2.png)
^*Figure 3. Left: Qwen Micro F1 by text length. Right: labels added and removed relative to image only.*

**9.3 Per-frame performance (C3).** [[TBD after lock: per-label F1 with support for the best model, Qwen image only and E2; rare frames and None.]]

**9.4 Error analysis (C4).** [[TBD after lock: about 8 cases from development and test.]] Draft from the development cases: both Qwen prompts never answer None, so the two development images without a frame are always wrong; text adds plausible article topics to images that show something narrower (an image labeled Political gains Cultural identity, Health and Public opinion when Qwen reads the full article); and Quality of life and Health and safety are often confused. The full-text prompt also fixes two development images that image only gets wrong and breaks one, so the two end up almost tied on development.

**9.5 Regions and attention (C5).** [[LEO: detector coverage for COCO and Grounding DINO, two or three development/test examples, and attention maps; state that attention weights are not explanations.]]

**9.6 Splitting by article matters (C6).** Splitting paragraphs at random lets a model see other paragraphs of the same article, and this inflates its score. On the 617 non-test second-batch paragraphs, TF-IDF with logistic regression gets Micro F1 0.292 with random paragraph folds but 0.203 when whole articles are held out (one run each). Every split in this report is by article.

## 10. Limitations, Lessons and Conclusion

[[TBD after lock; written by both.]] Draft points, to become two or three paragraphs:
the 20-image development set makes selection noisy, which the selected best model illustrates [[TBD: keep only if still true]]; 40 test images give intervals about as wide as the differences we hoped to detect; model families were trained on different data (437 English against 647 multilingual images); the VLM never predicts None; ASL was unstable on Subtask 1; and we read test predictions once before locking. Lessons: the simplest strong baseline (Qwen on the image alone) was the hardest to beat; more text is not more signal when the label belongs to the image; fixing the rules in advance kept us from picking the best-looking test number. [[LEO: lessons from the region-aware model.]]

## 11. Contributions

Our team has two members; the instructor approved a two-person team. **Leo Kao** built the PyTorch training pipeline and the SigLIP features, the region-aware cross-attention model with ASL, the COCO and Grounding DINO region extraction, experiments E1–E8 [[LEO: and E11?]], the region and attention analysis (C5) and the architecture figure. **En-Shuo Zhang** built the data preprocessing and splits, the baselines (logistic regression, late fusion with Leo, Subtask 1 models), the VLM pipeline with its prompt and text-length variants (E10, E12), the shared evaluation layer with the pre-registered rules and bootstrap intervals, the combination (E9), analyses C1–C4 and C6, and the report tooling. [[LEO: check and edit your part.]]

## References

- J. Bai et al. Qwen2.5-VL technical report. arXiv:2502.13923, 2025.
- A. Arora, S. Yadav, M. Antoniak, S. Belongie, and I. Augenstein. Multi-modal framing analysis of news. In *Proc. of EMNLP*, pages 31531–31553, 2025.
- T. Berg-Kirkpatrick, D. Burkett, and D. Klein. An empirical investigation of statistical significance in NLP. In *Proc. of EMNLP-CoNLL*, 2012.
- D. Card, A. E. Boydstun, J. H. Gross, P. Resnik, and N. A. Smith. The Media Frames Corpus: Annotations of frames across issues. In *Proc. of ACL*, 2015.
- P. He, J. Gao, and W. Chen. DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In *Proc. of ICLR*, 2023.
- S. Liu et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In *Proc. of ECCV*, 2024.
- J. Piskorski, N. Stefanovitch, G. Da San Martino, and P. Nakov. SemEval-2023 Task 3: Detecting the category, the framing, and the persuasion techniques in online news in a multi-lingual setup. In *Proc. of SemEval*, 2023.
- S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In *Proc. of NeurIPS*, 2015.
- T. Ridnik et al. Asymmetric loss for multi-label classification. In *Proc. of ICCV*, 2021.
- X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In *Proc. of ICCV*, 2023.
