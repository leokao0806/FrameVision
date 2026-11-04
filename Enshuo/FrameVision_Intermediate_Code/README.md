# Multimodal Framing in News

CS 5134/6034 NLP, Fall 2026, Team FrameVision: Leo Kao, En-Shuo Zhang.

Baselines for SemEval-2027 Task 9 (Subtask 1 paragraph frames, Subtask 2 image frames) on the official training data. Results and their limits are in `Intermediate_Report.pdf`. The primary metric is Micro F1, following the task definition, and test scores come from the unmodified official scorers.

## Data

The code zip does not contain data, models or the starter kit. Place the official release under `data/`:

- `data/data-second-batch/data/`, containing `training-second-batch-oct` and `initial-training-data`
- `data/trialdataSemEval2027Task9/` (only the Russian trial articles are new; the rest repeat in the second batch)
- `data/starter-kit/starter-kit/` (the official scorers)

## Setup

Python 3.14 was used. Exact tested versions are in `requirements.lock.txt` (not needed to run).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv\Scripts\pip install -r requirements.txt
```

The VLM steps need [Ollama](https://ollama.com) running locally with the model pulled:

```bash
ollama pull qwen2.5vl:7b
```

XLM-RoBERTa, DeBERTa-v3 and SigLIP are downloaded from Hugging Face on first use.

## Reproduce

Run from the project root with the virtual environment's Python. Fine-tuning is slow (hours per seed on an 8 GB GPU); finished runs are saved in `results/runs` and are not repeated.

```bash
python -m unittest discover -s tests
```

```bash
python scripts/run_experiments.py subtask1 subtask2 vlm_dev vlm_test analysis compare
```

```bash
python src/visualize.py
```

```bash
python scripts/generate_doc.py
```

`generate_doc.py` needs Microsoft Word for the PDF step. It builds the `.docx`, `.pdf` and code zip from `Intermediate_Report.md`, which is the only file to edit for report text. It stops if the PDF is over 4 pages or the zip is over 10 MB.

## Splits

Course dev (20 articles) and test (40 articles) are held out from the 97 second-batch English articles by a fixed-seed shuffle of article IDs; the rest is training data. See `src/splits.py` and `results/course_split.json`. The 5 trial articles stay in training, and articles that share an image or text stay together. The official dev set has no labels, so it is not used here.

## Layout

| Path | Contents |
|---|---|
| `src/data_loader.py`, `src/splits.py` | Official data parsing and course split |
| `src/official_scorer.py` | Runs the unmodified official scorers |
| `src/metrics.py`, `src/loss.py` | Diagnostics and bootstrap intervals, BCE and ASL |
| `src/features.py` | Frozen SigLIP and DeBERTa-v3 features, cached in `results/cache` |
| `src/models/linear_baselines.py` | Trivial, TF-IDF and logistic regression baselines |
| `src/models/starter_baseline.py` | Starter-kit XLM-RoBERTa baseline |
| `src/models/text_baseline.py` | Fine-tuned DeBERTa-v3 |
| `src/models/vlm_pipeline.py` | Zero-shot Qwen2.5-VL through Ollama |
| `src/models/cross_attention_model.py` | Prototype only: smoke-tested, not evaluated, no SAM yet |
| `scripts/run_experiments.py` | All experiments; writes `results/*.json` |
| `scripts/generate_doc.py` | Report and zip builder |
| `tests/test_pipeline.py` | Unit tests |

## Results

Micro F1 on the course test data, with article-level 95% intervals. No Subtask 2 model clearly beats the most-frequent baseline yet. See section 4 of the report.

| Subtask 2 (40 images) | Micro F1 |
|---|---|
| Most frequent label | 0.295 [0.169, 0.422] |
| Image LR (SigLIP) | 0.353 [0.269, 0.444] |
| Late fusion, image + text | 0.367 [0.248, 0.486] |
| Qwen2.5-VL, image only | 0.417 [0.292, 0.553] |
| Qwen2.5-VL, image + text (2a) | 0.333 [0.216, 0.463] |
| Qwen2.5-VL, image + text + gold frames (2b) | 0.340 [0.222, 0.468] |

| Subtask 1 (432 paragraphs) | Micro F1 |
|---|---|
| Most frequent label | 0.144 [0.108, 0.181] |
| TF-IDF + LR | 0.253 [0.217, 0.289] |
| Frozen DeBERTa-v3 + LR | 0.214 [0.184, 0.245] |
| Starter XLM-RoBERTa (3 seeds) | 0.320 ± 0.005 [0.294, 0.359] |
| Fine-tuned DeBERTa-v3, BCE (3 seeds) | 0.297 ± 0.015 [0.241, 0.322] |
| Fine-tuned DeBERTa-v3, ASL (3 seeds) | 0.278 ± 0.017 [0.230, 0.304] |

For 3-seed models, scores are mean ± std, and intervals use the seed-42 run. ASL seed 42 failed to train after epoch 2 and is kept in these results.
