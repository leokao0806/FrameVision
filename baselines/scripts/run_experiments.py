"""
Experiment Runner on the Official SemEval-2027 Task 9 Training Data
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Usage: python scripts/run_experiments.py [subtask1] [subtask2] [vlm_dev] [vlm_test] [analysis] [compare]

- subtask1: text baselines on the by-article split, including the starter-kit baseline.
- subtask2: supervised image baselines on the same articles.
- vlm_dev:  zero-shot VLM on the English dev images.
- vlm_test: zero-shot VLM on the English test images, with the prompt left unchanged.
- analysis: leakage demonstration that never touches the course test articles.
- compare:  paired bootstrap comparisons of the saved predictions.

Every test score comes from the unmodified official scorer and is checked against metrics.py.
Finished neural runs are saved under results/runs, so an interrupted experiment can resume.
"""

import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.model_selection import GroupKFold, KFold

from src.data_loader import FRAME_TAXONOMY, load_official_data, split_article_paragraphs, stack_labels, NUM_CLASSES
from src.features import encode_texts, extract_image_features
from src.loss import AsymmetricLoss, WeightedBCELoss
from src.metrics import bootstrap_micro_difference, bootstrap_micro_f1, compute_mismatch_matrix, compute_multilabel_metrics, select_threshold_on_dev
from src.models.linear_baselines import (
    fit_predict_ovr,
    most_frequent_label_predict,
    sample_label_prior,
    tfidf_fit_predict,
)
from src.official_scorer import score_subtask1, score_subtask2
from src.splits import make_course_split

RESULTS_DIR = os.path.join(project_root, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
SEEDS = [13, 42, 2026]
N_RANDOM_DRAWS = 1000
SCORE_TOLERANCE = 1e-9


def save_json(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def cached_run(name, compute):
    """Runs compute() once and keeps its JSON-ready result, so a rerun can skip finished work."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    result = compute()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


def score_with_official(scorer, records, y_true, y_pred, tag):
    """Official scores, plus our diagnostics; stops if the two implementations ever disagree."""
    metrics = compute_multilabel_metrics(y_true, y_pred)
    micro, macro = scorer(records, y_pred, tag)
    if abs(micro - metrics["micro_f1"]) > SCORE_TOLERANCE or abs(macro - metrics["macro_f1"]) > SCORE_TOLERANCE:
        raise RuntimeError(f"Official and local scores disagree for {tag}: {micro}, {macro}, {metrics['micro_f1']}, {metrics['macro_f1']}")
    return metrics


def mean_random_metrics(y_train, y_test):
    rng = np.random.RandomState(0)
    draws = [compute_multilabel_metrics(y_test, sample_label_prior(y_train, len(y_test), rng))
             for _ in range(N_RANDOM_DRAWS)]
    return {key: float(np.mean([d[key] for d in draws])) for key in ("micro_f1", "macro_f1", "macro_f1_present", "hamming_loss")}


def course_split(documents):
    ids = [d["doc_id"] for d in documents if d["lang"] == "EN" and d["source"] == "second_chunk"]
    split = make_course_split(ids)
    save_json("course_split.json", split)
    return split


def by_ids(records, ids):
    ids = set(ids)
    return [r for r in records if r["doc_id"] in ids]


def summarize_seeds(runs):
    summary = {key: float(np.mean([r[key] for r in runs])) for key in ("micro_f1", "macro_f1", "macro_f1_present", "hamming_loss", "dev_micro_f1")}
    summary["micro_f1_std"] = float(np.std([r["micro_f1"] for r in runs]))
    summary["macro_f1_std"] = float(np.std([r["macro_f1"] for r in runs]))
    summary["runs"] = [{k: v for k, v in r.items() if k != "per_label"} for r in runs]
    return summary


def run_subtask1(paragraphs, documents):
    from src.models import starter_baseline
    from src.models.text_baseline import train_and_predict

    split = course_split(documents)
    second_en = [p for p in paragraphs if p["lang"] == "EN" and p["source"] == "second_chunk"]
    train_second = by_ids(second_en, split["train"])
    dev, test = by_ids(second_en, split["dev"]), by_ids(second_en, split["test"])
    train = train_second + [p for p in paragraphs if p["lang"] == "EN" and p["source"] == "semeval23"]
    held_out = set(split["dev"] + split["test"])
    multilingual_train = [p for p in paragraphs if p["doc_id"] not in held_out]
    assert not held_out & {p["doc_id"] for p in train + multilingual_train}

    y_train, y_dev, y_test = stack_labels(train), stack_labels(dev), stack_labels(test)
    groups = [p["doc_id"] for p in test]
    print(f"Subtask 1 split: train={len(train)} multilingual_train={len(multilingual_train)} dev={len(dev)} test={len(test)} paragraphs", flush=True)

    results = {"split": {"train_en": len(train), "train_en_second_chunk": len(train_second), "train_multilingual": len(multilingual_train),
                         "dev": len(dev), "test": len(test), "test_none": sum(1 for p in test if not p["frames"]),
                         "dev_articles": len(split["dev"]), "test_articles": len(split["test"])}, "models": {}}
    predictions = {}

    def add_single(name, tag, preds, extra=None):
        metrics = score_with_official(score_subtask1, test, y_test, preds, tag)
        metrics["micro_f1_ci95"] = bootstrap_micro_f1(y_test, preds, groups)
        results["models"][name] = {**metrics, **(extra or {})}
        predictions[name] = np.asarray(preds).tolist()
        print(f"  {name}: test micro={metrics['micro_f1']:.3f} macro={metrics['macro_f1']:.3f} {extra or ''}", flush=True)

    results["models"]["Random (training prior)"] = mean_random_metrics(y_train, y_test)
    add_single("Most frequent label", "most_frequent", most_frequent_label_predict(y_train, len(test)))
    # Predicting no frame is scored as the None label, which is common in the second chunk.
    add_single("Always None", "always_none", np.zeros_like(y_test),
               {"dev_micro_f1": compute_multilabel_metrics(y_dev, np.zeros_like(y_dev))["micro_f1"]})

    dev_probs, test_probs = tfidf_fit_predict([p["text"] for p in train], y_train,
                                              [[p["text"] for p in dev], [p["text"] for p in test]])
    threshold, dev_f1 = select_threshold_on_dev(dev_probs, y_dev)
    add_single("TF-IDF + LR", "tfidf", (test_probs >= threshold).astype(int), {"threshold": threshold, "dev_micro_f1": dev_f1})

    encoded = train + dev + test
    embeddings = encode_texts([p["text"] for p in encoded], "deberta_official_paragraphs")
    x_train, x_dev, x_test = np.split(embeddings, [len(train), len(train) + len(dev)])
    all_probs = fit_predict_ovr(x_train, y_train, np.vstack([x_dev, x_test]))
    dev_probs, test_probs = all_probs[:len(dev)], all_probs[len(dev):]
    threshold, dev_f1 = select_threshold_on_dev(dev_probs, y_dev)
    add_single("Frozen DeBERTa-v3 + LR", "frozen_deberta", (test_probs >= threshold).astype(int), {"threshold": threshold, "dev_micro_f1": dev_f1})

    def seed_runs(name, slug, train_one):
        runs, preds_by_seed = [], []
        for seed in SEEDS:
            run = cached_run(f"st1_{slug}_{seed}", lambda s=seed: train_one(s))
            preds = np.array(run["test_preds"])
            metrics = score_with_official(score_subtask1, test, y_test, preds, f"{slug}_{seed}")
            runs.append({"seed": seed, **run["summary"], **metrics})
            preds_by_seed.append(preds)
            print(f"  {name} seed={seed}: dev={run['summary']['dev_micro_f1']:.3f} test={metrics['micro_f1']:.3f}", flush=True)
        results["models"][name] = summarize_seeds(runs)
        # The seed-42 run represents the model in the article-level bootstrap and comparisons.
        middle = preds_by_seed[SEEDS.index(42)]
        results["models"][name]["micro_f1_ci95_seed42"] = bootstrap_micro_f1(y_test, middle, groups)
        predictions[name] = middle.tolist()
        return preds_by_seed

    def fine_tune(loss_fn):
        def train_one(seed):
            preds, summary, history = train_and_predict(train, dev, test, loss_fn, seed)
            return {"test_preds": preds.tolist(), "summary": summary, "history": history}
        return train_one

    def starter(seed):
        dev_probs, test_probs, summary, history = starter_baseline.train_and_predict(multilingual_train, dev, test, seed)
        dev_preds = (dev_probs >= summary["threshold"]).astype(int)
        summary["dev_micro_f1"] = compute_multilabel_metrics(y_dev, dev_preds)["micro_f1"]
        return {"test_preds": (test_probs >= summary["threshold"]).astype(int).tolist(), "summary": summary, "history": history}

    all_seed_preds = {
        "Starter XLM-RoBERTa": seed_runs("Starter XLM-RoBERTa", "starter", starter),
        "Fine-tuned DeBERTa-v3 (BCE)": seed_runs("Fine-tuned DeBERTa-v3 (BCE)", "deberta_bce", fine_tune(WeightedBCELoss())),
        "Fine-tuned DeBERTa-v3 (ASL)": seed_runs("Fine-tuned DeBERTa-v3 (ASL)", "deberta_asl", fine_tune(AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05))),
    }

    # The model for the per-label figure is chosen by dev Micro F1 only.
    candidates = [n for n, m in results["models"].items() if "dev_micro_f1" in m]
    selected = max(candidates, key=lambda n: results["models"][n]["dev_micro_f1"])
    results["selected_on_dev"] = selected
    selected_preds = all_seed_preds.get(selected, [np.array(predictions[selected])])
    per_seed = [compute_multilabel_metrics(y_test, p)["per_label"] for p in selected_preds]
    results["selected_per_label_mean_f1"] = {label: float(np.mean([s[label]["f1"] for s in per_seed])) for label in per_seed[0]}
    results["selected_support"] = {label: v["support"] for label, v in per_seed[0].items()}
    results["tfidf_per_label"] = results["models"]["TF-IDF + LR"]["per_label"]
    results["mismatch_matrix"] = compute_mismatch_matrix(y_test, selected_preds[0]).tolist()
    save_json("subtask1_metrics.json", results)
    by_seed = {name: {str(seed): preds.tolist() for seed, preds in zip(SEEDS, seed_preds)} for name, seed_preds in all_seed_preds.items()}
    save_json("subtask1_predictions.json", {"doc_ids": groups, "gold": y_test.tolist(), "predictions": predictions, "by_seed": by_seed})


def run_subtask2(documents):
    split = course_split(documents)
    held_out = set(split["dev"] + split["test"])
    second_en = [d for d in documents if d["lang"] == "EN" and d["source"] == "second_chunk"]
    dev, test = by_ids(second_en, split["dev"]), by_ids(second_en, split["test"])
    train_all = [d for d in documents if d["doc_id"] not in held_out]
    train_en = [d for d in train_all if d["lang"] == "EN"]
    train_frames = [d for d in train_all if d["has_text_frames"]]
    assert not held_out & {d["doc_id"] for d in train_all}
    y_dev, y_test = stack_labels(dev), stack_labels(test)
    print(f"Subtask 2 split: train_images={len(train_all)} train_en_articles={len(train_en)} train_with_text_frames={len(train_frames)} dev={len(dev)} test={len(test)}", flush=True)

    image_feats = extract_image_features(documents)
    en_docs = [d for d in documents if d["lang"] == "EN"]
    pieces = [split_article_paragraphs(d["text"]) for d in en_docs]
    flat = encode_texts([text for doc_pieces in pieces for text in doc_pieces], "deberta_official_articles")
    # An article's text vector is the mean of its paragraph vectors. DeBERTa-v3 is an
    # English model, so the text channel only trains on the EN articles.
    bounds = np.cumsum([len(doc_pieces) for doc_pieces in pieces])[:-1]
    text_feats = {d["doc_id"]: part.mean(axis=0) for d, part in zip(en_docs, np.split(flat, bounds))}

    def stack(feats, records):
        return np.vstack([feats[d["doc_id"]] for d in records])

    def channel_probs(x_train, y_train, make_x):
        probs = fit_predict_ovr(x_train, y_train, np.vstack([make_x(dev), make_x(test)]))
        return probs[:len(dev)], probs[len(dev):]

    image_probs = channel_probs(stack(image_feats, train_all), stack_labels(train_all), lambda r: stack(image_feats, r))
    text_probs = channel_probs(stack(text_feats, train_en), stack_labels(train_en), lambda r: stack(text_feats, r))
    frame_matrix = lambda records: np.vstack([d["text_frame_labels"] for d in records]).astype(float)
    frame_probs = channel_probs(frame_matrix(train_frames), stack_labels(train_frames), frame_matrix)

    results = {"n_test_images": len(test), "n_dev_images": len(dev), "n_train_images": len(train_all),
               "n_train_en_articles": len(train_en), "n_train_with_text_frames": len(train_frames),
               "test_gold_labels": int(y_test.sum()), "test_none": int((y_test.sum(axis=1) == 0).sum()),
               "doc_order": [d["doc_id"] for d in test], "models": {}, "predictions": {}, "gold": y_test.tolist(),
               "dev_doc_order": [d["doc_id"] for d in dev], "dev_predictions": {}}
    groups = [d["doc_id"] for d in test]

    def add(name, tag, preds, extra=None, dev_preds=None):
        metrics = score_with_official(score_subtask2, test, y_test, preds, tag)
        metrics["micro_f1_ci95"] = bootstrap_micro_f1(y_test, preds, groups)
        results["models"][name] = {**metrics, **(extra or {})}
        results["predictions"][name] = np.asarray(preds).tolist()
        results["dev_predictions"][name] = np.asarray(dev_preds).astype(int).tolist()
        print(f"  {name}: test micro={metrics['micro_f1']:.3f} macro={metrics['macro_f1']:.3f} {extra or ''}", flush=True)

    def add_thresholded(name, tag, probs):
        # The dev predictions use the same threshold that was selected on dev.
        threshold, dev_f1 = select_threshold_on_dev(probs[0], y_dev)
        add(name, tag, (probs[1] >= threshold).astype(int), {"threshold": threshold, "dev_micro_f1": dev_f1},
            dev_preds=(probs[0] >= threshold).astype(int))

    results["models"]["Random (training prior)"] = mean_random_metrics(stack_labels(train_all), y_test)
    add("Most frequent label", "most_frequent", most_frequent_label_predict(stack_labels(train_all), len(test)),
        dev_preds=most_frequent_label_predict(stack_labels(train_all), len(dev)))
    add("Copy text frames (2b)", "copy_frames", frame_matrix(test).astype(int), dev_preds=frame_matrix(dev).astype(int))
    add_thresholded("Image LR (SigLIP)", "image_lr", image_probs)
    add_thresholded("Text LR (DeBERTa-v3)", "text_lr", text_probs)
    add_thresholded("Late fusion (image + text)", "late_fusion", tuple((a + b) / 2 for a, b in zip(image_probs, text_probs)))
    add_thresholded("Late fusion + text frames (2b)", "late_fusion_frames",
                    tuple((a + b + c) / 3 for a, b, c in zip(image_probs, text_probs, frame_probs)))
    results["most_frequent_label_name"] = FRAME_TAXONOMY[int(np.argmax(stack_labels(train_all).sum(axis=0)))]
    save_json("subtask2_metrics.json", results)


def run_vlm(documents, part, out_name):
    from src.models.vlm_pipeline import VARIANTS, query_vlm

    split = course_split(documents)
    second_en = [d for d in documents if d["lang"] == "EN" and d["source"] == "second_chunk"]
    selected = by_ids(second_en, split[part])
    y_true = stack_labels(selected)
    groups = [d["doc_id"] for d in selected]
    replies, summary = [], {}
    for variant in VARIANTS:
        rows = []
        for doc in selected:
            reply = cached_run(f"vlm_{variant}_{doc['doc_id']}", lambda: query_vlm(doc, variant))
            replies.append(reply)
            rows.append(reply["labels"])
        preds = np.array(rows)
        summary[variant] = {
            **score_with_official(score_subtask2, selected, y_true, preds, f"vlm_{part}_{variant}"),
            "micro_f1_ci95": bootstrap_micro_f1(y_true, preds, groups),
            "predictions": rows,
            "parse_failures": sum(1 for r in replies if r["variant"] == variant and not r["parse_ok"]),
        }
        print(f"  VLM {part} {variant}: micro={summary[variant]['micro_f1']:.3f} macro={summary[variant]['macro_f1']:.3f} parse_failures={summary[variant]['parse_failures']}", flush=True)
    save_json(out_name, {"doc_order": groups, "gold": y_true.tolist(), "variants": summary, "raw_replies": replies})


def run_leakage_analysis(paragraphs, documents):
    """Compares random-paragraph and by-article folds on non-test second-chunk EN paragraphs."""
    split = course_split(documents)
    rows = by_ids([p for p in paragraphs if p["lang"] == "EN" and p["source"] == "second_chunk"], split["train"] + split["dev"])
    assert not {p["doc_id"] for p in rows} & set(split["test"])
    texts = np.array([p["text"] for p in rows], dtype=object)
    labels = stack_labels(rows)
    docs = np.array([p["doc_id"] for p in rows])

    def pooled(folds):
        true_rows, pred_rows = [], []
        for train_idx, test_idx in folds:
            probs = tfidf_fit_predict(texts[train_idx].tolist(), labels[train_idx], [texts[test_idx].tolist()])[0]
            true_rows.append(labels[test_idx])
            pred_rows.append((probs >= 0.5).astype(int))
        return compute_multilabel_metrics(np.vstack(true_rows), np.vstack(pred_rows))

    random_metrics = pooled(list(KFold(n_splits=5, shuffle=True, random_state=42).split(texts)))
    article_metrics = pooled(list(GroupKFold(n_splits=5).split(texts, groups=docs)))
    save_json("leakage_analysis.json", {
        "n_paragraphs": len(rows),
        "n_articles": len(set(docs)),
        "random_paragraph_split": {k: random_metrics[k] for k in ("micro_f1", "macro_f1")},
        "by_article_split": {k: article_metrics[k] for k in ("micro_f1", "macro_f1")},
    })
    print("Leakage analysis:", random_metrics["micro_f1"], article_metrics["micro_f1"], flush=True)


def run_comparisons():
    """Paired article-level intervals for the comparisons discussed in the report."""
    st1 = load_json("subtask1_predictions.json")
    st2 = load_json("subtask2_metrics.json")
    vlm = load_json("vlm_test_metrics.json")
    assert st2["doc_order"] == vlm["doc_order"]
    st2_preds = {**st2["predictions"], **{f"VLM {k}": v["predictions"] for k, v in vlm["variants"].items()}}

    def interval(source, gold, groups, a, b):
        low, high = bootstrap_micro_difference(np.array(gold), np.array(source[a]), np.array(source[b]), groups)
        return {"a": a, "b": b, "ci95_low": low, "ci95_high": high}

    st1_names = st1["predictions"]
    st1_pairs = [(a, "Most frequent label") for a in st1_names if a != "Most frequent label"]
    st1_pairs += [(a, "Always None") for a in st1_names if a not in ("Always None", "Most frequent label")]
    st1_pairs += [("Fine-tuned DeBERTa-v3 (ASL)", "Fine-tuned DeBERTa-v3 (BCE)"), ("Fine-tuned DeBERTa-v3 (BCE)", "Starter XLM-RoBERTa"),
                  ("Fine-tuned DeBERTa-v3 (BCE)", "Frozen DeBERTa-v3 + LR")]
    st2_pairs = [(a, "Most frequent label") for a in st2_preds if a != "Most frequent label"]
    st2_pairs += [("VLM subtask_2a", "VLM image_only"), ("VLM subtask_2b", "VLM image_only"), ("VLM subtask_2b", "VLM subtask_2a"),
                  ("VLM image_only", "Image LR (SigLIP)"), ("Late fusion (image + text)", "Image LR (SigLIP)"),
                  ("Late fusion + text frames (2b)", "Late fusion (image + text)")]
    # Every seed is compared separately, because one seed-42 run can misrepresent a model.
    by_seed_pairs = [(a, "Always None") for a in st1["by_seed"]]
    by_seed_pairs += [("Fine-tuned DeBERTa-v3 (BCE)", "Starter XLM-RoBERTa"), ("Fine-tuned DeBERTa-v3 (ASL)", "Fine-tuned DeBERTa-v3 (BCE)")]
    by_seed = []
    for seed in SEEDS:
        source = {name: preds[str(seed)] for name, preds in st1["by_seed"].items()}
        source["Always None"] = st1_names["Always None"]
        by_seed += [{"seed": seed, **interval(source, st1["gold"], st1["doc_ids"], a, b)} for a, b in by_seed_pairs]
    save_json("comparisons.json", {
        "subtask1": [interval(st1_names, st1["gold"], st1["doc_ids"], a, b) for a, b in st1_pairs],
        "subtask1_by_seed": by_seed,
        "subtask2": [interval(st2_preds, st2["gold"], st2["doc_order"], a, b) for a, b in st2_pairs],
    })


def main():
    steps = sys.argv[1:] or ["subtask1", "subtask2", "vlm_dev", "vlm_test", "analysis", "compare"]
    paragraphs, documents = load_official_data()
    print(f"Loaded {len(paragraphs)} paragraphs and {len(documents)} documents; {NUM_CLASSES} frames", flush=True)
    if "subtask1" in steps:
        run_subtask1(paragraphs, documents)
    if "subtask2" in steps:
        run_subtask2(documents)
    if "vlm_dev" in steps:
        run_vlm(documents, "dev", "vlm_dev_metrics.json")
    if "vlm_test" in steps:
        run_vlm(documents, "test", "vlm_test_metrics.json")
    if "analysis" in steps:
        run_leakage_analysis(paragraphs, documents)
    if "compare" in steps:
        run_comparisons()


if __name__ == "__main__":
    main()
