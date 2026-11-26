"""
Official Starter-Kit Baseline: Weighted XLM-RoBERTa (Subtask 1)
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Uses the settings of the starter kit's roberta-baseline.py: 8 epochs, lr 2e-5, batch 8 with
2 accumulation steps, 10% warmup, clipped positive weights, no "None" training rows, and a
0.5 inference threshold. One difference: the kit picks its checkpoint on a random 10% of
paragraphs, while this code picks it on the by-article dev set to avoid article leakage.
"""

import math
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import NUM_CLASSES, stack_labels
from src.features import DEVICE

MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 256
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
POS_WEIGHT_POWER = 0.5
POS_WEIGHT_MAX = 4.0
CHECKPOINT_THRESHOLD = 0.25
INFERENCE_THRESHOLD = 0.50
PATIENCE = 2
MIN_IMPROVEMENT = 0.001


def _positive_weights(labels):
    positives = labels.sum(axis=0).astype(np.float32)
    ratio = (len(labels) - positives) / np.maximum(positives, 1.0)
    return torch.tensor(np.clip(np.power(ratio, POS_WEIGHT_POWER), 1.0, POS_WEIGHT_MAX), dtype=torch.float32)


def _predict_probs(model, tokenizer, texts, batch_size=16):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            rows.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.vstack(rows)


def train_and_predict(train, dev, test, seed):
    """Returns (dev_probs, test_probs, summary) from the checkpoint with the best dev Macro F1."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train = [p for p in train if p["frames"]]
    texts = [p["text"] for p in train]
    y_train = stack_labels(train)
    y_dev = stack_labels(dev)
    pos_weight = _positive_weights(y_train).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, problem_type="multi_label_classification", dtype=torch.float32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(texts) / (BATCH_SIZE * ACCUMULATION_STEPS)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(WARMUP_FRACTION * total_steps)), total_steps)
    targets = torch.tensor(y_train, dtype=torch.float32)

    best = {"dev_macro_f1": -1.0}
    history = []
    stale_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(texts))
        total_loss = 0.0
        optimizer.zero_grad()
        for step, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
            idx = order[start:start + BATCH_SIZE].tolist()
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets[idx].to(DEVICE), pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            (loss / ACCUMULATION_STEPS).backward()
            total_loss += loss.item() * len(idx)
            if step % ACCUMULATION_STEPS == 0 or start + BATCH_SIZE >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        dev_probs = _predict_probs(model, tokenizer, [p["text"] for p in dev])
        dev_macro = float(f1_score(y_dev, (dev_probs >= CHECKPOINT_THRESHOLD).astype(int), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "dev_macro_f1": dev_macro})
        print(f"    XLM-R epoch {epoch}: loss={history[-1]['train_loss']:.4f} dev_macro@0.25={dev_macro:.3f}", flush=True)
        if dev_macro > best["dev_macro_f1"] + MIN_IMPROVEMENT:
            best = {"dev_macro_f1": dev_macro, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best["state"])
    dev_probs = _predict_probs(model, tokenizer, [p["text"] for p in dev])
    test_probs = _predict_probs(model, tokenizer, [p["text"] for p in test])
    del model
    torch.cuda.empty_cache()
    summary = {"dev_macro_f1": best["dev_macro_f1"], "epoch": best["epoch"], "threshold": INFERENCE_THRESHOLD}
    return dev_probs, test_probs, summary, history
