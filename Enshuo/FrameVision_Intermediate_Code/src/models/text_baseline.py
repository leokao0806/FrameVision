"""
Fine-Tuned DeBERTa-v3 Paragraph Classifier (Subtask 1 text baseline)
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang
"""

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.data_loader import NUM_CLASSES
from src.features import DEBERTA_NAME, DEVICE
from src.metrics import select_threshold_on_dev


class TextFramingBaseline(nn.Module):
    def __init__(self, model_name=DEBERTA_NAME, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        # float32 is required: the checkpoint's float16 weights turn Adam's eps into 0.
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def _predict_probs(model, tokenizer, texts, batch_size=8, max_length=256):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(DEVICE)
            rows.append(torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])).cpu().numpy())
    return np.vstack(rows)


def train_and_predict(train, dev, test, loss_fn, seed, epochs=10, lr=2e-5, batch_size=4, max_length=256):
    """
    Fine-tunes on train, keeps the epoch and threshold with the best dev Micro F1, and
    only then predicts on test. Returns test predictions and the training history.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    model = TextFramingBaseline().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_texts = [p["text"] for p in train]
    Y_train = torch.tensor(np.vstack([p["labels"] for p in train]), dtype=torch.float32)
    dev_texts = [p["text"] for p in dev]
    Y_dev = np.vstack([p["labels"] for p in dev])

    best = {"dev_micro_f1": -1.0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_texts))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size].tolist()
            enc = tokenizer([train_texts[i] for i in idx], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(enc["input_ids"], enc["attention_mask"]), Y_train[idx].to(DEVICE))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)

        threshold, dev_f1 = select_threshold_on_dev(_predict_probs(model, tokenizer, dev_texts), Y_dev)
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "dev_micro_f1": dev_f1})
        print(f"    epoch {epoch}: loss={history[-1]['train_loss']:.4f} dev_micro={dev_f1:.3f}", flush=True)
        if dev_f1 > best["dev_micro_f1"]:
            best = {
                "dev_micro_f1": dev_f1,
                "threshold": threshold,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    model.load_state_dict(best["state"])
    test_probs = _predict_probs(model, tokenizer, [p["text"] for p in test])
    test_preds = (test_probs >= best["threshold"]).astype(int)

    del model
    torch.cuda.empty_cache()
    summary = {k: best[k] for k in ("dev_micro_f1", "threshold", "epoch")}
    return test_preds, summary, history
