"""
Exploratory C2 look of 2026-11-10, kept as a record (see docs/notes/2026-11-10-c2-prelock-test-look.md).
It reads test predictions and test labels before the results lock; do not use it to choose anything.
The planned C2 analysis lives in analysis/vlm_text_overlap.py (plan Task 10).
"""
import json, sys
from collections import Counter
import numpy as np
sys.path.insert(0, r"C:/UC-main/nlp_project")
sys.stdout.reconfigure(encoding="utf-8")
from src.data_loader import load_official_data, FRAME_TAXONOMY
from src.models.vlm_pipeline import MAX_TEXT_CHARS

_, docs = load_official_data()
by_id = {d["doc_id"]: d for d in docs if d["lang"] == "EN" and d["source"] == "second_chunk"}
split = json.load(open(r"C:/UC-main/nlp_project/results/course_split.json", encoding="utf-8"))

def text_stats(ids, name):
    lens = np.array([len(by_id[i]["text"]) for i in ids])
    title = np.array([len(by_id[i]["text"].split("\n", 1)[0]) for i in ids])
    cover = np.minimum(MAX_TEXT_CHARS / lens, 1)
    print(f"[{name}] n={len(ids)} article chars median={np.median(lens):.0f} min={lens.min()} max={lens.max()}; "
          f"truncated at {MAX_TEXT_CHARS}: {np.mean(lens > MAX_TEXT_CHARS):.0%}; median share seen={np.median(cover):.0%}; title chars median={np.median(title):.0f}")

def copy_text(ids, name):
    tp = fp = fn = 0
    for i in ids:
        g, t = set(by_id[i]["frames"]), set(by_id[i]["text_frames"])
        tp += len(g & t); fp += len(t - g); fn += len(g - t)
    print(f"[{name}] copy gold text frames -> precision={tp/(tp+fp):.3f} recall={tp/(tp+fn):.3f}; "
          f"mean text frames/article={np.mean([len(by_id[i]['text_frames']) for i in ids]):.2f}, "
          f"mean image frames/image={np.mean([len(by_id[i]['frames']) for i in ids]):.2f}")

def vlm(part):
    d = json.load(open(rf"C:/UC-main/nlp_project/results/vlm_{part}_metrics.json", encoding="utf-8"))
    ids = d["doc_order"]
    gold = [set(by_id[i]["frames"]) for i in ids]
    text = [set(by_id[i]["text_frames"]) for i in ids]
    preds = {}
    for v, s in d["variants"].items():
        preds[v] = [{FRAME_TAXONOMY[k] for k, x in enumerate(row) if x} for row in s["predictions"]]
    for v, P in preds.items():
        n = sum(len(p) for p in P); tp = sum(len(p & g) for p, g in zip(P, gold))
        in_text = sum(len(p & t) for p, t in zip(P, text))
        print(f"  {v:11} labels/img={n/len(P):.2f} empty={sum(not p for p in P)} precision={tp/max(n,1):.3f} "
              f"recall={tp/sum(len(g) for g in gold):.3f} share_of_preds_in_text_frames={in_text/max(n,1):.0%}")
    base = preds["image_only"]
    for v in ("subtask_2a", "subtask_2b"):
        added = [(f, f in g, f in t) for p, b, g, t in zip(preds[v], base, gold, text) for f in p - b]
        removed = [(f, f in g) for p, b, g in zip(preds[v], base, gold) for f in b - p]
        print(f"  image_only -> {v}: added {len(added)} labels (correct {sum(a[1] for a in added)}, in gold text frames {sum(a[2] for a in added)}); "
              f"removed {len(removed)} (of which correct {sum(r[1] for r in removed)})")
        print(f"     most added: {Counter(a[0] for a in added).most_common(4)}")
    gold_none = sum(not g for g in gold)
    print(f"  gold None images: {gold_none}")

text_stats(split["train"] + split["dev"], "train+dev")
text_stats(split["test"], "test")
copy_text([i for i in split["train"] + split["dev"]], "train+dev (second-batch EN)")
print("== VLM dev (20 images)"); vlm("dev")
print("== VLM test (40 images, analysis only)"); vlm("test")
