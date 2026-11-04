"""
Course Data Split for SemEval-2027 Task 9 (English)
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

The official development set has no public labels, so course dev and test articles come
from the second training chunk, the only English chunk with paragraph labels, image labels,
and "None" examples. The split is a seeded shuffle of article ids; it never reads labels.

- The 5 trial articles stay in training, because earlier experiments already scored them.
- Articles that share an image or overlapping text always stay in the same part.
"""

import random

SPLIT_SEED = 2027
N_TEST_ARTICLES = 40
N_DEV_ARTICLES = 20
TRIAL_ARTICLES = ("EN_TRAIL_1", "EN_TRAIL_2", "EN_TRAIL_3", "EN_TRIAL_4", "EN_TRIAL_5")
LINKED_ARTICLES = (("EN_M_TRAIN_11", "EN_M_TRAIN_4"), ("EN_M_TRAIN_37", "EN_M_TRAIN_72"), ("EN_M_TRAIN_13", "EN_M_TRAIN_21"))


def make_course_split(article_ids):
    """Returns {"train": [...], "dev": [...], "test": [...]} of second-chunk English article ids."""
    linked = {doc: group for group in LINKED_ARTICLES for doc in group}
    groups = sorted({linked.get(doc, (doc,)) for doc in article_ids if doc not in TRIAL_ARTICLES})
    random.Random(SPLIT_SEED).shuffle(groups)

    split = {"train": [doc for doc in article_ids if doc in TRIAL_ARTICLES], "dev": [], "test": []}
    for group in groups:
        if len(split["test"]) + len(group) <= N_TEST_ARTICLES:
            split["test"] += group
        elif len(split["dev"]) + len(group) <= N_DEV_ARTICLES:
            split["dev"] += group
        else:
            split["train"] += group
    return {part: sorted(docs) for part, docs in split.items()}
