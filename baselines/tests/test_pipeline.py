"""
Unit Tests for the NLP Framing Analysis Pipeline
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang
"""

import hashlib
import os
import re
import sys
import unittest

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import NUM_CLASSES, clean_frames, load_official_data, split_article_paragraphs
from src.loss import AsymmetricLoss
from src.metrics import add_none_column, bootstrap_micro_difference, bootstrap_micro_f1, compute_multilabel_metrics, compute_mismatch_matrix
from src.models.cross_attention_model import RegionAwareCrossModalAttention
from src.models.linear_baselines import tfidf_fit_predict
from src.models.vlm_pipeline import build_prompt, parse_frames
from src.official_scorer import score_subtask1, score_subtask2
from src.splits import LINKED_ARTICLES, N_DEV_ARTICLES, N_TEST_ARTICLES, TRIAL_ARTICLES, make_course_split


class OfficialDataCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paragraphs, cls.documents = load_official_data()
        cls.second_en_docs = [d for d in cls.documents if d["lang"] == "EN" and d["source"] == "second_chunk"]
        cls.second_en_paragraphs = [p for p in cls.paragraphs if p["lang"] == "EN" and p["source"] == "second_chunk"]
        cls.split = make_course_split([d["doc_id"] for d in cls.second_en_docs])


class TestData(OfficialDataCase):
    def test_counts_match_release(self):
        self.assertEqual(NUM_CLASSES, 14)
        self.assertEqual(len(self.second_en_docs), 97)
        self.assertEqual(sum(1 for d in self.documents if d["source"] == "first_chunk"), 400)
        self.assertEqual(sum(1 for d in self.documents if d["source"] == "trial"), 5)
        self.assertEqual({p["lang"] for p in self.paragraphs if p["source"] == "trial"}, {"RU"})

    def test_no_framing_marks_are_empty(self):
        self.assertEqual(clean_frames(["-"]), [])
        self.assertEqual(clean_frames(["No framing", ""]), [])
        self.assertEqual(clean_frames(["Political", "No framing"]), ["Political"])
        self.assertFalse({"-", "No framing"} & {f for p in self.paragraphs for f in p["frames"]})

    def test_second_chunk_trial_labels_match_trial_release(self):
        by_id = {d["doc_id"]: d for d in self.second_en_docs}
        self.assertEqual(by_id["EN_TRIAL_5"]["frames"], ["Public_opinion"])
        self.assertEqual(by_id["EN_TRAIL_3"]["frames"], [])

    def test_every_image_file_exists(self):
        for doc in self.documents:
            self.assertTrue(os.path.exists(doc["image_path"]), doc["image_path"])

    def test_first_chunk_has_no_gold_text_frames(self):
        for doc in self.documents:
            self.assertEqual(doc["has_text_frames"], doc["source"] != "first_chunk")

    def test_article_paragraph_split(self):
        self.assertEqual(split_article_paragraphs("Title\n\nOne\r\n\r\nTwo\n"), ["Title", "One", "Two"])


class TestSplits(OfficialDataCase):
    def test_parts_are_disjoint_and_complete(self):
        parts = [set(self.split[k]) for k in ("train", "dev", "test")]
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        self.assertEqual(sum(map(len, parts)), 97)
        self.assertEqual(len(parts[1]), N_DEV_ARTICLES)
        self.assertEqual(len(parts[2]), N_TEST_ARTICLES)

    def test_trial_articles_stay_in_training(self):
        self.assertTrue(set(TRIAL_ARTICLES) <= set(self.split["train"]))

    def test_linked_articles_stay_together(self):
        for group in LINKED_ARTICLES:
            homes = [next(k for k, docs in self.split.items() if doc in docs) for doc in group]
            self.assertEqual(len(set(homes)), 1)

    def test_split_is_repeatable(self):
        ids = list(reversed([d["doc_id"] for d in self.second_en_docs]))
        self.assertEqual(make_course_split(ids), self.split)

    def test_no_exact_copy_across_train_and_held_out(self):
        def norm(text):
            return re.sub(r"\W+", " ", text.lower()).strip()

        def image_hash(doc):
            with open(doc["image_path"], "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        held_out = set(self.split["dev"] + self.split["test"])
        held_paragraphs = [p for p in self.paragraphs if p["doc_id"] in held_out]
        train_paragraphs = [p for p in self.paragraphs if p["doc_id"] not in held_out]
        held_docs = [d for d in self.documents if d["doc_id"] in held_out]
        train_docs = [d for d in self.documents if d["doc_id"] not in held_out]
        # Very short paragraphs, such as a repeated byline, are not treated as copies.
        held_texts = {norm(p["text"]) for p in held_paragraphs if len(norm(p["text"])) > 40}
        self.assertFalse(held_texts & {norm(p["text"]) for p in train_paragraphs})
        self.assertFalse({norm(d["text"]) for d in held_docs} & {norm(d["text"]) for d in train_docs})
        self.assertFalse({image_hash(d) for d in held_docs} & {image_hash(d) for d in train_docs})

    def test_tfidf_vocabulary_excludes_evaluation_text(self):
        y = np.zeros((4, NUM_CLASSES), dtype=int)
        y[:2, 0] = 1
        probs_a = tfidf_fit_predict(["aa bb", "aa cc", "dd ee", "dd ff"], y, [["zzz"]])[0]
        probs_b = tfidf_fit_predict(["aa bb", "aa cc", "dd ee", "dd ff"], y, [["qqq"]])[0]
        np.testing.assert_allclose(probs_a, probs_b)


class TestMetrics(OfficialDataCase):
    def _assert_agreement(self, scorer, records, tag):
        y_true = np.vstack([r["labels"] for r in records])
        rng = np.random.RandomState(0)
        y_pred = (rng.rand(*y_true.shape) < 0.1).astype(int)
        micro, macro = scorer(records, y_pred, tag)
        local = compute_multilabel_metrics(y_true, y_pred)
        self.assertAlmostEqual(micro, local["micro_f1"], places=9)
        self.assertAlmostEqual(macro, local["macro_f1"], places=9)

    def test_local_metrics_agree_with_official_scorers(self):
        self._assert_agreement(score_subtask1, self.second_en_paragraphs, "unit_test")
        self._assert_agreement(score_subtask2, self.second_en_docs, "unit_test")

    def test_none_label_is_scored(self):
        y_true = np.zeros((2, NUM_CLASSES), dtype=int)
        y_pred = np.zeros((2, NUM_CLASSES), dtype=int)
        y_true[0, 0] = 1
        y_pred[0, 0] = 1
        self.assertEqual(add_none_column(y_true)[:, -1].tolist(), [0, 1])
        self.assertEqual(compute_multilabel_metrics(y_true, y_pred)["micro_f1"], 1.0)

    def test_wrong_empty_prediction_is_penalized(self):
        y_true = np.zeros((1, NUM_CLASSES), dtype=int)
        y_pred = np.zeros((1, NUM_CLASSES), dtype=int)
        y_pred[0, 3] = 1
        self.assertEqual(compute_multilabel_metrics(y_true, y_pred)["micro_f1"], 0.0)

    def test_article_bootstrap_of_perfect_predictions(self):
        y = np.zeros((4, NUM_CLASSES), dtype=int)
        y[:2, 0] = 1
        self.assertEqual(bootstrap_micro_f1(y, y, ["a", "a", "b", "b"], n_boot=50), (1.0, 1.0))
        self.assertEqual(bootstrap_micro_difference(y, y, y, ["a", "a", "b", "b"], n_boot=50), (0.0, 0.0))

    def test_mismatch_matrix_shape(self):
        y_true = np.zeros((1, NUM_CLASSES), dtype=int)
        y_pred = np.zeros((1, NUM_CLASSES), dtype=int)
        y_true[0, 1] = 1
        y_pred[0, 2] = 1
        matrix = compute_mismatch_matrix(y_true, y_pred)
        self.assertEqual(matrix[1, 2], 1)


class TestVLM(unittest.TestCase):
    def setUp(self):
        self.doc = {"text": "ARTICLE BODY", "text_frames": ["Political"]}

    def test_only_2b_sees_gold_text_frames(self):
        self.assertNotIn("Human-annotated", build_prompt(self.doc, "subtask_2a"))
        self.assertIn("Human-annotated", build_prompt(self.doc, "subtask_2b"))
        self.assertNotIn("ARTICLE BODY", build_prompt(self.doc, "image_only"))

    def test_parser_rejects_bad_output(self):
        self.assertEqual(parse_frames('{"frames": ["Political", "Made_up"]}'), (["Political"], True))
        self.assertEqual(parse_frames("not json"), ([], False))


class TestModels(unittest.TestCase):
    def test_asymmetric_loss(self):
        logits = torch.randn(8, NUM_CLASSES, requires_grad=True)
        targets = (torch.rand(8, NUM_CLASSES) > 0.7).float()
        loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_cross_attention_prototype_forward(self):
        model = RegionAwareCrossModalAttention(text_dim=1024, vision_dim=768)
        batch = {"text_feat": torch.randn(2, 1024), "region_feat": torch.randn(2, 5, 768), "vision_feat": torch.randn(2, 768)}
        self.assertEqual(model(batch).shape, (2, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()
