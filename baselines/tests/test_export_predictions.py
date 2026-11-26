"""Tests for scripts/export_predictions.py, using small in-memory records instead of the official data."""

import json
import os
import shutil
import sys
import tempfile
import unittest

scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import export_predictions as ep  # noqa: E402
from src.data_loader import FRAME_TAXONOMY, labels_to_vector  # noqa: E402

ECON, POL = FRAME_TAXONOMY[0], FRAME_TAXONOMY[-1]


def _doc(doc_id, frames, text_frames, has_text_frames=True, ext=".jpg"):
    return {"doc_id": doc_id, "image_name": doc_id + ext, "frames": frames, "labels": labels_to_vector(frames),
            "has_text_frames": has_text_frames, "text_frame_labels": labels_to_vector(text_frames)}


def _vec(frames):
    return [int(v) for v in labels_to_vector(frames)]


class ExportCase(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.docs = [
            _doc("T1", [ECON], [ECON, POL]), _doc("T2", [], []),
            _doc("D1", [POL], [POL], ext=".jpeg"),
            _doc("R1", [ECON], [ECON]), _doc("B1", [POL], [], has_text_frames=False),
        ]
        self.split = {"train": ["R1"], "dev": ["D1"], "test": ["T1", "T2"]}

    def tearDown(self):
        shutil.rmtree(self.out)

    def read(self, *parts):
        with open(os.path.join(self.out, *parts), encoding="utf-8", newline="") as f:
            return f.read()

    def test_gold_files_and_train_keeps_only_text_frame_articles(self):
        ep.export_gold(self.docs, self.split, self.out)
        self.assertEqual(self.read("_gold", "labels.txt"), "\n".join(FRAME_TAXONOMY) + "\n")
        self.assertEqual(self.read("_gold", "test.tsv"), f"T1.jpg\t{ECON}\nT2.jpg\n")
        self.assertEqual(self.read("_gold", "dev.tsv"), f"D1.jpeg\t{POL}\n")
        self.assertEqual(self.read("_gold", "train.tsv"), f"R1.jpg\t{ECON}\n")
        self.assertEqual(self.read("_gold", "test_text_frames.tsv"), f"T1.jpg\t{ECON}\t{POL}\nT2.jpg\n")
        self.assertEqual(self.read("_gold", "train_text_frames.tsv"), f"R1.jpg\t{ECON}\n")

    def test_baseline_predictions_use_fixed_names_and_doc_order(self):
        st2 = {"doc_order": ["T2", "T1"], "dev_doc_order": ["D1"],
               "predictions": {"Image LR (SigLIP)": [_vec([POL]), _vec([])]},
               "dev_predictions": {"Image LR (SigLIP)": [_vec([ECON, POL])]}}
        written = ep.export_baselines(st2, {d["doc_id"]: d for d in self.docs}, self.out)
        self.assertEqual(written, ["image_lr"])
        self.assertEqual(self.read("image_lr", "seed0", "test.tsv"), f"T2.jpg\t{POL}\nT1.jpg\n")
        self.assertEqual(self.read("image_lr", "seed0", "dev.tsv"), f"D1.jpeg\t{ECON}\t{POL}\n")

    def test_missing_dev_predictions_says_to_rerun_subtask2(self):
        st2 = {"doc_order": ["T1"], "predictions": {"Most frequent label": [_vec([POL])]}}
        with self.assertRaisesRegex(KeyError, "run_experiments.py subtask2"):
            ep.export_baselines(st2, {d["doc_id"]: d for d in self.docs}, self.out)

    def test_vlm_variants_map_to_qwen_names(self):
        vlm = {"dev": {"doc_order": ["D1"], "variants": {"image_only": {"predictions": [_vec([])]}}},
               "test": {"doc_order": ["T1", "T2"], "variants": {"image_only": {"predictions": [_vec([ECON]), _vec([])]}}}}
        written = ep.export_vlm(vlm, {d["doc_id"]: d for d in self.docs}, self.out)
        self.assertEqual(written, ["qwen_image_only"])
        self.assertEqual(self.read("qwen_image_only", "seed0", "dev.tsv"), "D1.jpeg\n")
        self.assertEqual(self.read("qwen_image_only", "seed0", "test.tsv"), f"T1.jpg\t{ECON}\nT2.jpg\n")

    def test_random_prior_is_copied_verbatim(self):
        prior = {"micro_f1": 0.123, "macro_f1": 0.05, "macro_f1_present": 0.1, "hamming_loss": 0.2}
        ep.export_random_prior({"models": {"Random (training prior)": prior}}, self.out)
        self.assertEqual(json.loads(self.read("_reference", "random_prior.json")), prior)

    def test_every_name_is_a_planned_model_folder(self):
        self.assertEqual(sorted(ep.MODEL_NAMES.values()) + sorted(ep.VLM_NAMES.values()),
                         ["copy_text_frames_2b", "image_lr", "late_fusion_2a", "late_fusion_2b", "most_frequent",
                          "text_lr", "qwen_2a", "qwen_2b", "qwen_image_only", "qwen_image_only_v2"])


if __name__ == "__main__":
    unittest.main()
