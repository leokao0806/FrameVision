"""
Tests for scripts/data_manifest.py, the data-copy checker shared with Leo.
CS 5134/6034 NLP Fall 2026 - Team FrameVision
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(project_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import data_manifest as dm  # noqa: E402

EN = "training-second-batch-oct/EN"
B1 = "initial-training-data/subtask2"
LABELS = "Economic\nPolitical\nMorality"


def _write(root, rel, data):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def _make_release(root):
    _write(root, f"{EN}/subtask-2-annotations.txt", "EN_M_TRAIN_1.jpg\tPolitical\n")
    _write(root, f"{EN}/raw-documents/EN_M_TRAIN_1.jpg", b"\xff\xd8image-one")
    _write(root, f"{EN}/raw-documents/EN_M_TRAIN_1.txt", "Title\n\nBody text.\n")
    _write(root, f"{B1}/subtask-2-annotations.txt", "EN_doc_1_image.jpg\tEconomic\n")
    _write(root, f"{B1}/images/EN_doc_1_image.jpg", b"\xff\xd8image-two")
    _write(root, "initial-training-data/subtask1/README_ST1.md", "readme\n")
    _write(root, "labels.txt", LABELS)


class DataManifestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ref = os.path.join(self.tmp, "ref")
        self.other = os.path.join(self.tmp, "other")
        _make_release(self.ref)
        _make_release(self.other)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _compare(self):
        return dm.compare_manifests(dm.build_manifest(self.ref), dm.build_manifest(self.other))

    def test_identical_trees_pass(self):
        result = self._compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["critical_failures"], [])
        self.assertEqual(len(result["identical"]), 6)
        self.assertEqual(result["missing"] + result["extra"] + result["content"] + result["text_only"], [])
        self.assertEqual(result["labels"]["status"], "same")

    def test_missing_critical_file_fails(self):
        os.remove(os.path.join(self.other, *f"{EN}/raw-documents/EN_M_TRAIN_1.txt".split("/")))
        result = self._compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing"], [f"{EN}/raw-documents/EN_M_TRAIN_1.txt"])
        self.assertIn(f"{EN}/raw-documents/EN_M_TRAIN_1.txt", result["critical_failures"])

    def test_changed_image_fails(self):
        _write(self.other, f"{B1}/images/EN_doc_1_image.jpg", b"\xff\xd8re-encoded")
        result = self._compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["content"], [f"{B1}/images/EN_doc_1_image.jpg"])

    def test_line_ending_and_bom_difference_is_only_a_warning(self):
        _write(self.other, f"{EN}/raw-documents/EN_M_TRAIN_1.txt", b"\xef\xbb\xbfTitle\r\n\r\nBody text.\r\n")
        result = self._compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["text_only"], [f"{EN}/raw-documents/EN_M_TRAIN_1.txt"])
        self.assertTrue(result["warnings"])

    def test_non_critical_missing_file_does_not_fail(self):
        os.remove(os.path.join(self.other, "initial-training-data", "subtask1", "README_ST1.md"))
        result = self._compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["missing"], ["initial-training-data/subtask1/README_ST1.md"])

    def test_extra_file_in_other_is_reported_not_failed(self):
        _write(self.other, f"{EN}/raw-documents/notes.txt", "mine\n")
        result = self._compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["extra"], [f"{EN}/raw-documents/notes.txt"])

    def test_moved_file_is_paired_and_still_fails(self):
        src = os.path.join(self.other, *f"{EN}/raw-documents/EN_M_TRAIN_1.jpg".split("/"))
        dst = os.path.join(self.other, "EN_M_TRAIN_1.jpg")
        shutil.move(src, dst)
        result = self._compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["moved"], [[f"{EN}/raw-documents/EN_M_TRAIN_1.jpg", "EN_M_TRAIN_1.jpg"]])

    def test_os_junk_files_are_ignored(self):
        _write(self.other, ".DS_Store", b"junk")
        _write(self.other, f"{EN}/raw-documents/._EN_M_TRAIN_1.jpg", b"junk")
        _write(self.other, "__MACOSX/anything.txt", b"junk")
        _write(self.other, "Thumbs.db", b"junk")
        result = self._compare()
        self.assertEqual(result["extra"], [])
        self.assertTrue(result["passed"])

    def test_labels_trailing_newline_is_same(self):
        _write(self.other, "labels.txt", LABELS + "\r\n")
        result = self._compare()
        self.assertEqual(result["labels"]["status"], "same")
        self.assertTrue(result["passed"])

    def test_labels_other_order_is_a_warning(self):
        _write(self.other, "labels.txt", "Political\nEconomic\nMorality\n")
        result = self._compare()
        self.assertEqual(result["labels"]["status"], "order")
        self.assertTrue(result["passed"])

    def test_labels_different_names_fail(self):
        _write(self.other, "labels.txt", "Economic\nPolitical\nNone\n")
        result = self._compare()
        self.assertEqual(result["labels"]["status"], "different")
        self.assertEqual(result["labels"]["only_ref"], ["Morality"])
        self.assertEqual(result["labels"]["only_other"], ["None"])
        self.assertFalse(result["passed"])

    def test_labels_file_outside_root(self):
        os.remove(os.path.join(self.ref, "labels.txt"))
        _write(self.tmp, "trial/labels.txt", LABELS)
        manifest = dm.build_manifest(self.ref, labels_path=os.path.join(self.tmp, "trial", "labels.txt"))
        self.assertEqual(manifest["labels"], ["Economic", "Political", "Morality"])
        self.assertNotIn("labels.txt", manifest["files"])

    def test_manifest_json_round_trip(self):
        out = os.path.join(self.tmp, "ref.json")
        self.assertEqual(dm.main(["write", "--root", self.ref, "--out", out]), 0)
        with open(out, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["files"], dm.build_manifest(self.ref)["files"])
        self.assertEqual(dm.load_side(out)["files"], saved["files"])

    def test_main_compare_exit_codes(self):
        out = os.path.join(self.tmp, "ref.json")
        dm.main(["write", "--root", self.ref, "--out", out])
        self.assertEqual(dm.main(["compare", out, self.other]), 0)
        _write(self.other, f"{EN}/subtask-2-annotations.txt", "EN_M_TRAIN_1.jpg\tEconomic\n")
        self.assertEqual(dm.main(["compare", out, self.other]), 1)

    def test_wrong_root_level_is_rejected(self):
        out = os.path.join(self.tmp, "ref.json")
        dm.main(["write", "--root", self.ref, "--out", out])
        self.assertEqual(dm.main(["compare", out, self.tmp]), 2)
        self.assertEqual(dm.main(["compare", out, os.path.join(self.tmp, "absent")]), 2)
        self.assertIn("use", dm._check_root(self.tmp))

    def test_report_names_the_failing_file(self):
        _write(self.other, f"{EN}/subtask-2-annotations.txt", "EN_M_TRAIN_1.jpg\tEconomic\n")
        report = dm.format_report(self._compare(), "En-Shuo", "Leo")
        self.assertIn("FAIL", report)
        self.assertIn(f"{EN}/subtask-2-annotations.txt", report)


if __name__ == "__main__":
    unittest.main()
