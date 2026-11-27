import json
import math
import re

import numpy as np
import pytest

import analysis.lock as lock
from analysis.image_text_frames import conditional_matrix, frame_relation, main

LABELS = ["Economic", "Political", "Morality"]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def no_lock(tmp_path, monkeypatch):
    path = tmp_path / "results" / "LOCKED"
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    return path


def _gold_root(tmp_path):
    root = tmp_path / "pred"
    gold = root / "_gold"
    _write(gold / "labels.txt", "\n".join(LABELS) + "\n")
    # train: aligned, divergent, image_none; dev: complementary; test: aligned
    _write(gold / "train.tsv", "EN_a.jpg\tPolitical\nBG_b.jpg\tMorality\nEN_c.jpg\n")
    _write(gold / "train_text_frames.tsv", "EN_a.jpg\tPolitical\tEconomic\nBG_b.jpg\tPolitical\nEN_c.jpg\tEconomic\n")
    _write(gold / "dev.tsv", "EN_d.jpg\tPolitical\tMorality\n")
    _write(gold / "dev_text_frames.tsv", "EN_d.jpg\tPolitical\n")
    _write(gold / "test.tsv", "EN_t.jpg\tEconomic\n")
    _write(gold / "test_text_frames.tsv", "EN_t.jpg\tEconomic\n")
    return str(root)


def test_frame_relation_categories():
    assert frame_relation({"Political"}, {"Political", "Economic"}) == "aligned"
    assert frame_relation({"Political", "Morality"}, {"Political"}) == "complementary"
    assert frame_relation({"Morality"}, {"Political"}) == "divergent"
    assert frame_relation(set(), {"Political"}) == "image_none"
    assert frame_relation({"Morality"}, set()) == "divergent"


def test_conditional_matrix_values():
    pairs = [({"Economic"}, {"Economic", "Political"}), (set(), {"Economic"})]
    m = conditional_matrix(pairs, LABELS)
    assert m.shape == (3, 3)
    assert m[0].tolist()[:2] == [0.5, 0.0]
    assert m[1].tolist()[:2] == [1.0, 0.0]


def test_conditional_matrix_row_without_text_frame_is_nan():
    m = conditional_matrix([({"Economic"}, {"Economic"})], LABELS)
    assert all(math.isnan(x) for x in m[2])
    assert not any(math.isnan(x) for x in m[0])


def test_include_test_requires_lock(tmp_path, no_lock):
    with pytest.raises(SystemExit, match="LOCKED"):
        main(["--pred_root", _gold_root(tmp_path), "--out", str(tmp_path / "c1.json"), "--include_test"])


def test_default_run_uses_train_and_dev_only(tmp_path, no_lock):
    out = tmp_path / "c1.json"
    main(["--pred_root", _gold_root(tmp_path), "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["splits"] == ["train", "dev"]
    assert data["n_articles"] == 4
    assert data["relations"] == {"aligned": 1, "complementary": 1, "divergent": 1, "image_none": 1}
    assert data["relation_share"]["aligned"] == 0.25
    assert data["mean_text_frames_per_article"] == pytest.approx(5 / 4)
    assert data["mean_image_frames_per_image"] == pytest.approx(4 / 4)
    # copying text frames: tp = Political(a) + Political(d) = 2; fp = Economic(a) + Political(b) + Economic(c) = 3
    assert data["copy_text_frames"] == {"precision": pytest.approx(2 / 5), "recall": pytest.approx(2 / 4)}
    assert set(data["by_language"]) == {"EN", "BG"} and data["by_language"]["BG"]["n_articles"] == 1
    assert data["by_split"]["dev"]["relations"]["complementary"] == 1
    assert data["conditional_matrix"][2] == [None, None, None]


def test_include_test_after_lock_adds_test(tmp_path, no_lock):
    no_lock.parent.mkdir(parents=True)
    no_lock.write_text("commit x\n", encoding="utf-8")
    out = tmp_path / "c1.json"
    main(["--pred_root", _gold_root(tmp_path), "--out", str(out), "--include_test"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["splits"] == ["train", "dev", "test"] and data["n_articles"] == 5


def test_lock_create_records_commit_and_refuses_a_second_lock(no_lock):
    assert not lock.is_locked()
    lock.main(["--create"])
    assert lock.is_locked()
    assert re.search(r"^commit [0-9a-f]{40}$", no_lock.read_text(encoding="utf-8"), re.M)
    with pytest.raises(SystemExit, match="already"):
        lock.main(["--create"])
    lock.require_lock()


# ---- Task 10: C2 (VLM text overlap) and C4 (case table) ----

from analysis.vlm_text_overlap import analyse as analyse_c2  # noqa: E402
from analysis.vlm_text_overlap import main as c2_main  # noqa: E402
from analysis.vlm_text_overlap import overlap_stats, swap_stats, text_coverage  # noqa: E402
from analysis.case_table import build_case_table  # noqa: E402
from analysis.case_table import main as case_main  # noqa: E402


def test_overlap_stats_counts_copied_text_frames():
    s = overlap_stats({"a.jpg": ["Political", "Morality"]}, {"a.jpg": ["Political"]})
    assert s["mean_n_labels"] == 2 and s["frac_pred_in_text"] == 0.5 and s["jaccard_mean"] == 0.5


def test_overlap_stats_both_empty_is_jaccard_one():
    s = overlap_stats({"a.jpg": [], "b.jpg": ["Economic"]}, {"a.jpg": [], "b.jpg": ["Political"]})
    assert s["jaccard_mean"] == 0.5 and s["mean_n_labels"] == 0.5 and s["frac_pred_in_text"] == 0.0


def test_swap_stats_counts_added_and_removed():
    s = swap_stats({"a.jpg": ["Political"]}, {"a.jpg": ["Health_and_safety", "Economic"]},
                   {"a.jpg": ["Political", "Economic"]}, {"a.jpg": ["Health_and_safety"]})
    assert s == {"added": 2, "added_correct": 1, "added_in_text": 1, "removed": 1, "removed_correct": 1}


def test_text_coverage():
    c = text_coverage(["a" * 1000, "a" * 3000, "a" * 6000], 1500)
    assert c["median_chars"] == 3000 and abs(c["frac_truncated"] - 2 / 3) < 1e-9 and c["median_share_seen"] == 0.5


def _c2_root(tmp_path):
    root = tmp_path / "pred"
    gold = root / "_gold"
    _write(gold / "labels.txt", "\n".join(LABELS) + "\n")
    _write(gold / "dev.tsv", "EN_a.jpg\tPolitical\nEN_b.jpg\n")
    _write(gold / "dev_text_frames.tsv", "EN_a.jpg\tPolitical\tEconomic\nEN_b.jpg\tMorality\n")
    _write(gold / "test.tsv", "EN_t.jpg\tEconomic\n")
    _write(gold / "test_text_frames.tsv", "EN_t.jpg\tEconomic\n")
    _write(root / "qwen_image_only" / "seed0" / "dev.tsv", "EN_a.jpg\tPolitical\nEN_b.jpg\tMorality\n")
    _write(root / "qwen_2a" / "seed0" / "dev.tsv", "EN_a.jpg\tEconomic\nEN_b.jpg\tMorality\n")
    return str(root)


def test_c2_compares_variants_against_image_only_on_dev(tmp_path, no_lock):
    out = analyse_c2(_c2_root(tmp_path), ["dev"], {"EN_a.jpg": "x" * 3000, "EN_b.jpg": "y" * 1000})
    assert out["splits"] == ["dev"] and out["base"] == "qwen_image_only"
    assert out["models"] == ["qwen_image_only", "qwen_2a"]
    dev = out["by_split"]["dev"]
    assert dev["variants"]["qwen_2a"]["swap_vs_base"] == {"added": 1, "added_correct": 0, "added_in_text": 1,
                                                          "removed": 1, "removed_correct": 1}
    assert dev["variants"]["qwen_image_only"]["swap_vs_base"] is None
    assert dev["variants"]["qwen_2a"]["overlap"]["frac_pred_in_text"] == 1.0
    assert dev["gold"]["frac_pred_in_text"] == 1.0
    assert dev["text_coverage_1500"]["frac_truncated"] == 0.5


def test_c2_include_test_requires_lock(tmp_path, no_lock):
    with pytest.raises(SystemExit, match="LOCKED"):
        c2_main(["--pred_root", _c2_root(tmp_path), "--out", str(tmp_path / "c2.json"), "--include_test"])


def test_case_table_marks_exact_matches_and_skips_missing_models(tmp_path, no_lock):
    md = build_case_table(_c2_root(tmp_path), ["qwen_image_only", "qwen_2a", "e2_vision_global"], ["dev"])
    lines = md.splitlines()
    assert any("e2_vision_global" in line and "no predictions" in line for line in lines)
    row_a = next(line for line in lines if line.startswith("| dev | EN_a.jpg"))
    assert "| Political | Political ✓ | Economic ✗ |" in row_a
    row_b = next(line for line in lines if line.startswith("| dev | EN_b.jpg"))
    assert "| None | Morality ✗ | Morality ✗ |" in row_b


def test_case_table_test_rows_require_lock(tmp_path, no_lock):
    with pytest.raises(SystemExit, match="LOCKED"):
        case_main(["--pred_root", _c2_root(tmp_path), "--models", "qwen_image_only",
                   "--out", str(tmp_path / "case.md"), "--include_test"])


def test_case_table_resolves_best_from_final_json(tmp_path, no_lock):
    final = _write(tmp_path / "final.json", json.dumps({"selection": {"best_model": "qwen_2a"}}))
    out = tmp_path / "case.md"
    case_main(["--pred_root", _c2_root(tmp_path), "--models", "best", "qwen_image_only",
               "--final", final, "--out", str(out)])
    header = next(line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("| split |"))
    assert header == "| split | image | gold | qwen_2a (best) | qwen_image_only |"
