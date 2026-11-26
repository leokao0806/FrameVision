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
