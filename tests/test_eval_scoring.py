import os
import sys

import numpy as np
import pytest

from evaluation.bootstrap import bootstrap_ci, group_of, paired_ci
from evaluation.scoring import hamming, micro_f1, per_label_f1, run_official_scorer, score_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_SCORER = os.path.join(REPO, "starter-kit", "subtask2a_2b", "scorer-subtask-2.py")
LABELS = ["Economic", "Political", "Morality"]


def _with_none(y14):
    none = (y14.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    return np.hstack([y14, none])


def _random_pair(n=40, k=14, seed=7):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n, k) < 0.12).astype(int)
    p = (rng.rand(n, k) < 0.15).astype(int)
    return y, p


# Copy of baselines/src/metrics.py::bootstrap_micro_f1 (Intermediate Report), kept here as the reference.
def _intermediate_bootstrap(y_true, y_pred, groups, n_boot=2000, seed=0):
    t, p = _with_none(np.asarray(y_true).astype(int)), _with_none(np.asarray(y_pred).astype(int))
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    counts = np.zeros((len(names), 3))
    for row, group in enumerate(groups):
        counts[index[group]] += [(t[row] & p[row]).sum(), ((1 - t[row]) & p[row]).sum(), (t[row] & (1 - p[row])).sum()]
    idx = np.random.RandomState(seed).randint(0, len(counts), (n_boot, len(counts)))
    tp, fp, fn = counts[idx].sum(axis=-2).T
    scores = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def test_group_of_strips_extension():
    assert group_of("EN_M_TRAIN_11.jpeg") == "EN_M_TRAIN_11"


def test_bootstrap_single_seed_matches_intermediate_function():
    y14, p14 = _random_pair()
    groups = [f"doc{i:02d}" for i in range(len(y14))][::-1]
    expected = _intermediate_bootstrap(y14, p14, groups)
    assert bootstrap_ci(_with_none(y14), [_with_none(p14)], groups) == expected


def test_bootstrap_averages_seeds_within_each_resample():
    y14, p14 = _random_pair()
    y, p = _with_none(y14), _with_none(p14)
    g = [f"doc{i}" for i in range(len(y))]
    assert bootstrap_ci(y, [p, p], g) == bootstrap_ci(y, [p], g)


def test_bootstrap_of_two_different_seeds_averages_per_resample():
    y14, p14 = _random_pair()
    _, q14 = _random_pair(seed=8)
    y, p, q = _with_none(y14), _with_none(p14), _with_none(q14)
    g = [f"doc{i:02d}" for i in range(len(y))]
    idx = np.random.RandomState(0).randint(0, len(g), (2000, len(g)))

    def scores(pred):  # rows are already one article each, in name order
        t, s = y.astype(int), pred.astype(int)
        counts = np.stack([(t & s).sum(1), ((1 - t) & s).sum(1), (t & (1 - s)).sum(1)], axis=1)[idx].sum(axis=1)
        tp, fp, fn = counts.T
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    mean = (scores(p) + scores(q)) / 2
    expected = (float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5)))
    assert bootstrap_ci(y, [p, q], g) == pytest.approx(expected, abs=1e-12)


def test_paired_ci_of_identical_models_is_zero():
    y14, p14 = _random_pair()
    y, p = _with_none(y14), _with_none(p14)
    g = [f"doc{i}" for i in range(len(y))]
    assert paired_ci(y, [p], [p], g) == (0.0, 0.0)


def test_micro_and_hamming_on_a_small_example():
    y = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    p = np.array([[1, 1, 0, 0], [0, 0, 0, 1]])
    assert micro_f1(y, p) == pytest.approx(2 * 2 / (2 * 2 + 1))
    assert hamming(y, p) == pytest.approx(1 / 8)


def test_per_label_f1_absent_label_is_none():
    y = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    p = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    out = per_label_f1(y, p, LABELS + ["None"])
    assert out["Economic"] == pytest.approx(2 / 3)
    assert out["Political"] is None and out["Morality"] is None
    assert out["None"] == 0.0


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _fixture(tmp_path):
    labels = _write(tmp_path / "labels.txt", "\n".join(LABELS) + "\n")
    gold = _write(tmp_path / "gold.tsv", "a.jpg\tEconomic\nb.jpg\nc.jpg\tPolitical\tMorality\n")
    pred = _write(tmp_path / "m" / "seed0" / "test.tsv", "c.jpg\tPolitical\na.jpg\tEconomic\nb.jpg\tMorality\n")
    return labels, gold, pred


def test_official_score_wins_and_mismatch_is_counted(tmp_path):
    labels, gold, pred = _fixture(tmp_path)
    fake_scorer = _write(tmp_path / "fake_scorer.py", 'print("some log line")\nprint("0.5\\t0.4")\n')
    out = score_split([pred], gold, labels, fake_scorer)
    assert out["micro"] == [0.5] and out["macro"] == [0.4]
    assert out["mismatches"] == 1
    assert out["image_names"] == ["a.jpg", "b.jpg", "c.jpg"]
    assert out["support"] == {"Economic": 1, "Political": 1, "Morality": 1, "None": 1}
    assert out["matrices"][0].tolist() == [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]


def test_scorer_without_output_raises_with_pred_path(tmp_path):
    labels, gold, pred = _fixture(tmp_path)
    silent = _write(tmp_path / "silent_scorer.py", "import sys\nsys.stderr.write('bad format')\n")
    with pytest.raises(RuntimeError, match="test.tsv"):
        run_official_scorer(silent, gold, pred, labels)


def test_scorer_receives_canonical_rows_in_gold_order(tmp_path):
    labels, gold, _ = _fixture(tmp_path)
    pred = _write(tmp_path / "m" / "seed0" / "dev.tsv",
                  "﻿c.jpg\tPolitical  \nb.jpg\tNone\na.jpg\tEconomic\t\n")
    seen = tmp_path / "seen.tsv"
    spy = _write(tmp_path / "spy_scorer.py",
                 "import shutil, sys\n"
                 f"shutil.copy(sys.argv[sys.argv.index('-p') + 1], {str(seen)!r})\n"
                 'print("0.5\\t0.4")\n')
    score_split([pred], gold, labels, spy)
    assert seen.read_bytes() == b"a.jpg\tEconomic\nb.jpg\nc.jpg\tPolitical\n"


def test_scorer_crash_reports_stderr_and_pred_path(tmp_path):
    labels, gold, pred = _fixture(tmp_path)
    crash = _write(tmp_path / "crash_scorer.py", "import sys\nsys.stderr.write('boom')\nsys.exit(2)\n")
    with pytest.raises(RuntimeError, match=r"(?s)test\.tsv.*boom"):
        run_official_scorer(crash, gold, pred, labels)


@pytest.mark.skipif(not os.path.isfile(REAL_SCORER), reason="official starter kit not linked at repo root")
@pytest.mark.parametrize("pred_text", [
    "a.jpg\tEconomic\nb.jpg\tNone\nc.jpg\tPolitical\n",
    "a.jpg\tEconomic\t\nb.jpg\t\nc.jpg\tPolitical\n",
    "﻿a.jpg\tEconomic\nb.jpg\nc.jpg\tPolitical  \n",
], ids=["literal_none", "trailing_tabs", "bom_and_spaces"])
def test_lenient_forms_are_scored_by_the_real_scorer(tmp_path, pred_text):
    labels, gold, _ = _fixture(tmp_path)
    pred = _write(tmp_path / "m" / "seed0" / "dev.tsv", pred_text)
    out = score_split([pred], gold, labels, REAL_SCORER)
    assert out["mismatches"] == 0
    assert out["micro"][0] == pytest.approx(micro_f1(out["y_true"], out["matrices"][0]), abs=1e-6)


@pytest.mark.skipif(not os.path.isfile(REAL_SCORER), reason="official starter kit not linked at repo root")
def test_real_scorer_agrees_with_micro_f1(tmp_path):
    labels, gold, pred = _fixture(tmp_path)
    out = score_split([pred], gold, labels, REAL_SCORER)
    assert out["mismatches"] == 0
    assert out["micro"][0] == pytest.approx(micro_f1(out["y_true"], out["matrices"][0]), abs=1e-6)
