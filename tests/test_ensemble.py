import json

import numpy as np
import pytest

from ensemble.combine import combine, fit_on_dev, load_probs, main

LABELS = ["Economic", "Political", "Morality"]
ALPHAS = (0.25, 0.5, 0.75)
THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _probs_json(path, names, classes, probs, threshold=0.5):
    return _write(path, json.dumps({"image_names": names, "classes": classes, "probs": probs, "threshold": threshold}))


def test_load_probs_reorders_by_class_name(tmp_path):
    path = _probs_json(tmp_path / "p.json", ["a.jpg", "b.jpg"], ["Political", "Morality", "Economic"],
                       [[0.9, 0.1, 0.2], [0.3, 0.4, 0.5]])
    names, probs = load_probs(path, LABELS)
    assert names == ["a.jpg", "b.jpg"]
    assert probs.tolist() == [[0.2, 0.9, 0.1], [0.5, 0.3, 0.4]]


def test_load_probs_rejects_different_class_set(tmp_path):
    path = _probs_json(tmp_path / "p.json", ["a.jpg"], ["Political", "Morality", "Economy"], [[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="p.json"):
        load_probs(path, LABELS)


def test_combine_alpha_one_returns_qwen():
    q = np.array([[1, 0, 0], [0, 1, 1]], dtype=float)
    p = np.array([[0.2, 0.9, 0.1], [0.5, 0.3, 0.4]])
    assert np.array_equal(combine(q, p, 1.0), q)
    assert np.allclose(combine(q, p, 0.25), 0.25 * q + 0.75 * p)


def test_fit_on_dev_only_uses_allowed_grid():
    rng = np.random.RandomState(0)
    q = (rng.rand(20, 3) < 0.3).astype(float)
    p = rng.rand(20, 3)
    y = (rng.rand(20, 3) < 0.3).astype(int)
    a, t, dev = fit_on_dev(q, p, y)
    assert a in ALPHAS and round(t, 2) in THRESHOLDS and 0.0 <= dev <= 1.0


def test_fit_on_dev_finds_a_perfect_setting():
    q = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
    p = np.array([[0.1, 0.1, 0.1], [0.1, 0.9, 0.1]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    a, t, dev = fit_on_dev(q, p, y)
    assert dev == 1.0
    assert np.array_equal((combine(q, p, a) >= t).astype(int), y)


def _root(tmp_path, best_trained="image_lr"):
    root = tmp_path / "pred"
    gold = root / "_gold"
    _write(gold / "labels.txt", "\n".join(LABELS) + "\n")
    _write(gold / "dev.tsv", "d1.jpg\tEconomic\nd2.jpg\tPolitical\n")
    _write(gold / "test.tsv", "t1.jpg\tEconomic\nt2.jpg\n")
    _write(root / "qwen_image_only" / "seed0" / "dev.tsv", "d1.jpg\tEconomic\nd2.jpg\n")
    _write(root / "qwen_image_only" / "seed0" / "test.tsv", "t1.jpg\nt2.jpg\n")
    for seed in (13, 42):
        _probs_json(root / "image_lr" / f"seed{seed}" / "dev.probs.json", ["d2.jpg", "d1.jpg"], LABELS,
                    [[0.1, 0.9, 0.1], [0.1, 0.1, 0.1]])
        _probs_json(root / "image_lr" / f"seed{seed}" / "test.probs.json", ["t1.jpg", "t2.jpg"], LABELS,
                    [[0.2, 0.1, 0.1], [0.1, 0.1, 0.1]])
    final = _write(tmp_path / "final.json", json.dumps({"selection": {"best_trained_model": best_trained}}))
    return root, final


def test_cli_writes_each_seed_and_selection(tmp_path):
    root, final = _root(tmp_path)
    main(["--pred_root", str(root), "--source", "image_lr", "--name", "combo_qwen_image_lr", "--final", final])
    out = root / "combo_qwen_image_lr"
    assert sorted(p.name for p in out.iterdir()) == ["seed13", "seed42", "selection.json"]
    assert (out / "seed13" / "dev.tsv").read_text(encoding="utf-8") == "d1.jpg\tEconomic\nd2.jpg\tPolitical\n"
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert selection["source"] == "image_lr" and selection["qwen"] == "qwen_image_only"
    assert set(selection["seeds"]) == {"13", "42"}
    assert selection["seeds"]["13"]["dev_micro"] == 1.0 and selection["seeds"]["13"]["alpha"] in ALPHAS


def test_no_label_above_threshold_means_none(tmp_path):
    root, final = _root(tmp_path)
    main(["--pred_root", str(root), "--source", "image_lr", "--name", "combo_qwen_image_lr", "--final", final])
    test_rows = (root / "combo_qwen_image_lr" / "seed42" / "test.tsv").read_text(encoding="utf-8").splitlines()
    assert test_rows[1] == "t2.jpg"


def test_default_source_is_best_trained_model(tmp_path):
    root, final = _root(tmp_path)
    main(["--pred_root", str(root), "--final", final])
    assert (root / "combo_qwen_best" / "seed13" / "test.tsv").is_file()
    root2, final_none = _root(tmp_path / "x", best_trained=None)
    with pytest.raises(SystemExit, match="best_trained_model"):
        main(["--pred_root", str(root2), "--final", final_none])
