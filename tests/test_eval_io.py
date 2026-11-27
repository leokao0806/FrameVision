import pytest

from evaluation.prediction_io import (
    NONE_LABEL,
    check_same_images,
    discover_runs,
    read_frames_tsv,
    read_labels,
    to_matrix,
    write_frames_tsv,
)

LABELS = ["Economic", "Political", "Morality"]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_read_labels_keeps_file_order(tmp_path):
    path = _write(tmp_path / "labels.txt", "Political\nEconomic\n\n")
    assert read_labels(path) == ["Political", "Economic"]
    assert NONE_LABEL == "None"


def test_empty_row_and_literal_none_mean_none(tmp_path):
    path = _write(tmp_path / "pred.tsv", "a.jpg\nb.jpg\tNone\n")
    rows = read_frames_tsv(path, LABELS)
    assert rows == {"a.jpg": [], "b.jpg": []}
    assert to_matrix(rows, ["a.jpg", "b.jpg"], LABELS).tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]


def test_frames_fill_their_columns_and_none_column_stays_zero(tmp_path):
    path = _write(tmp_path / "pred.tsv", "a.jpg\tMorality\tEconomic\r\n")
    rows = read_frames_tsv(path, LABELS)
    assert to_matrix(rows, ["a.jpg"], LABELS).tolist() == [[1, 0, 1, 0]]


def test_unknown_label_raises_with_path(tmp_path):
    path = _write(tmp_path / "pred.tsv", "a.jpg\tEconomy\n")
    with pytest.raises(ValueError, match="pred.tsv"):
        read_frames_tsv(path, LABELS)


def test_duplicate_image_raises(tmp_path):
    path = _write(tmp_path / "pred.tsv", "a.jpg\tEconomic\na.jpg\n")
    with pytest.raises(ValueError, match="a.jpg"):
        read_frames_tsv(path, LABELS)


def test_check_same_images_reports_missing_and_extra():
    with pytest.raises(ValueError, match="missing.*c.jpg") as err:
        check_same_images({"a.jpg": [], "b.jpg": []}, {"a.jpg": [], "c.jpg": []}, "m/seed13/test.tsv")
    assert "b.jpg" in str(err.value) and "m/seed13/test.tsv" in str(err.value)


def test_check_same_images_passes_when_sets_match():
    check_same_images({"a.jpg": ["Economic"]}, {"a.jpg": []}, "m/seed0/dev.tsv")


def test_discover_runs_skips_underscore_dirs(tmp_path):
    for rel in ["_gold/test.tsv", "image_lr/seed0/test.tsv",
                "e5_fusion_coco_asl/seed13/dev.tsv", "e5_fusion_coco_asl/seed13/test.tsv"]:
        _write(tmp_path / rel, "a.jpg\n")
    runs = discover_runs(str(tmp_path))
    assert set(runs) == {"image_lr", "e5_fusion_coco_asl"}
    assert runs["e5_fusion_coco_asl"][13]["dev"].endswith("dev.tsv")
    assert set(runs["image_lr"][0]) == {"test"}


def test_write_then_read_roundtrip(tmp_path):
    rows = {"a.jpg": ["Political", "Economic"], "b.jpg": []}
    path = str(tmp_path / "new" / "dir" / "pred.tsv")
    write_frames_tsv(path, rows)
    with open(path, "rb") as f:
        assert b"\r\n" not in f.read()
    assert read_frames_tsv(path, LABELS) == rows
