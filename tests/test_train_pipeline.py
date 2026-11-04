"""
tests/test_train_pipeline.py
Unit and integration tests for CachedFramingDataset and the modular training pipeline.
"""

import json
import os
import subprocess
import sys
import pytest
import torch

from src.cached_dataset import CachedFramingDataset, collate_cached_batch
from scripts import train_pipeline
from src.metrics import DEFAULT_CLASSES


@pytest.fixture
def mock_dataset_env(tmp_path):
    """
    Creates a hermetic mock dataset environment in tmp_path with:
      - data_official/labels.txt
      - data_official/training-second-batch-oct/EN/subtask-2-annotations.txt
      - course_split.json
      - cache/features/{id}.pt for train, dev, and test samples
    """
    data_root = tmp_path / "data_official"
    b2_dir = data_root / "training-second-batch-oct" / "EN"
    raw_docs = b2_dir / "raw-documents"
    raw_docs.mkdir(parents=True, exist_ok=True)

    labels_file = data_root / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write("\n".join(DEFAULT_CLASSES) + "\n")

    # Sample IDs
    train_ids = ["mock_train_1", "mock_train_2", "mock_train_3", "mock_train_4"]
    dev_ids = ["mock_dev_1", "mock_dev_2"]
    test_ids = ["mock_test_1", "mock_test_2"]
    all_samples = [
        ("mock_train_1", "mock_train_1.jpg", ["Economic"]),
        ("mock_train_2", "mock_train_2.jpg", ["Morality", "Political"]),
        ("mock_train_3", "mock_train_3.jpg", ["Quality_of_life"]),
        ("mock_train_4", "mock_train_4.jpg", []),
        ("mock_dev_1", "mock_dev_1.jpg", ["Security_and_defense"]),
        ("mock_dev_2", "mock_dev_2.jpg", ["Economic"]),
        ("mock_test_1", "mock_test_1.jpg", ["Health_and_safety"]),
        ("mock_test_2", "mock_test_2.jpg", ["Political"]),
    ]

    annot_file = b2_dir / "subtask-2-annotations.txt"
    with open(annot_file, "w", encoding="utf-8") as f:
        for sid, img_name, lbls in all_samples:
            if lbls:
                f.write(f"{img_name}\t" + "\t".join(lbls) + "\n")
            else:
                f.write(f"{img_name}\n")
            # Create dummy text and image files
            (raw_docs / f"{sid}.txt").write_text(f"Text for {sid}", encoding="utf-8")
            (raw_docs / img_name).write_bytes(b"\x00")

    split_file = tmp_path / "course_split.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump({
            "train": train_ids,
            "dev": dev_ids,
            "test": test_ids,
        }, f)

    feature_dir = tmp_path / "cache" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Save mock .pt files
    for sid, img_name, lbls in all_samples:
        target = torch.zeros(len(DEFAULT_CLASSES), dtype=torch.float32)
        for l in lbls:
            if l in DEFAULT_CLASSES:
                target[DEFAULT_CLASSES.index(l)] = 1.0

        feat_payload = {
            "id": sid,
            "image_name": img_name,
            "visual_tokens": torch.randn(4, 768, dtype=torch.float32),
            "text_tokens": torch.randn(32, 768, dtype=torch.float32),
            "text_mask": torch.ones(32, dtype=torch.int64),
            "target": target,
            "labels": lbls,
        }
        torch.save(feat_payload, feature_dir / f"{sid}.pt")

    return {
        "tmp_path": tmp_path,
        "data_root": str(data_root),
        "split_file": str(split_file),
        "feature_dir": str(feature_dir),
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
    }


def test_cached_dataset_loading(mock_dataset_env):
    """
    Verifies CachedFramingDataset loads cached .pt files and returns correct sample shapes,
    and checks that collate_cached_batch stacks them into batched tensors properly.
    """
    feature_dir = mock_dataset_env["feature_dir"]
    split_file = mock_dataset_env["split_file"]
    data_root = mock_dataset_env["data_root"]

    train_ds = CachedFramingDataset(
        feature_dir=feature_dir,
        split="train",
        split_file=split_file,
        data_root=data_root,
    )

    assert len(train_ds) == 4
    item = train_ds[0]

    assert "id" in item and item["id"] == "mock_train_1"
    assert "image_name" in item and item["image_name"] == "mock_train_1.jpg"
    assert "visual_tokens" in item and item["visual_tokens"].shape == (4, 768)
    assert "text_tokens" in item and item["text_tokens"].shape == (32, 768)
    assert "text_mask" in item and item["text_mask"].shape == (32,)
    assert "target" in item and item["target"].shape == (14,)
    assert "labels" in item and isinstance(item["labels"], list)

    # Test collate function
    batch = collate_cached_batch([train_ds[0], train_ds[1]])
    assert batch["visual_tokens"].shape == (2, 4, 768)
    assert batch["text_tokens"].shape == (2, 32, 768)
    assert batch["text_mask"].shape == (2, 32)
    assert batch["target"].shape == (2, 14)
    assert batch["id"] == ["mock_train_1", "mock_train_2"]
    assert batch["image_name"] == ["mock_train_1.jpg", "mock_train_2.jpg"]


def test_train_pipeline_dry_run(mock_dataset_env):
    """
    Executes train_pipeline for 2 epochs on the synthetic mock environment.
    Verifies model training, threshold calibration, dev checkpointing,
    test evaluation, TSV export, and returned metric dictionary structure.
    """
    ckpt_dir = mock_dataset_env["tmp_path"] / "checkpoints"

    results = train_pipeline(
        feature_dir=mock_dataset_env["feature_dir"],
        split_file=mock_dataset_env["split_file"],
        data_root=mock_dataset_env["data_root"],
        epochs=2,
        batch_size=2,
        lr=1e-3,
        embed_dim=64,
        num_heads=2,
        loss_type="asl",
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
    )

    expected_keys = {
        "best_epoch",
        "dev_micro",
        "dev_macro",
        "best_threshold",
        "test_micro",
        "test_macro",
        "checkpoint_path",
        "predictions_tsv",
    }
    assert expected_keys.issubset(results.keys()), f"Missing keys in results: {results.keys()}"

    assert 1 <= results["best_epoch"] <= 2
    assert 0.0 <= results["best_threshold"] <= 1.0
    assert 0.0 <= results["dev_micro"] <= 1.0
    assert 0.0 <= results["dev_macro"] <= 1.0
    assert 0.0 <= results["test_micro"] <= 1.0
    assert 0.0 <= results["test_macro"] <= 1.0

    # Verify best model checkpoint exists and is valid
    ckpt_path = results["checkpoint_path"]
    assert os.path.exists(ckpt_path)
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt_data
    assert "best_threshold" in ckpt_data
    assert ckpt_data["best_threshold"] == results["best_threshold"]

    # Verify predictions TSV exists and format matches official specification
    tsv_path = results["predictions_tsv"]
    assert os.path.exists(tsv_path)
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    assert len(lines) == len(mock_dataset_env["test_ids"])
    for line in lines:
        parts = line.split("\t")
        assert parts[0].endswith(".jpg")


def test_train_pipeline_bce_loss(mock_dataset_env):
    """
    Verifies that train_pipeline supports the BCE loss ablation (--loss_type bce).
    """
    ckpt_dir = mock_dataset_env["tmp_path"] / "checkpoints_bce"

    results = train_pipeline(
        feature_dir=mock_dataset_env["feature_dir"],
        split_file=mock_dataset_env["split_file"],
        data_root=mock_dataset_env["data_root"],
        epochs=1,
        batch_size=2,
        lr=1e-3,
        embed_dim=64,
        num_heads=2,
        loss_type="bce",
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
    )

    assert os.path.exists(results["checkpoint_path"])
    assert os.path.exists(results["predictions_tsv"])


def test_train_pipeline_cli(mock_dataset_env):
    """
    Verifies CLI invocation of scripts/03_train.py via subprocess.
    """
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "03_train.py",
    )
    ckpt_dir = mock_dataset_env["tmp_path"] / "checkpoints_cli"

    cmd = [
        sys.executable,
        script_path,
        "--feature_dir", mock_dataset_env["feature_dir"],
        "--split_file", mock_dataset_env["split_file"],
        "--data_root", mock_dataset_env["data_root"],
        "--epochs", "1",
        "--batch_size", "2",
        "--embed_dim", "64",
        "--num_heads", "2",
        "--loss_type", "asl",
        "--checkpoint_dir", str(ckpt_dir),
        "--device", "cpu",
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Training pipeline finished successfully" in res.stdout or "Post-Training Test Evaluation" in res.stdout
    assert os.path.exists(ckpt_dir / "best_model.pt")
    assert os.path.exists(ckpt_dir / "predictions_test_subtask2.tsv")
