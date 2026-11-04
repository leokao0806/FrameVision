"""
tests/test_ablations.py
Unit and integration tests for the Automated Ablation Benchmark Suite
and markdown report generator for SemEval-2027 Task 9 Subtask 2.
"""

import importlib.util
import json
import os
import subprocess
import sys
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "04_run_ablations.py")
if not os.path.exists(_SCRIPT_PATH):
    raise ImportError(f"Target script not found: {_SCRIPT_PATH}")

_spec = importlib.util.spec_from_file_location("scripts.04_run_ablations", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load spec for {_SCRIPT_PATH}")
_ablation_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts.04_run_ablations"] = _ablation_mod
_spec.loader.exec_module(_ablation_mod)

format_ablation_markdown_table = getattr(_ablation_mod, "format_ablation_markdown_table")
train_unimodal_or_ablation = getattr(_ablation_mod, "train_unimodal_or_ablation")
run_ablation_suite = getattr(_ablation_mod, "run_ablation_suite")


@pytest.fixture
def mock_ablation_env(tmp_path):
    """
    Creates a hermetic synthetic dataset environment in tmp_path with:
      - data_official/labels.txt
      - data_official/training-second-batch-oct/EN/subtask-2-annotations.txt
      - course_split.json
      - cache/features/{id}.pt for train, dev, and test samples
    """
    data_root = tmp_path / "data_official"
    b2_dir = data_root / "training-second-batch-oct" / "EN"
    raw_docs = b2_dir / "raw-documents"
    raw_docs.mkdir(parents=True, exist_ok=True)

    from src.metrics import DEFAULT_CLASSES

    labels_file = data_root / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write("\n".join(DEFAULT_CLASSES) + "\n")

    train_ids = ["mock_tr_1", "mock_tr_2", "mock_tr_3", "mock_tr_4"]
    dev_ids = ["mock_dv_1", "mock_dv_2"]
    test_ids = ["mock_ts_1", "mock_ts_2"]
    all_samples = [
        ("mock_tr_1", "mock_tr_1.jpg", ["Economic"]),
        ("mock_tr_2", "mock_tr_2.jpg", ["Morality", "Political"]),
        ("mock_tr_3", "mock_tr_3.jpg", ["Quality_of_life"]),
        ("mock_tr_4", "mock_tr_4.jpg", []),
        ("mock_dv_1", "mock_dv_1.jpg", ["Security_and_defense"]),
        ("mock_dv_2", "mock_dv_2.jpg", ["Economic"]),
        ("mock_ts_1", "mock_ts_1.jpg", ["Health_and_safety"]),
        ("mock_ts_2", "mock_ts_2.jpg", ["Political"]),
    ]

    annot_file = b2_dir / "subtask-2-annotations.txt"
    with open(annot_file, "w", encoding="utf-8") as f:
        for sid, img_name, lbls in all_samples:
            if lbls:
                f.write(f"{img_name}\t" + "\t".join(lbls) + "\n")
            else:
                f.write(f"{img_name}\n")
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

    for sid, img_name, lbls in all_samples:
        target = torch.zeros(len(DEFAULT_CLASSES), dtype=torch.float32)
        for l in lbls:
            if l in DEFAULT_CLASSES:
                target[DEFAULT_CLASSES.index(l)] = 1.0

        feat_payload = {
            "id": sid,
            "image_name": img_name,
            "visual_tokens": torch.randn(4, 768, dtype=torch.float32),
            "text_tokens": torch.randn(16, 768, dtype=torch.float32),
            "text_mask": torch.ones(16, dtype=torch.int64),
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


def test_ablation_table_formatting():
    """
    Tests markdown formatting of results and presence of reference baselines:
      - Required columns: | Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |
      - Intermediate Report Table 1 baselines present:
        - Image LR (SigLIP) (0.353, 0.281)
        - Late fusion (Image + Text) (0.367, 0.350)
        - Qwen2.5-VL (Image only) (0.417, 0.205)
        - Qwen2.5-VL (Image + Text) (0.333, 0.229)
      - Formats ablation experiment rows.
    """
    results = [
        {
            "name": "Exp 1: Text-Only (DeBERTa-v3 + ASL)",
            "micro_f1": 0.285,
            "macro_f1": 0.220,
            "hamming_loss": 0.125,
            "notes": "Text modality only",
        },
        {
            "name": "Exp 2: Vision-Only Global (SigLIP Global + ASL)",
            "micro_f1": 0.355,
            "macro_f1": 0.285,
            "hamming_loss": 0.110,
            "notes": "Global image token only",
        },
        {
            "name": "Exp 3: Vision-Only Region Attention (SigLIP Global + K Regions + ASL)",
            "micro_f1": 0.380,
            "macro_f1": 0.315,
            "hamming_loss": 0.098,
            "notes": "Visual tokens with self-attention",
        },
        {
            "name": "Exp 4: Multimodal Cross-Attention (BCE Loss)",
            "micro_f1": 0.395,
            "macro_f1": 0.320,
            "hamming_loss": 0.092,
            "notes": "Cross-attention with standard BCE",
        },
        {
            "name": "Exp 5: Multimodal Cross-Attention (ASL Full Model)",
            "micro_f1": 0.448,
            "macro_f1": 0.375,
            "hamming_loss": 0.076,
            "notes": "Proposed full multimodal model",
        },
    ]

    md = format_ablation_markdown_table(results)

    # 1. Header validation
    assert "| Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |" in md

    # 2. Reference baselines validation
    assert "Image LR (SigLIP)" in md
    assert "0.353" in md
    assert "0.281" in md

    assert "Late fusion (Image + Text)" in md
    assert "0.367" in md
    assert "0.350" in md

    assert "Qwen2.5-VL (Image only)" in md
    assert "0.417" in md
    assert "0.205" in md

    assert "Qwen2.5-VL (Image + Text)" in md
    assert "0.333" in md
    assert "0.229" in md

    # 3. Experiment rows validation
    assert "Exp 1: Text-Only (DeBERTa-v3 + ASL)" in md
    assert "0.285" in md
    assert "Exp 5: Multimodal Cross-Attention (ASL Full Model)" in md
    assert "0.448" in md


def test_run_ablation_suite_dry_run(tmp_path):
    """
    Runs run_ablation_suite with dry_run=True and verifies:
      - Returns list of 5 ablation experiment dictionaries
      - Output markdown report file ablation_results.md is generated
      - Content of report contains the table and baselines
    """
    report_dir = tmp_path / "reports"
    results = run_ablation_suite(
        report_dir=str(report_dir),
        dry_run=True,
    )

    assert isinstance(results, list)
    assert len(results) == 5

    exp_names = [r["name"] for r in results]
    assert any("Exp 1" in name or "text_only" in name.lower() for name in exp_names)
    assert any("Exp 2" in name or "vision_global" in name.lower() for name in exp_names)
    assert any("Exp 3" in name or "vision_region" in name.lower() for name in exp_names)
    assert any("Exp 4" in name or "fusion_bce" in name.lower() for name in exp_names)
    assert any("Exp 5" in name or "fusion_asl" in name.lower() for name in exp_names)

    report_path = report_dir / "ablation_results.md"
    assert os.path.exists(report_path)
    content = report_path.read_text(encoding="utf-8")
    assert "| Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |" in content
    assert "Image LR (SigLIP)" in content
    assert "Qwen2.5-VL" in content


def test_train_unimodal_or_ablation_dry_run(mock_ablation_env):
    """
    Tests train_unimodal_or_ablation for unimodal and cross-modal model variants:
      - text_only
      - vision_global
      - vision_region
      - fusion_bce
      - fusion_asl
    """
    models_to_test = ["text_only", "vision_global", "vision_region", "fusion_bce", "fusion_asl"]
    base_ckpt = mock_ablation_env["tmp_path"] / "checkpoints_ablation"

    for model_type in models_to_test:
        loss_type = "bce" if model_type == "fusion_bce" else "asl"
        ckpt_dir = base_ckpt / model_type
        res = train_unimodal_or_ablation(
            model_type=model_type,
            loss_type=loss_type,
            feature_dir=mock_ablation_env["feature_dir"],
            split_file=mock_ablation_env["split_file"],
            data_root=mock_ablation_env["data_root"],
            epochs=1,
            batch_size=2,
            embed_dim=64,
            num_heads=2,
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
        )

        assert isinstance(res, dict)
        for key in ["model_type", "micro_f1", "macro_f1", "hamming_loss", "best_threshold", "checkpoint_path"]:
            assert key in res, f"Missing key {key} in results for {model_type}"

        assert 0.0 <= res["micro_f1"] <= 1.0
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert 0.0 <= res["hamming_loss"] <= 1.0
        assert os.path.exists(res["checkpoint_path"])


def test_ablation_suite_cli(tmp_path):
    """
    Verifies CLI invocation of scripts/04_run_ablations.py with --dry_run.
    """
    report_dir = tmp_path / "reports_cli"
    cmd = [
        sys.executable,
        _SCRIPT_PATH,
        "--dry_run",
        "--report_dir", str(report_dir),
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Ablation Suite Completed" in res.stdout or "ablation" in res.stdout.lower()
    assert os.path.exists(report_dir / "ablation_results.md")
