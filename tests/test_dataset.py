import os
import pytest
import torch
from src.dataset import FramingDataset

def test_course_split_partitions():
    ds_test = FramingDataset(split="test", data_root="data_official", split_file="course_split.json")
    ds_dev = FramingDataset(split="dev", data_root="data_official", split_file="course_split.json")
    ds_train = FramingDataset(split="train", data_root="data_official", split_file="course_split.json")
    
    assert len(ds_test) == 40
    assert len(ds_dev) == 20
    assert len(ds_train) >= 437  # 400 Batch 1 + 37 Batch 2 EN
    
    # Verify sole Legality instance is in train
    sample_trail2 = [s for s in ds_train.samples if "EN_TRAIL_2" in s["id"]]
    assert len(sample_trail2) == 1
    assert "Legality_Constitutionality_and_jurisprudence" in sample_trail2[0]["labels"]
    
    # Verify target encoding for Legality
    legality_idx = ds_train.class_to_idx["Legality_Constitutionality_and_jurisprudence"]
    assert sample_trail2[0]["target"][legality_idx] == 1.0

def test_dataset_sample_item():
    ds_test = FramingDataset(split="test", data_root="data_official", split_file="course_split.json")
    item = ds_test[0]
    assert "id" in item
    assert "image_name" in item
    assert "image_path" in item and os.path.exists(item["image_path"])
    assert "text" in item and len(item["text"]) > 0
    assert "labels" in item and isinstance(item["labels"], list)
    assert "target" in item and isinstance(item["target"], torch.Tensor)
    assert item["target"].shape == (14,)
    assert item["target"].dtype == torch.float32

def test_invalid_split_raises():
    with pytest.raises(ValueError):
        FramingDataset(split="invalid_split", data_root="data_official", split_file="course_split.json")
