import os
import glob
import pytest
from PIL import Image
from src.entity_extractor import SalientEntityExtractor, extract_salient_regions


def test_extract_salient_regions_returns_k_crops():
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    crops, boxes = extract_salient_regions(img, top_k=3)
    assert len(crops) == 3
    assert len(boxes) == 3
    for crop in crops:
        assert isinstance(crop, Image.Image)
        assert crop.width > 0 and crop.height > 0
    for box in boxes:
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480

    # Test with top_k = 1
    crops1, boxes1 = extract_salient_regions(img, top_k=1)
    assert len(crops1) == 1
    assert len(boxes1) == 1
    assert 0 <= boxes1[0][0] < boxes1[0][2] <= 640
    assert 0 <= boxes1[0][1] < boxes1[0][3] <= 480


def test_fallback_mechanism():
    # A completely blank image with zero contrast will yield 0 object detections
    w, h = 640, 480
    blank_img = Image.new("RGB", (w, h), color=(0, 0, 0))
    crops, boxes = extract_salient_regions(blank_img, top_k=3, conf_thresh=0.5)

    assert len(crops) == 3
    assert len(boxes) == 3

    # Expected deterministic fallback boxes:
    # Fallback 1: Center focal crop [0.15*w, 0.15*h, 0.85*w, 0.85*h]
    # Fallback 2: Upper focus / headline region [0.05*w, 0.05*h, 0.95*w, 0.60*h]
    # Fallback 3: Golden-ratio center-left crop [0.05*w, 0.20*h, 0.70*w, 0.80*h]
    fb1 = [int(round(0.15 * w)), int(round(0.15 * h)), int(round(0.85 * w)), int(round(0.85 * h))]
    fb2 = [int(round(0.05 * w)), int(round(0.05 * h)), int(round(0.95 * w)), int(round(0.60 * h))]
    fb3 = [int(round(0.05 * w)), int(round(0.20 * h)), int(round(0.70 * w)), int(round(0.80 * h))]

    assert boxes[0] == fb1
    assert boxes[1] == fb2
    assert boxes[2] == fb3

    for crop, box in zip(crops, boxes):
        assert isinstance(crop, Image.Image)
        expected_w = box[2] - box[0]
        expected_h = box[3] - box[1]
        assert crop.size == (expected_w, expected_h)


def test_extractor_real_image():
    # Find any official image if available
    official_candidates = glob.glob("data_official/**/*.jpg", recursive=True)
    if not official_candidates:
        official_candidates = glob.glob("../data_official/**/*.jpg", recursive=True)

    if not official_candidates:
        pytest.skip("No real image found in data_official to test.")

    sample_path = official_candidates[0]
    with Image.open(sample_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        crops, boxes = extract_salient_regions(img_rgb, top_k=3)

        assert len(crops) == 3
        assert len(boxes) == 3
        for crop in crops:
            assert isinstance(crop, Image.Image)
            assert crop.width > 0 and crop.height > 0
        for box in boxes:
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h


def test_salient_entity_extractor_class():
    extractor = SalientEntityExtractor(device="cpu")
    img = Image.new("RGB", (320, 240), color=(50, 50, 50))
    crops, boxes = extractor.extract_regions(img, top_k=2)
    assert len(crops) == 2
    assert len(boxes) == 2
    for crop in crops:
        assert isinstance(crop, Image.Image)
        assert crop.width > 0 and crop.height > 0
    for box in boxes:
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= 320
        assert 0 <= y1 < y2 <= 240
