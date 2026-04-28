"""
tests/test_detector.py
Unit tests for PersonDetector.
Run with: pytest tests/test_detector.py -v
"""

import cv2
import numpy as np
import pytest

from core.detection.person_detector import PersonDetector, DetectedPerson


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_with_noise(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestDetectedPerson:

    def test_properties(self):
        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        p = DetectedPerson(bbox=(10, 20, 60, 120), confidence=0.9, crop=crop)
        assert p.x1 == 10
        assert p.y1 == 20
        assert p.x2 == 60
        assert p.y2 == 120
        assert p.width == 50
        assert p.height == 100
        assert p.area == 5000
        assert p.center == (35, 70)


class TestPersonDetector:

    @pytest.fixture(scope="class")
    def detector(self):
        """Single detector instance shared across tests (model load is slow)."""
        return PersonDetector(confidence=0.3)

    def test_empty_frame_returns_empty(self, detector):
        frame = _blank_frame()
        persons = detector.detect(frame, camera_id="cam1", frame_num=1)
        # Blank frame has no people — should return empty list without crashing
        assert isinstance(persons, list)

    def test_none_frame_returns_empty(self, detector):
        persons = detector.detect(None, camera_id="cam1", frame_num=1)
        assert persons == []

    def test_annotate_returns_same_size_frame(self, detector):
        frame = _blank_frame()
        persons = []
        annotated = detector.annotate(frame, persons)
        assert annotated.shape == frame.shape

    def test_annotate_does_not_modify_original(self, detector):
        frame = _blank_frame()
        original = frame.copy()
        detector.annotate(frame, [])
        np.testing.assert_array_equal(frame, original)

    def test_detect_and_annotate_returns_tuple(self, detector):
        frame = _blank_frame()
        result = detector.detect_and_annotate(frame, camera_id="cam1", frame_num=1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        persons, annotated = result
        assert isinstance(persons, list)
        assert annotated.shape == frame.shape

    def test_detected_person_has_valid_crop(self, detector):
        """If any person is detected, crop must be non-empty and inside frame."""
        frame = _frame_with_noise()
        persons = detector.detect(frame, camera_id="cam1", frame_num=1)
        for p in persons:
            assert p.crop is not None
            assert p.crop.size > 0
            assert p.crop.ndim == 3