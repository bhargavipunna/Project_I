"""
tests/test_interaction.py
Unit tests for InteractionDetector.
Run with: pytest tests/test_interaction.py -v
"""

import time
import pytest
from unittest.mock import patch

from core.interaction.interaction_detector import (
    InteractionDetector, InteractionEvent, ProximityState
)


class TestInteractionDetector:

    @pytest.fixture
    def detector(self):
        return InteractionDetector(
            distance_m   = 1.5,
            duration_s   = 0.1,   # short for tests
            pixels_per_m = 100.0,
        )

    def test_no_interaction_with_one_person(self, detector):
        events = detector.update(
            persons   = [("PERSON_00001", (100, 100))],
            camera_id = "cam1",
            frame_num = 1,
        )
        assert events == []

    def test_no_interaction_when_far_apart(self, detector):
        # 500 pixels apart = 5 metres > 1.5m threshold
        events = detector.update(
            persons   = [
                ("PERSON_00001", (100, 100)),
                ("PERSON_00002", (600, 100)),
            ],
            camera_id = "cam1",
            frame_num = 1,
        )
        assert events == []

    def test_proximity_starts_when_close(self, detector):
        # 50 pixels = 0.5m < 1.5m threshold
        detector.update(
            persons   = [
                ("PERSON_00001", (100, 100)),
                ("PERSON_00002", (150, 100)),
            ],
            camera_id = "cam1",
            frame_num = 1,
        )
        assert len(detector.get_active_proximities()) == 1

    def test_interaction_event_fired_after_duration(self, detector):
        persons = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (150, 100)),
        ]

        # First frame — start proximity
        detector.update(persons=persons, camera_id="cam1", frame_num=1)

        # Wait longer than duration threshold (0.1s in test fixture)
        time.sleep(0.15)

        # Second frame — should fire event
        events = detector.update(persons=persons, camera_id="cam1", frame_num=2)
        assert len(events) == 1
        assert events[0].person_id_a in ["PERSON_00001", "PERSON_00002"]
        assert events[0].person_id_b in ["PERSON_00001", "PERSON_00002"]
        assert events[0].duration_s >= 0.1

    def test_event_fired_only_once_per_proximity(self, detector):
        persons = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (150, 100)),
        ]
        detector.update(persons=persons, camera_id="cam1", frame_num=1)
        time.sleep(0.15)

        events1 = detector.update(persons=persons, camera_id="cam1", frame_num=2)
        events2 = detector.update(persons=persons, camera_id="cam1", frame_num=3)
        events3 = detector.update(persons=persons, camera_id="cam1", frame_num=4)

        assert len(events1) == 1
        assert len(events2) == 0   # already fired — no repeat
        assert len(events3) == 0

    def test_proximity_cleared_when_people_move_apart(self, detector):
        close = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (150, 100)),
        ]
        far = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (600, 100)),
        ]
        detector.update(persons=close, camera_id="cam1", frame_num=1)
        assert len(detector.get_active_proximities()) == 1

        detector.update(persons=far, camera_id="cam1", frame_num=2)
        assert len(detector.get_active_proximities()) == 0

    def test_new_pair_flag(self, detector):
        persons = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (150, 100)),
        ]
        detector.update(persons=persons, camera_id="cam1", frame_num=1)
        time.sleep(0.15)
        events = detector.update(persons=persons, camera_id="cam1", frame_num=2)

        assert events[0].is_new_pair == True

    def test_pair_key_is_order_independent(self, detector):
        e1 = InteractionEvent(
            person_id_a="PERSON_00001", person_id_b="PERSON_00002",
            camera_id="cam1", frame_num=1, start_time=0,
            duration_s=2.0, distance_m=1.0
        )
        e2 = InteractionEvent(
            person_id_a="PERSON_00002", person_id_b="PERSON_00001",
            camera_id="cam1", frame_num=1, start_time=0,
            duration_s=2.0, distance_m=1.0
        )
        assert e1.pair_key == e2.pair_key

    def test_pixel_distance_to_metres(self, detector):
        # 100 pixels at 100px/m = 1.0 metre
        dist = detector._pixel_distance_to_metres((0, 0), (100, 0))
        assert abs(dist - 1.0) < 0.01

    def test_reset_clears_everything(self, detector):
        persons = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (150, 100)),
        ]
        detector.update(persons=persons, camera_id="cam1", frame_num=1)
        detector.reset()
        assert len(detector.get_active_proximities()) == 0
        assert detector.get_interaction_count() == 0

    def test_get_pair_interaction_count(self, detector):
        persons = [
            ("PERSON_00001", (100, 100)),
            ("PERSON_00002", (150, 100)),
        ]
        detector.update(persons=persons, camera_id="cam1", frame_num=1)
        time.sleep(0.15)
        detector.update(persons=persons, camera_id="cam1", frame_num=2)

        count = detector.get_pair_interaction_count("PERSON_00001", "PERSON_00002")
        assert count == 1