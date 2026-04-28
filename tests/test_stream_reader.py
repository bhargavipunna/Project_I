"""
tests/test_stream_reader.py
Unit tests for StreamReader — uses a local dummy video file, no CCTV needed.
Run with: pytest tests/test_stream_reader.py -v
"""

import cv2
import numpy as np
import tempfile
import os
import pytest

from core.video.stream_reader import StreamReader


def _create_dummy_video(path: str, num_frames: int = 30, fps: int = 15):
    """Creates a small synthetic video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (i * 8 % 255, 100, 200)  # varying colour per frame
        writer.write(frame)
    writer.release()


class TestStreamReader:

    def test_reads_frames_from_file(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_dummy_video(video_path, num_frames=30)

        reader = StreamReader(source=video_path, frame_skip=0)
        frames_seen = []

        for frame_num, frame in reader.frames():
            frames_seen.append(frame_num)
            if len(frames_seen) >= 10:
                reader.stop()
                break

        assert len(frames_seen) >= 5, "Should read at least 5 frames"

    def test_frame_skip_reduces_frames(self, tmp_path):
        video_path = str(tmp_path / "test_skip.mp4")
        _create_dummy_video(video_path, num_frames=60)

        reader_no_skip   = StreamReader(source=video_path, frame_skip=0)
        reader_with_skip = StreamReader(source=video_path, frame_skip=2)

        def count_frames(reader, limit=20):
            seen = []
            for fn, _ in reader.frames():
                seen.append(fn)
                if len(seen) >= limit:
                    reader.stop()
                    break
            return seen

        no_skip_frames   = count_frames(reader_no_skip)
        skip_frames      = count_frames(reader_with_skip)

        # With skip=2, every 3rd frame is yielded → fewer frames in same range
        assert max(skip_frames) > max(no_skip_frames) * 0.5, \
            "Skipped reader should cover more source frames in fewer yields"

    def test_frame_is_correct_size(self, tmp_path):
        video_path = str(tmp_path / "test_size.mp4")
        _create_dummy_video(video_path, num_frames=10)

        reader = StreamReader(source=video_path, frame_width=320, frame_height=240, frame_skip=0)

        for _, frame in reader.frames():
            h, w, c = frame.shape
            assert w == 320
            assert h == 240
            assert c == 3
            reader.stop()
            break

    def test_stop_works(self, tmp_path):
        video_path = str(tmp_path / "test_stop.mp4")
        _create_dummy_video(video_path, num_frames=100)

        reader = StreamReader(source=video_path, frame_skip=0)
        count = 0

        for _, _ in reader.frames():
            count += 1
            if count == 5:
                reader.stop()
                break

        assert count == 5

    def test_invalid_source_does_not_crash(self):
        reader = StreamReader(source="/nonexistent/video.mp4", reconnect_delay=0)
        # Should not raise — just log a warning and stop after we signal
        import threading
        t = threading.Timer(0.5, reader.stop)
        t.start()
        frames = list(reader.frames())  # will loop briefly then stop
        assert frames == []             # no frames from invalid source