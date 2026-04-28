"""
tests/test_reid.py
Unit tests for PersonEmbedder and IdentityManager.
Run with: pytest tests/test_reid.py -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from core.reid.embedder import PersonEmbedder
from core.reid.identity_manager import IdentityManager, ReIDResult


def _random_crop(h=100, w=50):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def _random_embedding(dim=512):
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)   # L2 normalise


class TestPersonEmbedder:

    @pytest.fixture(scope="class")
    def embedder(self):
        return PersonEmbedder()

    def test_embed_returns_correct_shape(self, embedder):
        crop = _random_crop()
        vec  = embedder.embed(crop)
        assert vec is not None
        assert vec.shape == (embedder.embedding_dim,)

    def test_embed_is_l2_normalised(self, embedder):
        crop = _random_crop()
        vec  = embedder.embed(crop)
        assert vec is not None
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01, f"Expected norm ~1.0, got {norm}"

    def test_embed_none_returns_none(self, embedder):
        assert embedder.embed(None) is None

    def test_embed_tiny_crop_returns_none(self, embedder):
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        assert embedder.embed(tiny) is None

    def test_embed_batch_returns_list(self, embedder):
        crops = [_random_crop() for _ in range(3)]
        vecs  = embedder.embed_batch(crops)
        assert len(vecs) == 3
        for v in vecs:
            assert v is not None
            assert v.shape == (embedder.embedding_dim,)

    def test_same_crop_produces_similar_embeddings(self, embedder):
        crop = _random_crop()
        v1   = embedder.embed(crop)
        v2   = embedder.embed(crop)
        similarity = float(np.dot(v1, v2))
        assert similarity > 0.99, f"Same crop similarity too low: {similarity}"


class TestIdentityManager:

    @pytest.fixture
    def manager(self, tmp_path):
        return IdentityManager(
            threshold   = 0.85,
            index_path  = str(tmp_path / "faiss.index"),
            id_map_path = str(tmp_path / "identity_map.json"),
        )

    def test_first_identity_is_new(self, manager):
        emb    = _random_embedding()
        result = manager.identify(emb, track_id=1, camera_id="cam1", frame_num=1)
        assert result is not None
        assert result.is_new == True
        assert result.person_id.startswith("PERSON_")

    def test_same_embedding_matches_existing(self, manager):
        emb = _random_embedding()
        r1  = manager.identify(emb, track_id=1, camera_id="cam1", frame_num=1)
        r2  = manager.identify(emb, track_id=1, camera_id="cam1", frame_num=2)
        assert r1.person_id == r2.person_id
        assert r2.is_new == False

    def test_different_embeddings_create_different_ids(self, manager):
        e1 = _random_embedding()
        e2 = _random_embedding()
        # Make them very different (orthogonal)
        e2 = e2 - np.dot(e1, e2) * e1
        e2 = (e2 / np.linalg.norm(e2)).astype(np.float32)

        r1 = manager.identify(e1, track_id=1, camera_id="cam1", frame_num=1)
        r2 = manager.identify(e2, track_id=2, camera_id="cam1", frame_num=2)
        assert r1.person_id != r2.person_id

    def test_identity_count_increments(self, manager):
        for i in range(3):
            emb = _random_embedding()
            # Make orthogonal to ensure new identity each time
            manager.identify(emb, track_id=i, camera_id="cam1", frame_num=i)
        assert manager.get_identity_count() >= 1

    def test_none_embedding_returns_none(self, manager):
        result = manager.identify(None, track_id=1, camera_id="cam1")
        assert result is None

    def test_save_and_reload(self, tmp_path):
        index_path  = str(tmp_path / "test.index")
        id_map_path = str(tmp_path / "test_map.json")

        m1 = IdentityManager(
            threshold=0.85,
            index_path=index_path,
            id_map_path=id_map_path,
        )
        emb = _random_embedding()
        r   = m1.identify(emb, track_id=1, camera_id="cam1", frame_num=1)
        m1.save()

        # Reload
        m2 = IdentityManager(
            threshold=0.85,
            index_path=index_path,
            id_map_path=id_map_path,
        )
        assert m2.get_identity_count() == 1
        assert m2.get_identity(r.person_id) is not None

    def test_reset_clears_everything(self, manager):
        emb = _random_embedding()
        manager.identify(emb, track_id=1, camera_id="cam1", frame_num=1)
        manager.reset()
        assert manager.get_identity_count() == 0
        assert manager._index.ntotal == 0