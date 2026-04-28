"""
core/reid/identity_manager.py
STEP 4b — Identity Manager using FAISS

Maintains a database of known person identities.
When a new person crop arrives:
  1. Embed it → 512-dim vector
  2. Search FAISS index for similar vectors
  3. If match found (similarity >= threshold) → assign existing PERSON_ID
  4. If no match → create new PERSON_ID, store embedding

This is what enables cross-camera tracking:
  - cam1 sees PERSON_001 walking left
  - cam3 sees same person enter → FAISS matches → same PERSON_001

Design decisions:
  - FAISS IndexFlatIP = exact inner product search (fast for <10k identities)
  - Embeddings are L2-normalised so inner product = cosine similarity
  - Multiple embeddings stored per identity (appearance can vary with lighting)
  - Identity map persisted to JSON so identities survive restarts
  - FAISS threshold 0.85 (82-87% sweet spot from report §12)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

# torch MUST be imported before faiss on macOS Apple Silicon
# to prevent segmentation fault (known PyTorch + FAISS conflict)
import torch
import faiss

from config.settings import (
    FAISS_MATCH_THRESHOLD,
    FAISS_INDEX_PATH,
    IDENTITY_MAP_PATH,
    EMBEDDING_DIM,
)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Identity:
    """
    A known person identity in the database.

    person_id     : stable string ID e.g. "PERSON_00001"
    track_ids     : set of tracker IDs seen for this person (across cameras)
    camera_ids    : set of cameras this person was seen on
    embedding_count: how many embeddings stored for this identity
    first_seen    : frame number when first detected
    last_seen     : most recent frame number
    """
    person_id       : str
    track_ids       : set  = field(default_factory=set)
    camera_ids      : set  = field(default_factory=set)
    embedding_count : int  = 0
    first_seen      : int  = 0
    last_seen       : int  = 0

    def to_dict(self) -> dict:
        return {
            "person_id"       : self.person_id,
            "track_ids"       : list(self.track_ids),
            "camera_ids"      : list(self.camera_ids),
            "embedding_count" : self.embedding_count,
            "first_seen"      : self.first_seen,
            "last_seen"       : self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Identity":
        obj = cls(person_id=d["person_id"])
        obj.track_ids       = set(d.get("track_ids", []))
        obj.camera_ids      = set(d.get("camera_ids", []))
        obj.embedding_count = d.get("embedding_count", 0)
        obj.first_seen      = d.get("first_seen", 0)
        obj.last_seen       = d.get("last_seen", 0)
        return obj


@dataclass
class ReIDResult:
    """
    Result of a Re-ID lookup for one person.

    person_id   : assigned PERSON_ID (new or existing)
    is_new      : True = new identity created, False = matched existing
    similarity  : cosine similarity to best match (0.0 → 1.0)
    track_id    : tracker ID from this frame
    camera_id   : camera where person was seen
    """
    person_id  : str
    is_new     : bool
    similarity : float
    track_id   : int
    camera_id  : str


# ── Identity Manager ──────────────────────────────────────────────────────────

class IdentityManager:
    """
    FAISS-backed person identity store.

    Usage:
        manager = IdentityManager()

        # Identify a person from their crop embedding
        result = manager.identify(
            embedding  = embedder.embed(person.crop),
            track_id   = person.track_id,
            camera_id  = person.camera_id,
            frame_num  = person.frame_num,
        )
        print(result.person_id)   # "PERSON_00003"
        print(result.is_new)      # False = matched existing identity
    """

    MAX_EMBEDDINGS_PER_IDENTITY = 10   # store up to 10 looks per person

    def __init__(
        self,
        threshold     : float = FAISS_MATCH_THRESHOLD,
        index_path    : str   = FAISS_INDEX_PATH,
        id_map_path   : str   = IDENTITY_MAP_PATH,
        embedding_dim : int   = EMBEDDING_DIM,
    ):
        self.threshold     = threshold
        self.index_path    = Path(index_path)
        self.id_map_path   = Path(id_map_path)
        self.embedding_dim = embedding_dim

        # FAISS index — IndexFlatIP = exact inner product (cosine when L2-normed)
        self._index : faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)

        # Maps FAISS vector index → person_id
        self._vector_to_person : List[str] = []

        # Maps person_id → Identity object
        self._identities : Dict[str, Identity] = {}

        # Counter for generating new IDs
        self._next_id : int = 1

        # Load existing data if available
        self._load()
        logger.info(
            f"IdentityManager ready | "
            f"threshold={threshold} | "
            f"known identities={len(self._identities)} | "
            f"stored vectors={self._index.ntotal}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def identify(
        self,
        embedding : np.ndarray,
        track_id  : int,
        camera_id : str,
        frame_num : int = 0,
    ) -> Optional[ReIDResult]:
        """
        Main Re-ID method. Given an embedding, find or create an identity.

        Steps:
          1. Search FAISS for nearest neighbour
          2. If similarity >= threshold → match existing identity
          3. Else → create new identity
          4. Update identity metadata
          5. Periodically store new embedding for this identity

        Returns ReIDResult or None if embedding is invalid.
        """
        if embedding is None or embedding.size == 0:
            return None

        embedding = self._ensure_float32(embedding)

        # Search FAISS
        person_id, similarity, is_new = self._search_or_create(
            embedding, track_id, camera_id, frame_num
        )

        # Update identity metadata
        identity = self._identities[person_id]
        identity.track_ids.add(track_id)
        identity.camera_ids.add(camera_id)
        identity.last_seen = frame_num

        # Store embedding periodically (not every frame — too many vectors)
        if identity.embedding_count < self.MAX_EMBEDDINGS_PER_IDENTITY:
            self._add_to_index(embedding, person_id)
            identity.embedding_count += 1

        if is_new:
            logger.info(
                f"NEW identity created: {person_id} | "
                f"camera={camera_id} | track_id={track_id}"
            )
        else:
            logger.debug(
                f"Matched: {person_id} | "
                f"similarity={similarity:.3f} | "
                f"camera={camera_id} | track_id={track_id}"
            )

        return ReIDResult(
            person_id  = person_id,
            is_new     = is_new,
            similarity = similarity,
            track_id   = track_id,
            camera_id  = camera_id,
        )

    def get_identity(self, person_id: str) -> Optional[Identity]:
        return self._identities.get(person_id)

    def get_all_identities(self) -> List[Identity]:
        return list(self._identities.values())

    def get_identity_count(self) -> int:
        return len(self._identities)

    def save(self):
        """Persist FAISS index and identity map to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.id_map_path.parent.mkdir(parents=True, exist_ok=True)

            faiss.write_index(self._index, str(self.index_path))

            data = {
                "next_id"          : self._next_id,
                "vector_to_person" : self._vector_to_person,
                "identities"       : {
                    pid: ident.to_dict()
                    for pid, ident in self._identities.items()
                },
            }
            with open(self.id_map_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Saved {len(self._identities)} identities | "
                f"{self._index.ntotal} vectors"
            )
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def reset(self):
        """Clear all identities (useful for testing)."""
        self._index  = faiss.IndexFlatIP(self.embedding_dim)
        self._vector_to_person.clear()
        self._identities.clear()
        self._next_id = 1
        logger.info("IdentityManager reset.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _search_or_create(
        self,
        embedding : np.ndarray,
        track_id  : int,
        camera_id : str,
        frame_num : int,
    ) -> Tuple[str, float, bool]:
        """
        Search FAISS. Returns (person_id, similarity, is_new).
        """
        if self._index.ntotal == 0:
            # No identities yet — create first one
            return self._create_identity(frame_num), 0.0, True

        query = embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self._index.search(query, k=1)

        best_sim = float(similarities[0][0])
        best_idx = int(indices[0][0])

        if best_sim >= self.threshold and best_idx < len(self._vector_to_person):
            # Match found
            person_id = self._vector_to_person[best_idx]
            return person_id, best_sim, False
        else:
            # No match — new identity
            return self._create_identity(frame_num), best_sim, True

    def _create_identity(self, frame_num: int) -> str:
        person_id = str(self._next_id)   # simple integer: "1", "2", "3"...
        self._identities[person_id] = Identity(
            person_id  = person_id,
            first_seen = frame_num,
            last_seen  = frame_num,
        )
        self._next_id += 1
        return person_id

    def _add_to_index(self, embedding: np.ndarray, person_id: str):
        vec = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._vector_to_person.append(person_id)

    def _ensure_float32(self, embedding: np.ndarray) -> np.ndarray:
        if embedding.dtype != np.float32:
            return embedding.astype(np.float32)
        return embedding

    def _load(self):
        """Load persisted FAISS index and identity map if they exist."""
        try:
            if self.index_path.exists() and self.id_map_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                with open(self.id_map_path) as f:
                    data = json.load(f)
                self._next_id           = data.get("next_id", 1)
                self._vector_to_person  = data.get("vector_to_person", [])
                self._identities        = {
                    pid: Identity.from_dict(d)
                    for pid, d in data.get("identities", {}).items()
                }
                logger.info(
                    f"Loaded {len(self._identities)} identities from disk."
                )
        except Exception as e:
            logger.warning(f"Could not load saved identities: {e} — starting fresh.")


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from core.video.stream_reader import StreamReader
    from core.tracking.person_tracker import PersonTracker
    from core.reid.embedder import PersonEmbedder

    source = sys.argv[1] if len(sys.argv) > 1 else "data/cam1.mp4"
    print(f"\nStep 4 — Full Re-ID pipeline test on: {source}")
    print("Watch PERSON_IDs stay stable even as tracker IDs change.\n")

    embedder = PersonEmbedder()
    manager  = IdentityManager()
    tracker  = PersonTracker()
    reader   = StreamReader(source=source, frame_skip=2)

    import cv2
    try:
        for frame_num, frame in reader.frames():
            tracked = tracker.track(frame, camera_id="cam1", frame_num=frame_num)

            annotated = frame.copy()
            for person in tracked:
                vec    = embedder.embed(person.crop)
                result = manager.identify(
                    embedding = vec,
                    track_id  = person.track_id,
                    camera_id = person.camera_id,
                    frame_num = frame_num,
                )

                if result:
                    label = f"{result.person_id}"
                    if result.is_new:
                        label += " NEW"
                    x1,y1,x2,y2 = person.bbox
                    color = (0,255,0) if not result.is_new else (0,165,255)
                    cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(annotated, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            total = manager.get_identity_count()
            cv2.putText(annotated, f"Identities: {total}", (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            if frame_num % 60 == 0:
                print(f"Frame #{frame_num:5d} | "
                      f"Total identities: {total}")

            cv2.imshow("URG-IS | Step 4 — Re-Identification", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        manager.save()
        reader.stop()
        cv2.destroyAllWindows()
        print(f"\nFinal identities: {manager.get_identity_count()}")
        for ident in manager.get_all_identities():
            print(f"  {ident.person_id} | "
                  f"cameras={ident.camera_ids} | "
                  f"embeddings={ident.embedding_count}")
        print("Identities saved to data/embeddings/")