import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval import VectorStore


def _chunks(count: int) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "text": f"text {i}", "source": "doc.pdf", "page": i + 1, "chunk_index": 0}
        for i in range(count)
    ]


def _vectors(count: int, dimension: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(count, dimension)).astype("float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class VectorStoreConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _saved_store(self, pipeline: dict) -> VectorStore:
        store = VectorStore(self.storage, pipeline_config=pipeline)
        store.build(_vectors(4), _chunks(4))
        store.save()
        return store

    def test_same_settings_have_no_mismatch(self) -> None:
        pipeline = {"embedding_model": "m1", "chunking_strategy": "recursive"}
        self._saved_store(pipeline)
        self.assertEqual(VectorStore(self.storage, pipeline_config=pipeline).config_mismatches(), {})

    def test_changed_chunking_strategy_is_reported(self) -> None:
        self._saved_store({"embedding_model": "m1", "chunking_strategy": "recursive"})
        current = VectorStore(
            self.storage,
            pipeline_config={"embedding_model": "m1", "chunking_strategy": "parent_child"},
        )
        self.assertEqual(
            current.config_mismatches(),
            {"pipeline.chunking_strategy": ("recursive", "parent_child")},
        )

    def test_old_config_without_pipeline_is_reported(self) -> None:
        self.storage.joinpath("index_config.json").write_text(
            json.dumps({"index_type": "flat", "hnsw_m": 32, "ivf_nlist": 64, "ivf_nprobe": 8}),
            encoding="utf-8",
        )
        current = VectorStore(self.storage, pipeline_config={"embedding_model": "m1"})
        self.assertEqual(current.config_mismatches(), {"pipeline.embedding_model": (None, "m1")})

    def test_save_load_and_search(self) -> None:
        vectors = _vectors(4)
        store = VectorStore(self.storage)
        store.build(vectors, _chunks(4))
        store.save()

        loaded = VectorStore(self.storage)
        self.assertTrue(loaded.exists())
        loaded.load()
        results = loaded.search(vectors[2], top_k=1)
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=5)


class SearchPositionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Path(self._tmp.name)
        self.vectors = _vectors(200)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_every_index_type_ranks_the_given_positions_exactly(self) -> None:
        positions = [3, 50, 120, 199]
        query = self.vectors[120]
        expected = sorted(positions, key=lambda position: -float(self.vectors[position] @ query))
        for index_type in ("flat", "hnsw", "ivf"):
            with self.subTest(index_type=index_type):
                store = VectorStore(self.storage / index_type, index_type=index_type, ivf_nprobe=1)
                store.build(self.vectors, _chunks(200))
                results = store.search_positions(query, positions, top_k=4)
                self.assertEqual([result["position"] for result in results], expected)
                self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=5)

    def test_loaded_ivf_index_can_score_positions(self) -> None:
        store = VectorStore(self.storage, index_type="ivf", ivf_nprobe=1)
        store.build(self.vectors, _chunks(200))
        store.save()
        loaded = VectorStore(self.storage, index_type="ivf", ivf_nprobe=1)
        loaded.load()
        results = loaded.search_positions(self.vectors[7], [7, 8], top_k=1)
        self.assertEqual(results[0]["chunk_id"], "c7")

    def test_no_positions_returns_nothing(self) -> None:
        store = VectorStore(self.storage)
        store.build(self.vectors, _chunks(200))
        self.assertEqual(store.search_positions(self.vectors[0], [], top_k=3), [])


if __name__ == "__main__":
    unittest.main()
