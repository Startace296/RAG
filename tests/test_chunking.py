import unittest

from src.chunking import split_documents


def _page(text: str, page: int = 1) -> dict:
    return {"document_id": "doc", "source": "doc.pdf", "page": page, "text": text}


class ParentChildChunkingTest(unittest.TestCase):
    def test_children_of_same_parent_share_parent_id(self) -> None:
        words = " ".join(f"w{i}" for i in range(100))
        chunks = split_documents(
            [_page(words)],
            chunk_size=20,
            chunk_overlap=0,
            chunking_strategy="parent_child",
            min_chunk_size=5,
        )

        by_parent: dict[str, set[str]] = {}
        for chunk in chunks:
            by_parent.setdefault(chunk["parent_text"], set()).add(chunk["parent_id"])

        # Parent size is 2 x chunk_size, so 100 words -> 3 parents.
        self.assertEqual(len(by_parent), 3)
        for parent_ids in by_parent.values():
            self.assertEqual(len(parent_ids), 1)
        self.assertEqual(len({chunk["parent_id"] for chunk in chunks}), 3)

    def test_child_text_is_inside_parent_text(self) -> None:
        words = " ".join(f"w{i}" for i in range(60))
        chunks = split_documents(
            [_page(words)],
            chunk_size=20,
            chunk_overlap=5,
            chunking_strategy="parent_child",
            min_chunk_size=5,
        )
        for chunk in chunks:
            self.assertIn(chunk["text"], chunk["parent_text"])
            self.assertNotIn("PARENT_TEXT", chunk["text"])

    def test_other_strategies_have_no_parent(self) -> None:
        chunks = split_documents(
            [_page("Câu một. Câu hai. Câu ba.")],
            chunk_size=20,
            chunk_overlap=0,
            chunking_strategy="recursive",
            min_chunk_size=1,
        )
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk["parent_id"], "")
            self.assertEqual(chunk["parent_text"], "")

    def test_document_chunk_index_counts_across_pages(self) -> None:
        chunks = split_documents(
            [_page("a b c d e", page=1), _page("f g h i j", page=2)],
            chunk_size=3,
            chunk_overlap=0,
            chunking_strategy="fixed_word",
            min_chunk_size=1,
        )
        self.assertEqual(
            [chunk["document_chunk_index"] for chunk in chunks],
            list(range(len(chunks))),
        )
        self.assertEqual([chunk["page_chunk_index"] for chunk in chunks], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
