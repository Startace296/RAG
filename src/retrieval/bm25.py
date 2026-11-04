import math
import re
from collections import Counter
from typing import Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; Vietnamese syllables become separate tokens."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Okapi BM25 keyword scores over chunk texts, kept in memory.

    Dense embeddings blur a short fact inside a long chunk; BM25 still rewards
    the chunk for sharing the exact terms of the question (numbers, names, dates).
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_term_counts = [Counter(tokenize(text)) for text in texts]
        self.doc_lengths = np.array(
            [sum(counts.values()) for counts in self.doc_term_counts], dtype="float32"
        )
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(texts) else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self.doc_term_counts:
            document_frequency.update(counts.keys())
        total = len(texts)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.doc_term_counts)

    def scores(self, query: str) -> np.ndarray:
        """Return one BM25 score per indexed text, in index order."""
        scores = np.zeros(len(self), dtype="float32")
        if not len(self) or self.avg_doc_length == 0:
            return scores

        query_terms = [term for term in set(tokenize(query)) if term in self.idf]
        length_norm = self.k1 * (1 - self.b + self.b * self.doc_lengths / self.avg_doc_length)
        for position, counts in enumerate(self.doc_term_counts):
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency:
                    scores[position] += (
                        self.idf[term]
                        * frequency
                        * (self.k1 + 1)
                        / (frequency + length_norm[position])
                    )
        return scores

    def ranks(self, query: str) -> np.ndarray:
        """Return the 1-based BM25 rank of every indexed text."""
        order = np.argsort(-self.scores(query), kind="stable")
        ranks = np.empty(len(order), dtype="int64")
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks
