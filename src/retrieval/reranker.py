import os
from collections.abc import Sequence

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

from ..core import quiet_output


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Score (question, chunk) pairs with a cross-encoder, loaded on first use.

    Unlike the bi-encoder used for retrieval, a cross-encoder reads the question
    and the chunk together, so it judges relevance more precisely but is too slow
    to run over the whole index. It only re-orders the retrieved candidates.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None

    def load(self) -> None:
        """Load the cross-encoder weights once."""
        if self.model is not None:
            return

        import torch
        from sentence_transformers import CrossEncoder

        # Half precision on GPU halves the memory taken from the LLM.
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        with quiet_output():
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                model_kwargs={"torch_dtype": dtype},
            )

    def scores(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Return one relevance score per text; higher means more relevant."""
        if not texts:
            return np.empty(0, dtype="float32")

        self.load()
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype="float32")
