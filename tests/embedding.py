"""A stand-in for ChromaDB's embedding model in tests: hashed bags of words, deterministic and offline.

The real model (all-MiniLM-L6-v2) is downloaded on first use; tests must not depend on the network.
"""

import hashlib
import math
import re

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

DIMENSIONS = 512


class WordHashEmbedding(EmbeddingFunction[Documents]):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    @staticmethod
    def name() -> str:
        return "word_hash_test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordHashEmbedding":
        return WordHashEmbedding()

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
