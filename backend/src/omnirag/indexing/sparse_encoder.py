"""Deterministic sparse term-frequency vectors for Qdrant's IDF modifier."""

import hashlib
import math
import re
from collections import Counter


class SparseEncoder:
    """Hash terms to stable indices; Qdrant applies corpus IDF at query time."""

    method = "hashed_tf_qdrant_idf"
    tokenizer_version = "unicode_word_casefold_v1"

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        counts = Counter(tokens)
        pairs: dict[int, float] = {}
        for term, frequency in counts.items():
            index = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big")
            # Sublinear term frequency; collision values are accumulated deterministically.
            pairs[index] = pairs.get(index, 0.0) + 1.0 + math.log(frequency)
        indices = sorted(pairs)
        return indices, [pairs[index] for index in indices]
