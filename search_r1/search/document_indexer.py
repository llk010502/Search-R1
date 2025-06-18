import numpy as np
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


def slice_document(document: str, max_tokens: int = 128, stride: int = 64) -> List[str]:
    tokens = document.split()
    slices = []
    for start in range(0, len(tokens), stride):
        part = tokens[start:start + max_tokens]
        if not part:
            continue
        slices.append(" ".join(part))
    return slices


class DocumentIndexer:
    """A lightweight document indexer backed by TF-IDF vector search."""

    def __init__(self, document: str, max_tokens: int = 128, stride: int = 64):
        self.passages = slice_document(document, max_tokens=max_tokens, stride=stride)
        self.vectorizer = TfidfVectorizer()
        self._vectors = self.vectorizer.fit_transform(self.passages)

    def search(self, query: str, topk: int = 3) -> List[Dict]:
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self._vectors.T).toarray()[0]
        order = np.argsort(scores)[::-1][:topk]
        return [
            {"document": {"contents": self.passages[i]}, "score": float(scores[i])}
            for i in order
        ]

    def batch_search(self, queries: List[str], topk: int = 3) -> List[List[Dict]]:
        q_vecs = self.vectorizer.transform(queries)
        score_matrix = q_vecs @ self._vectors.T
        results = []
        for row in score_matrix:
            scores = row.toarray()[0]
            order = np.argsort(scores)[::-1][:topk]
            results.append([
                {"document": {"contents": self.passages[i]}, "score": float(scores[i])}
                for i in order
            ])
        return results
