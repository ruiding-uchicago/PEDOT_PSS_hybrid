"""Frozen original cased MatBERT field encoder, with no corpus or label access.

Uses the original 12 fields, 128-token right truncation, and nonspecial-token
mean pooling. Only the small supervised rank64 head is refitted elsewhere.

Inference port: the field projection runs here; the transformer forward pass runs in an
isolated torch-only worker process (``pedot_pss._encoder``), or in this process with a
single PyTorch thread when ``PEDOT_PSS_ENCODER=inprocess``. The numerical procedure
(unique strings ordered by token count, batches of four, float64 mean pooling, float32
storage) is unchanged; see ``pedot_pss._matbert_worker``.
"""
from __future__ import annotations
import time
import numpy as np
from .recipe_text_residual import FIELDS, project_text


class FrozenMatBERT:
    def __init__(self, asset_directory=None, threads=2):
        self.asset_directory = asset_directory
        # Retained for pickle compatibility. The original fit used 2 PyTorch threads; the
        # inference worker uses PEDOT_PSS_ENCODER_THREADS (default 1) for reproducibility.
        self.threads = threads
        self._encoder = None
        self._tokenizer = None

    def encode(self, rows, progress=None):
        audits = [project_text(row) for row in rows]
        matrix = np.zeros((len(rows), len(FIELDS), 768), dtype=np.float32)
        slots = {}
        for i, audit in enumerate(audits):
            for j, field in enumerate(FIELDS):
                if audit["field_mask"][field]:
                    slots.setdefault(audit["field_text"][field], []).append((i, j))
        if not slots:
            return matrix
        from .._encoder import encode_texts
        started = time.monotonic()
        texts = list(slots)
        vectors = encode_texts(texts, self.asset_directory)
        for value, vector in zip(texts, vectors):
            for i, j in slots[value]:
                matrix[i, j] = vector
        if progress is not None:
            progress({"event": "frozen_matbert_encoding", "completed_unique_strings": len(texts),
                      "total_unique_strings": len(texts), "seconds": time.monotonic() - started})
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite MatBERT field vectors")
        return matrix

    def close(self):
        self._encoder = self._tokenizer = None

    def __getstate__(self):
        # Pretrained weights are an explicitly bundled model asset, not a
        # duplicated checkpoint hidden in each CV or deployment pickle.
        state = self.__dict__.copy()
        state["_encoder"] = state["_tokenizer"] = None
        return state
