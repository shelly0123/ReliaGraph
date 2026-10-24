"""Low-overhead runtime profiling and optional local embedding/NLI model adapters.

The core graph pipeline stays usable without downloading any model. Models load
only when explicitly requested, run locally, and expose timing measurements.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import sqlite3
import time
import re
import os
import platform


@dataclass
class TimingEvent:
    stage: str
    seconds: float
    item_count: int = 0
    note: str = ""


class RuntimeProfiler:
    """Collect wall-clock timings without a third-party dependency."""
    def __init__(self) -> None:
        self.events: list[TimingEvent] = []

    @contextmanager
    def measure(self, stage: str, item_count: int = 0, note: str = "") -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(TimingEvent(stage, round(time.perf_counter() - start, 6), item_count, note))

    def add(self, stage: str, seconds: float, item_count: int = 0, note: str = "") -> None:
        self.events.append(TimingEvent(stage, round(float(seconds), 6), int(item_count), note))

    def report(self) -> dict[str, Any]:
        total = sum(event.seconds for event in self.events)
        return {
            "total_measured_seconds": round(total, 6),
            "environment": {
                "python_version": platform.python_version(),
                "operating_system": platform.system(),
                "machine_architecture": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
            },
            "events": [asdict(event) for event in self.events],
            "notes": [
                "Measurements are local to this run and machine.",
                "Model download time is separate from inference time unless explicitly included.",
                "Do not compare times from different machines without noting hardware and cache state.",
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")


def normalise_cache_text(text: str) -> str:
    # Normalize whitespace only. Case-folding could incorrectly reuse an embedding
    # across case-sensitive labels/acronyms such as CS vs cs.
    return re.sub(r"\s+", " ", str(text).strip())


class SQLiteVectorCache:
    """Small persistent cache for local embedding vectors, keyed by model and text.

    Only the derived vector and a text hash are stored; source text is not copied
    into the cache. SQLite is adequate for the prototype and can later be replaced.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                model_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(model_name, text_hash)
            )""")

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(normalise_cache_text(text).encode("utf-8")).hexdigest()

    def get_many(self, model_name: str, texts: list[str]) -> dict[str, list[float]]:
        hashes = {self.text_hash(text): text for text in texts}
        if not hashes:
            return {}
        found: dict[str, list[float]] = {}
        with sqlite3.connect(self.path) as con:
            for text_hash in hashes:
                row = con.execute(
                    "SELECT vector_json FROM embeddings WHERE model_name=? AND text_hash=?",
                    (model_name, text_hash),
                ).fetchone()
                if row:
                    found[text_hash] = json.loads(row[0])
        return found

    def put_many(self, model_name: str, pairs: list[tuple[str, list[float]]]) -> None:
        rows = [(model_name, self.text_hash(text), json.dumps(vector)) for text, vector in pairs]
        if not rows:
            return
        with sqlite3.connect(self.path) as con:
            con.executemany(
                "INSERT OR REPLACE INTO embeddings(model_name,text_hash,vector_json) VALUES(?,?,?)", rows
            )


class LocalSentenceEmbedder:
    """Lazy local embeddings; optional dependency is sentence-transformers.

    Default model: all-MiniLM-L6-v2 (Apache-2.0 model card, as checked 2026-10-24).
    The first opt-in use may download model files; later runs use local cache.
    """
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_db: str | Path = "data/cache/embeddings.sqlite3",
                 device: str = "cpu", batch_size: int = 32, max_length: int = 256) -> None:
        self.model_name = model_name
        self.cache = SQLiteVectorCache(cache_db)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self._model = None
        self.load_seconds: float | None = None
        self.last_encode_seconds: float = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        start = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Optional embeddings require `pip install -r requirements-optional-models.txt`. "
                "The base rule-based pipeline does not need this package."
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        try:
            self._model.max_seq_length = self.max_length
        except Exception:
            pass
        self.load_seconds = time.perf_counter() - start

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        start = time.perf_counter()
        norm_to_original: dict[str, str] = {}
        for text in texts:
            norm_to_original.setdefault(normalise_cache_text(text), str(text))
        unique = list(norm_to_original.values())
        cached = self.cache.get_many(self.model_name, unique)
        self.cache_hits += len(cached)
        missing = [text for text in unique if self.cache.text_hash(text) not in cached]
        self.cache_misses += len(missing)
        if missing:
            vectors = self._model.encode(
                missing, batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            self.cache.put_many(self.model_name, [(t, [float(x) for x in v]) for t, v in zip(missing, vectors)])
            cached.update({self.cache.text_hash(t): [float(x) for x in v] for t, v in zip(missing, vectors)})
        result = [cached[self.cache.text_hash(text)] for text in texts]
        self.last_encode_seconds = time.perf_counter() - start
        return result

    def similarity(self, text_a: str, text_b: str) -> float:
        va, vb = self.encode([text_a, text_b])
        # Vectors are normalized; dot product equals cosine similarity.
        return max(-1.0, min(1.0, sum(a * b for a, b in zip(va, vb))))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "provider": "local_sentence_transformers",
            "model_name": self.model_name,
            "device": self.device,
            "load_seconds": round(self.load_seconds, 4) if self.load_seconds is not None else None,
            "last_encode_seconds": round(self.last_encode_seconds, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_path": str(self.cache.path),
        }


class LocalNLIReviewer:
    """Optional local NLI checker. Returns model signals; never auto-verifies claims."""
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", max_length: int = 256) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max(64, int(max_length))
        self._model = None
        self.load_seconds: float | None = None
        self.last_score_seconds: float = 0.0
        self._memory_cache: dict[tuple[str, str], dict[str, float]] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        start = time.perf_counter()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Optional NLI requires `pip install -r requirements-optional-models.txt`. "
                "The base rule-based pipeline does not need this package."
            ) from exc
        self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        self.load_seconds = time.perf_counter() - start

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[dict[str, float]]:
        if not pairs:
            return []
        self._load()
        start = time.perf_counter()
        results: list[dict[str, float] | None] = [None] * len(pairs)
        missing_pairs: list[tuple[str, str]] = []
        missing_indices: list[int] = []
        for i, (premise, hypothesis) in enumerate(pairs):
            key = (premise.strip(), hypothesis.strip())
            if key in self._memory_cache:
                results[i] = self._memory_cache[key]
            else:
                missing_pairs.append(key)
                missing_indices.append(i)
        if missing_pairs:
            raw = self._model.predict(missing_pairs, batch_size=max(1, int(batch_size)), show_progress_bar=False)
            # CrossEncoder model card order is contradiction, entailment, neutral.
            # Prefer model config labels where available to avoid relying on class order.
            config_labels = {}
            try:
                config_labels = {int(k): str(v).casefold() for k, v in self._model.model.config.id2label.items()}
            except Exception:
                config_labels = {}
            import numpy as np
            arr = np.asarray(raw)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Convert logits to probabilities. If the model is configured to return
            # probability vectors, softmax remains a normalized monotonic transform.
            shifted = arr - arr.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=1, keepdims=True)
            for idx, row in zip(missing_indices, probs):
                labels = {i: config_labels.get(i, "") for i in range(len(row))}
                result: dict[str, float] = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
                fallback_order = ["contradiction", "entailment", "neutral"]
                for j, score in enumerate(row):
                    label = labels.get(j, "")
                    if "contradiction" in label:
                        key = "contradiction"
                    elif "entail" in label:
                        key = "entailment"
                    elif "neutral" in label:
                        key = "neutral"
                    else:
                        key = fallback_order[j] if j < len(fallback_order) else f"class_{j}"
                    result[key] = float(score)
                pair_key = pairs[idx]
                self._memory_cache[(pair_key[0].strip(), pair_key[1].strip())] = result
                results[idx] = result
        self.last_score_seconds = time.perf_counter() - start
        return [value or {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0} for value in results]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "provider": "local_cross_encoder_nli",
            "model_name": self.model_name,
            "device": self.device,
            "load_seconds": round(self.load_seconds, 4) if self.load_seconds is not None else None,
            "last_score_seconds": round(self.last_score_seconds, 4),
            "cache_entries": len(self._memory_cache),
        }
