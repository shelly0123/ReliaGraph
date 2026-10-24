"""Practical CIA-triad controls for ReliaGraph's local-first prototype.

These are defence-in-depth helpers, not a claim of formal certification.
- Confidentiality: minimize raw-text logs and default to local model providers.
- Integrity: content hashes and a tamper-evident hash-chained event log.
- Availability: controlled model fallback/circuit-break state and bounded processing metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import time


@dataclass(frozen=True)
class DataHandlingPolicy:
    local_only: bool = True
    log_raw_text: bool = False
    store_full_source_text_in_audit_log: bool = False
    allow_remote_model_inference: bool = False
    retain_deleted_source_text: bool = False

    def validate_provider(self, provider: str) -> None:
        normalized = str(provider or "local").casefold().strip()
        if self.local_only and normalized not in {"local", "offline", "none", "rule_based"}:
            raise PermissionError("Local-only policy blocks a remote model provider.")
        if not self.allow_remote_model_inference and normalized in {"openai", "anthropic", "remote_api", "cloud_api"}:
            raise PermissionError("Remote model inference is disabled by policy.")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def safe_log_record(event: str, *, doc_id: str = "", content_hash: str = "", metrics: Mapping[str, Any] | None = None,
                    raw_text: str | None = None, policy: DataHandlingPolicy | None = None) -> dict[str, Any]:
    """Create a non-content log record; raw text is omitted by default."""
    policy = policy or DataHandlingPolicy()
    row = {"event": str(event), "doc_id": str(doc_id), "content_hash": str(content_hash),
           "timestamp_unix": round(time.time(), 3), "metrics": dict(metrics or {})}
    if policy.log_raw_text and raw_text is not None:
        row["raw_text"] = raw_text
    return row


class HashChainAuditLog:
    """Tamper-evident JSONL audit log with cached append state.

    Full scans occur on initialization, when the on-disk signature changes, and when
    ``verify()`` is explicitly called. Normal append is O(1) relative to past event count
    for the single-writer local prototype. This avoids O(n^2) ingestion for 500+ documents.

    The log is tamper-evident, not digitally signed. Protect the audit file and anchor its
    head hash externally for higher-assurance deployment; this class does not provide a
    cross-process lock.
    """
    GENESIS = "0" * 64

    def __init__(self, path: str | Path):
        import threading
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.RLock()
        self._cached_signature = None
        self._cached_head = self.GENESIS
        self._cached_count = 0
        self._cached_valid = True
        self._cached_problems: list[str] = []
        self.verify()

    @staticmethod
    def _event_hash(event_without_hash: Mapping[str, Any]) -> str:
        blob = json.dumps(dict(event_without_hash), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _signature(self):
        st = self.path.stat()
        return (int(st.st_size), int(st.st_mtime_ns))

    def _scan(self) -> dict[str, Any]:
        previous = self.GENESIS
        count = 0
        problems: list[str] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            lines = []
            problems.append(f"read_error_{type(exc).__name__}")
        for line_number, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                problems.append(f"invalid_json_line_{line_number}")
                continue
            event_hash = row.pop("event_hash", None)
            if row.get("previous_hash") != previous:
                problems.append(f"broken_previous_hash_line_{line_number}")
            expected = self._event_hash(row)
            if event_hash != expected:
                problems.append(f"event_hash_mismatch_line_{line_number}")
            previous = str(event_hash or "")
            count += 1
        self._cached_head = previous if count else self.GENESIS
        self._cached_count = count
        self._cached_valid = not problems
        self._cached_problems = problems
        try:
            self._cached_signature = self._signature()
        except OSError:
            self._cached_signature = None
        return {"valid": self._cached_valid, "event_count": count, "problems": list(problems),
                "full_scan_performed": True,
                "note": "Hash-chaining detects modification/deletion when the trusted head and audit file are protected; this is not a digital signature."}

    def verify(self) -> dict[str, Any]:
        """Perform a full end-to-end chain verification, regardless of cache."""
        with self._lock:
            return self._scan()

    def verify_cached(self) -> dict[str, Any]:
        """Quick status; a changed file signature triggers a full verification scan."""
        with self._lock:
            try:
                current = self._signature()
            except OSError:
                return self._scan()
            if self._cached_signature != current:
                return self._scan()
            return {"valid": self._cached_valid, "event_count": self._cached_count,
                    "problems": list(self._cached_problems), "full_scan_performed": False,
                    "note": "Cached status; explicit verify() always scans the complete chain. Size/mtime changes trigger a scan."}

    def append(self, event: str, *, doc_id: str = "", content_hash: str = "", details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            report = self.verify_cached()
            if not report["valid"]:
                raise ValueError("Audit chain is invalid; refusing to append until integrity is investigated.")
            body = {"sequence": self._cached_count + 1, "event": str(event), "doc_id": str(doc_id),
                    "content_hash": str(content_hash), "previous_hash": self._cached_head,
                    "details": dict(details or {}), "timestamp_unix": round(time.time(), 3)}
            record = dict(body, event_hash=self._event_hash(body))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
            self._cached_head = record["event_hash"]
            self._cached_count += 1
            self._cached_valid = True
            self._cached_problems = []
            self._cached_signature = self._signature()
            return record
