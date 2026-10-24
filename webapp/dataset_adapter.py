"""Safe adapters for uploaded documents and Kaggle-style tabular exports.

CSV/JSON rows become individually source-linked text records. ZIPs are never extracted
onto disk; only allow-listed extensions and bounded members are accepted.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import PurePosixPath
from typing import Any, Iterable

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_UNPACKED = 200 * 1024 * 1024
MAX_ZIP_MEMBERS = 1000
MAX_ROWS = 5000
MAX_ROW_CHARS = 12000
ALLOWED_DOCS = {".txt", ".md", ".pdf", ".csv", ".json", ".jsonl", ".ndjson", ".zip"}
PREFERRED_FIELDS = (
    "title", "abstract", "summary", "text", "content", "body", "description",
    "question", "answer", "paragraph", "document", "article", "message", "description_text",
)


@dataclass
class PreparedRecord:
    filename: str
    content: bytes
    source_title: str
    source_url: str = ""
    source_kind: str = "uploaded_document"
    metadata: dict[str, Any] | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "source_title": self.source_title or self.filename,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            **(self.metadata or {}),
        }


def _safe_name(name: str) -> str:
    # Drop path components, control chars, and platform path oddities.
    normalized = str(name).replace("\\", "/")
    name = PurePosixPath(normalized).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"[^A-Za-z0-9._() -]+", "_", name).strip(" .")
    if not name or name in {".", ".."}:
        name = "uploaded_document.txt"
    return name[:180]


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(_scalar(x) for x in value[:30] if _scalar(x))
    if isinstance(value, dict):
        # Nested JSON fields stay compact and deterministic.
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:2000]
    return str(value)[:2000]


def _record_text(row: dict[str, Any], *, max_chars: int = MAX_ROW_CHARS) -> str:
    lowered = {str(k).casefold(): k for k in row}
    chosen: list[Any] = []
    for field in PREFERRED_FIELDS:
        if field in lowered and lowered[field] not in chosen:
            chosen.append(lowered[field])
    # Include other useful scalar columns after likely content fields; never encode a
    # record based only on its ID if it has a title/abstract/body.
    for key, value in row.items():
        if key not in chosen and _scalar(value) and str(key).casefold() not in {"id", "index", "row_id"}:
            chosen.append(key)
    chunks = []
    seen = set()
    for key in chosen:
        value = _scalar(row.get(key))
        if not value:
            continue
        sig = value.casefold()
        if sig in seen:
            continue
        seen.add(sig)
        chunks.append(f"{key}: {value}")
    if not chunks:
        chunks = [f"{k}: {_scalar(v)}" for k, v in row.items() if _scalar(v)]
    return "\n".join(chunks)[:max_chars]


def _row_title(row: dict[str, Any], fallback: str) -> str:
    for candidate in ("title", "name", "question", "id", "arxiv_id", "paper_id"):
        for k, v in row.items():
            if str(k).casefold() == candidate and _scalar(v):
                return _scalar(v)[:180]
    return fallback


def _rows_from_bytes(name: str, data: bytes) -> list[dict[str, Any]]:
    ext = PurePosixPath(name).suffix.casefold()
    text = _decode(data)
    if ext in {".jsonl", ".ndjson"}:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"text": _scalar(item)})
        return rows
    if ext == ".json":
        data_obj = json.loads(text)
        if isinstance(data_obj, list):
            return [r if isinstance(r, dict) else {"text": _scalar(r)} for r in data_obj]
        if isinstance(data_obj, dict):
            # Common Kaggle wrapper layouts, e.g. {"data": [...]}.
            for key in ("data", "records", "items", "rows", "papers", "articles"):
                if isinstance(data_obj.get(key), list):
                    return [r if isinstance(r, dict) else {"text": _scalar(r)} for r in data_obj[key]]
            return [data_obj]
        return [{"text": _scalar(data_obj)}]
    if ext == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader if any(_scalar(v) for v in row.values())]
    raise ValueError(f"Unsupported structured file: {name}")


def _unpack_zip(name: str, data: bytes) -> list[tuple[str, bytes]]:
    unpacked = []
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) > MAX_ZIP_MEMBERS:
            raise ValueError(f"ZIP has {len(infos)} files; maximum is {MAX_ZIP_MEMBERS}.")
        for info in infos:
            member_name = _safe_name(info.filename)
            ext = PurePosixPath(member_name).suffix.casefold()
            if ext not in ALLOWED_DOCS - {".zip"}:
                continue
            # Reject extreme compression ratios / suspiciously huge declared members.
            if info.file_size > MAX_FILE_BYTES or info.compress_size == 0 and info.file_size > 0:
                raise ValueError(f"ZIP member exceeds per-file limit: {member_name}")
            if info.compress_size and info.file_size / max(info.compress_size, 1) > 250:
                raise ValueError(f"ZIP member has a suspicious compression ratio: {member_name}")
            if ext == ".zip":
                continue
            content = zf.read(info)
            total += len(content)
            if total > MAX_TOTAL_UNPACKED:
                raise ValueError("ZIP exceeds the 50 MB unpacked-data limit.")
            unpacked.append((member_name, content))
    return unpacked


def prepare_uploads(
    files: Iterable[tuple[str, bytes]],
    *,
    dataset_name: str = "Uploaded dataset",
    source_url: str = "",
    max_records: int = MAX_ROWS,
) -> tuple[list[PreparedRecord], dict[str, Any]]:
    """Convert files to safe pipeline records and return an import manifest."""
    raw_files: list[tuple[str, bytes]] = []
    total = 0
    for name, data in files:
        name = _safe_name(name)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{name} exceeds the 20 MB per-file limit.")
        total += len(data)
        if total > MAX_TOTAL_UNPACKED:
            raise ValueError("Total upload exceeds the 50 MB limit.")
        ext = PurePosixPath(name).suffix.casefold()
        if ext not in ALLOWED_DOCS:
            raise ValueError(f"Unsupported file type: {name}. Allowed: PDF, TXT, MD, CSV, JSON, JSONL, ZIP.")
        if ext == ".zip":
            try:
                raw_files.extend(_unpack_zip(name, data))
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                raise ValueError(f"Could not read ZIP {name}: {type(exc).__name__}") from exc
        else:
            raw_files.append((name, data))

    prepared: list[PreparedRecord] = []
    file_manifest = []
    row_count = 0
    for name, data in raw_files:
        ext = PurePosixPath(name).suffix.casefold()
        if ext in {".csv", ".json", ".jsonl", ".ndjson"}:
            try:
                rows = _rows_from_bytes(name, data)
            except Exception as exc:
                raise ValueError(f"Could not parse {name}: {type(exc).__name__}: {exc}") from exc
            remain = max(0, max_records - row_count)
            rows = rows[:remain]
            row_count += len(rows)
            prefix = PurePosixPath(name).stem[:70]
            created = 0
            for i, row in enumerate(rows, 1):
                text = _record_text(row)
                if not text.strip():
                    continue
                title = _row_title(row, f"{prefix} record {i:05d}")
                outname = _safe_name(f"{prefix}_row_{i:05d}.txt")
                meta = {
                    "dataset_name": dataset_name,
                    "dataset_source_file": name,
                    "dataset_row": i,
                    "record_id": _scalar(row.get("id", row.get("arxiv_id", row.get("paper_id", i)))),
                    "language": "undetermined",
                }
                prepared.append(PreparedRecord(outname, text.encode("utf-8"), title, source_url,
                                               "tabular_dataset_row", meta))
                created += 1
            file_manifest.append({"file": name, "kind": "tabular", "rows_seen": len(rows), "records_created": created})
        else:
            title = name
            prepared.append(PreparedRecord(name, data, title, source_url, "uploaded_document",
                                           {"dataset_name": dataset_name, "dataset_source_file": name}))
            file_manifest.append({"file": name, "kind": "document", "bytes": len(data), "records_created": 1})
    if not prepared:
        raise ValueError("No usable documents were found in the upload.")
    if len(prepared) > max_records + 100:
        raise ValueError("Upload created too many records. Split the dataset into smaller uploads.")
    manifest = {
        "dataset_name": dataset_name,
        "source_url": source_url,
        "input_file_count": len(raw_files),
        "prepared_record_count": len(prepared),
        "tabular_rows_seen": row_count,
        "file_manifest": file_manifest,
        "truncated_to_max_records": row_count >= max_records,
        "limits": {"per_file_mb": MAX_FILE_BYTES // (1024 * 1024), "total_mb": MAX_TOTAL_UNPACKED // (1024 * 1024),
                   "max_tabular_records_per_upload": max_records},
    }
    return prepared, manifest
