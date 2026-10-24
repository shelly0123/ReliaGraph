"""Integration adapter: generic extraction + incremental KB + CIA audit events.

This is the service Member 3 can call. It uses local/deterministic fallbacks by default;
ML extraction is opt-in so the core app never depends on model downloads to start.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from collections import defaultdict
import hashlib
import json
import re

from models import Chunk
from core.robust_extraction import extract_with_fallback
from core.cia_controls import DataHandlingPolicy, HashChainAuditLog, sha256_text
from core.selective_quality import canonical_triple, assess_candidate_evidence
from core.phase2.incremental_kb import IncrementalKnowledgeBase


def _canonical(value: str) -> str:
    return re.sub(r"[^\w\s-]", "", re.sub(r"\s+", " ", str(value or "").casefold())).strip()


class TopicWeaveRobustService:
    """Document CRUD service that keeps extraction and storage under one interface.

    `upsert_document` measures the whole in-process path: sanitation, chunking,
    configured extraction, provenance validation, SQLite writes and graph mutation.
    It does not include initial model download; model load/inference is reported separately.
    """
    def __init__(self, db_path: str | Path = "data/topicweave_kb.sqlite3", *,
                 audit_log_path: str | Path = "data/audit/cia_events.jsonl",
                 enable_gliner: bool = False, enable_spacy: bool = False,
                 retain_history_text: bool = False,
                 policy: DataHandlingPolicy | None = None,
                 calibration_model: Mapping[str, Any] | None = None):
        self.policy = policy or DataHandlingPolicy()
        self.enable_gliner = bool(enable_gliner)
        self.enable_spacy = bool(enable_spacy)
        self.calibration_model = dict(calibration_model) if calibration_model is not None else None
        self.last_extraction_diagnostics: dict[str, Any] = {}
        self.audit_log = HashChainAuditLog(audit_log_path)
        self.kb = IncrementalKnowledgeBase(db_path, extractor=self._extract_claims,
                                           retain_history_text=retain_history_text)

    def close(self) -> None:
        self.kb.close()

    def _extract_claims(self, safe_text: str, chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        chunk_objects = [Chunk(chunk_id=str(c["chunk_id"]), doc_id=str(c["doc_id"]), text=str(c.get("text", "")),
                               page=c.get("page"), source_title=str(c.get("source_title", "") or "") or None,
                               source_url=str(c.get("source_url", "") or "") or None,
                               metadata=dict(c.get("metadata") or {})) for c in chunks]
        result = extract_with_fallback(chunk_objects, enable_gliner=self.enable_gliner,
                                       enable_spacy=self.enable_spacy, require_independent_support=True,
                                       calibration_model=self.calibration_model)
        # Keep quality-gate decisions attached to stored graph edges. A candidate is
        # never promoted to a verified fact by a model score alone.
        decision_by_triple = {
            canonical_triple(str(d.get("subject", "")), str(d.get("relation_type", "")), str(d.get("object", ""))): d
            for d in result.candidate_decisions
        }
        self.last_extraction_diagnostics = dict(result.diagnostics)
        self.last_extraction_diagnostics["provenance_rejected_before_storage"] = 0
        # IncrementalKnowledgeBase uses a unique claim ID for each triple/chunk.
        # Merge model signals before storage so different extractors do not collide.
        # Validate each source instance separately: one valid quote for a triple must
        # not legitimize another candidate whose quote/endpoint is not source-anchored.
        chunk_by_id = {c.chunk_id: c for c in chunk_objects}
        grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for rel in result.relations:
            key = (*canonical_triple(rel.subject, rel.relation_type, rel.object), rel.chunk_id)
            triple_decision = decision_by_triple.get(canonical_triple(rel.subject, rel.relation_type, rel.object), {})
            decision_name = str(triple_decision.get("decision", "review_required"))
            evidence_ok, _evidence_reasons = assess_candidate_evidence(rel, chunk_by_id)
            if (decision_name == "rejected_provenance" or triple_decision.get("evidence_valid") is False
                    or not evidence_ok):
                self.last_extraction_diagnostics["provenance_rejected_before_storage"] += 1
                continue
            if key not in grouped:
                grouped[key] = {"subject": rel.subject, "relation_type": rel.relation_type,
                                "object": rel.object, "evidence_text": rel.evidence_text,
                                "chunk_id": rel.chunk_id, "confidence": rel.confidence,
                                "status": decision_name, "metadata": {
                                    "extractors": [], "review_required": decision_name != "human_verified",
                                    "quality_decision": decision_name,
                                    "quality_explanation": str(triple_decision.get("explanation", "No promotion policy result; retain as review candidate.")),
                                    "evidence_valid": bool(triple_decision.get("evidence_valid", False)),
                                    "evidence_quality_score": triple_decision.get("evidence_quality_score"),
                                    "triage_model_score": triple_decision.get("triage_model_score"),
                                    "triage_score_kind": triple_decision.get("triage_score_kind", "not_enabled"),
                                    "independent_extractor_count": int(triple_decision.get("independent_extractor_count", 0)),
                                    "independent_source_count": int(triple_decision.get("independent_source_count", 0)),
                                    "score_kind": "uncalibrated_candidate_score",
                                    "human_verification_required": decision_name != "human_verified"}}
            row = grouped[key]
            attrs = dict(rel.attributes or {})
            name = str(attrs.get("extractor", "unknown"))
            if name not in row["metadata"]["extractors"]:
                row["metadata"]["extractors"].append(name)
            row["metadata"].setdefault("candidate_attributes", []).append(attrs)
            if rel.confidence is not None:
                old = row.get("confidence")
                row["confidence"] = max(float(old or 0.0), float(rel.confidence))
        self.last_extraction_diagnostics["stored_candidate_claims"] = len(grouped)
        self.last_extraction_diagnostics["candidate_claims_require_human_review"] = sum(
            1 for row in grouped.values() if row.get("metadata", {}).get("human_verification_required", True)
        )
        return list(grouped.values())

    def upsert_document(self, doc_id: str, text: str, *, title: str = "", source_url: str = "",
                        topic_assignments: Mapping[str, Sequence[Mapping[str, str] | str]] | None = None,
                        chunks: Sequence[Mapping[str, Any]] | None = None,
                        claims: Sequence[Mapping[str, Any]] | None = None,
                        force_refresh: bool = False) -> dict[str, Any]:
        """Incrementally update one document.

        `chunks` preserves Member 1's content-addressed chunk IDs. When `claims` is
        supplied, these are already extracted/retained claim instances and no model
        extraction is run; this is used for topic-only updates and unchanged chunks.
        `force_refresh` lets metadata/topic updates reach storage even if text hashes match.
        """
        self.policy.validate_provider("local")
        content_hash = sha256_text(text)
        if claims is not None:
            self.last_extraction_diagnostics = {"mode": "reused_claims_no_model_inference",
                                                "stored_claims_supplied": len(claims)}
        result = self.kb.upsert_document(doc_id, text, title=title, source_url=source_url,
                                         topic_assignments=topic_assignments, chunks=chunks,
                                         claims=claims, force_refresh=force_refresh)
        event = self.audit_log.append(result["operation"], doc_id=doc_id, content_hash=content_hash,
                                      details={"version": result.get("version"), "elapsed_seconds": result.get("elapsed_seconds"),
                                               "accepted_claims": result.get("accepted_claims", 0),
                                               "rejected_claims": result.get("rejected_claims", 0),
                                               "injection_findings": result.get("injection_findings", 0)})
        result["extraction_diagnostics"] = dict(self.last_extraction_diagnostics)
        result["audit_event_hash"] = event["event_hash"]
        result["cia_integrity"] = self.audit_log.verify_cached()
        return result

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        row = self.kb.conn.execute("SELECT content_hash FROM documents WHERE doc_id=?", (str(doc_id),)).fetchone()
        previous_hash = str(row["content_hash"]) if row else ""
        result = self.kb.delete_document(doc_id)
        event = self.audit_log.append("delete", doc_id=str(doc_id), content_hash=previous_hash,
                                      details={"operation": result.get("operation"),
                                               "elapsed_seconds": result.get("elapsed_seconds")})
        result["audit_event_hash"] = event["event_hash"]
        result["cia_integrity"] = self.audit_log.verify_cached()
        return result

    def query_claims(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        """Get all claims or perform an indexed query scoped to one document."""
        return self.kb.get_claims(doc_id=doc_id)

    def competing_claims(self) -> list[dict[str, Any]]:
        return self.kb.find_competing_claims()

    def metrics(self) -> dict[str, Any]:
        metrics = dict(self.kb.metrics())
        metrics["audit_log"] = self.audit_log.verify()
        metrics["model_policy"] = {"gliner_enabled": self.enable_gliner, "spacy_enabled": self.enable_spacy,
                                    "local_only": self.policy.local_only,
                                    "remote_model_inference_allowed": self.policy.allow_remote_model_inference}
        return metrics
