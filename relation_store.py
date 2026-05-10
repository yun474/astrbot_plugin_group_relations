from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


VECTOR_SIZE = 256


@dataclass
class RelationRecord:
    id: str
    group_id: str
    subject: str
    relation: str
    object: str
    note: str = ""
    source: str = "manual"
    confidence: float = 1.0
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    embedding_provider_id: str = ""
    embedding_dim: int = 0

    def text(self) -> str:
        parts = [self.subject, self.relation, self.object, self.note]
        return " ".join(part for part in parts if part).strip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def relation_id(group_id: str, subject: str, relation: str, object_: str) -> str:
    raw = f"{group_id}|{normalize_text(subject)}|{normalize_text(relation)}|{normalize_text(object_)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def clamp_confidence(value: Any, default: float = 1.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _tokens(text: str) -> list[str]:
    text = normalize_text(text)
    words = re.findall(r"[\w\u4e00-\u9fff]+", text)
    grams: list[str] = []
    for word in words:
        grams.append(word)
        if len(word) > 1:
            grams.extend(word[i : i + 2] for i in range(len(word) - 1))
        if len(word) > 2:
            grams.extend(word[i : i + 3] for i in range(len(word) - 2))
    return grams


def embed_text(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "big") % VECTOR_SIZE
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class RelationStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_file = data_dir / "relations.json"
        self.records: dict[str, RelationRecord] = {}
        self.vectors: dict[str, list[float]] = {}

    def load(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self.records = {}
            self.vectors = {}
            return
        try:
            payload = json.loads(self.data_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.records = {}
            self.vectors = {}
            return
        records = payload.get("relations", [])
        self.records = {}
        for item in records:
            record = self._coerce_record(item)
            if record:
                self.records[record.id] = record
        self.vectors = {
            record_id: vector
            for record_id, vector in payload.get("vectors", {}).items()
            if record_id in self.records
            and isinstance(vector, list)
            and all(isinstance(value, int | float) for value in vector)
        }
        for record_id, record in self.records.items():
            if record_id not in self.vectors:
                self.vectors[record_id] = embed_text(record.text())

    def _coerce_record(self, item: Any) -> RelationRecord | None:
        if not isinstance(item, dict):
            return None
        required = ["id", "group_id", "subject", "relation", "object"]
        if any(not item.get(key) for key in required):
            return None
        allowed = {field_.name for field_ in fields(RelationRecord)}
        payload = {key: value for key, value in item.items() if key in allowed}
        try:
            payload["confidence"] = clamp_confidence(payload.get("confidence", 1.0))
            payload["created_at"] = int(payload.get("created_at") or time.time())
            payload["updated_at"] = int(payload.get("updated_at") or payload["created_at"])
            payload["embedding_dim"] = int(payload.get("embedding_dim") or 0)
            return RelationRecord(**payload)
        except (TypeError, ValueError):
            return None

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "relations": [asdict(record) for record in self.records.values()],
            "vectors": self.vectors,
        }
        tmp_file = self.data_file.with_suffix(".json.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_file, self.data_file)

    def upsert(
        self,
        group_id: str,
        subject: str,
        relation: str,
        object_: str,
        note: str = "",
        source: str = "manual",
        confidence: float = 1.0,
        vector: list[float] | None = None,
        embedding_provider_id: str = "",
    ) -> RelationRecord:
        rid = relation_id(group_id, subject, relation, object_)
        now = int(time.time())
        existing = self.records.get(rid)
        if existing:
            existing.note = note or existing.note
            existing.source = source or existing.source
            existing.confidence = max(existing.confidence, clamp_confidence(confidence))
            existing.updated_at = now
            if embedding_provider_id:
                existing.embedding_provider_id = embedding_provider_id
            if vector is not None:
                existing.embedding_dim = len(vector)
            record = existing
        else:
            record = RelationRecord(
                id=rid,
                group_id=group_id,
                subject=subject.strip(),
                relation=relation.strip(),
                object=object_.strip(),
                note=note.strip(),
                source=source,
                confidence=clamp_confidence(confidence),
                embedding_provider_id=embedding_provider_id,
                embedding_dim=len(vector or []),
            )
            self.records[rid] = record
        self.vectors[rid] = (
            normalize_vector(vector) if vector is not None else embed_text(record.text())
        )
        self.save()
        return record

    def delete(self, relation_id_: str, group_id: str | None = None) -> bool:
        record = self.records.get(relation_id_)
        if not record or (group_id and record.group_id != group_id):
            return False
        del self.records[relation_id_]
        self.vectors.pop(relation_id_, None)
        self.save()
        return True

    def update(
        self,
        relation_id_: str,
        group_id: str,
        subject: str | None = None,
        relation: str | None = None,
        object_: str | None = None,
        note: str | None = None,
        confidence: float | None = None,
        vector: list[float] | None = None,
        embedding_provider_id: str = "",
    ) -> RelationRecord | None:
        record = self.records.get(relation_id_)
        if not record or record.group_id != group_id:
            return None
        if subject:
            record.subject = subject.strip()
        if relation:
            record.relation = relation.strip()
        if object_:
            record.object = object_.strip()
        if note is not None:
            record.note = note.strip()
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
        if embedding_provider_id:
            record.embedding_provider_id = embedding_provider_id
        if vector is not None:
            record.embedding_dim = len(vector)
            self.vectors[relation_id_] = normalize_vector(vector)
        else:
            self.vectors[relation_id_] = embed_text(record.text())
        record.updated_at = int(time.time())
        new_id = relation_id(record.group_id, record.subject, record.relation, record.object)
        if new_id != relation_id_:
            self.records.pop(relation_id_, None)
            self.vectors.pop(relation_id_, None)
            record.id = new_id
            if new_id in self.records:
                existing = self.records[new_id]
                existing.note = record.note or existing.note
                existing.source = record.source or existing.source
                existing.confidence = max(existing.confidence, record.confidence)
                existing.updated_at = record.updated_at
                existing.embedding_provider_id = record.embedding_provider_id
                existing.embedding_dim = record.embedding_dim
                record = existing
            self.records[new_id] = record
            self.vectors[new_id] = normalize_vector(vector) if vector is not None else embed_text(record.text())
        self.save()
        return record

    def by_person(self, group_id: str, person: str, limit: int = 8) -> list[RelationRecord]:
        needle = normalize_text(person)
        matches = [
            record
            for record in self.records.values()
            if record.group_id == group_id
            and (needle in normalize_text(record.subject) or needle in normalize_text(record.object))
        ]
        return sorted(matches, key=lambda item: item.updated_at, reverse=True)[:limit]

    def recent(self, group_id: str, limit: int = 8) -> list[RelationRecord]:
        records = [record for record in self.records.values() if record.group_id == group_id]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)[:limit]

    def search(
        self,
        group_id: str,
        query: str,
        limit: int = 8,
        threshold: float = 0.12,
    ) -> list[tuple[RelationRecord, float]]:
        query_vector = embed_text(query)
        return self.search_by_vector(group_id, query, query_vector, limit=limit, threshold=threshold)

    def search_by_text(
        self,
        group_id: str,
        query: str,
        limit: int = 8,
    ) -> list[tuple[RelationRecord, float]]:
        query_norm = normalize_text(query)
        if not query_norm:
            return [(record, 0.0) for record in self.recent(group_id, limit=limit)]
        scored: list[tuple[RelationRecord, float]] = []
        query_tokens = set(_tokens(query_norm))
        for record in self.records.values():
            if record.group_id != group_id:
                continue
            record_norm = normalize_text(record.text())
            if query_norm in record_norm:
                scored.append((record, 0.25))
                continue
            record_tokens = set(_tokens(record_norm))
            if not query_tokens or not record_tokens:
                continue
            overlap = len(query_tokens & record_tokens) / max(len(query_tokens), 1)
            if overlap >= 0.15:
                scored.append((record, overlap))
        scored.sort(key=lambda item: (item[1], item[0].updated_at), reverse=True)
        return scored[:limit]

    def search_by_vector(
        self,
        group_id: str,
        query: str,
        query_vector: list[float],
        limit: int = 8,
        threshold: float = 0.12,
    ) -> list[tuple[RelationRecord, float]]:
        query_norm_empty = not any(query_vector)
        scored: list[tuple[RelationRecord, float]] = []
        for record_id, record in self.records.items():
            if record.group_id != group_id:
                continue
            lexical = 0.25 if normalize_text(query) in normalize_text(record.text()) else 0.0
            semantic = 0.0 if query_norm_empty else cosine(query_vector, self.vectors.get(record_id, []))
            score = max(lexical, semantic)
            if score >= threshold:
                scored.append((record, score))
        scored.sort(key=lambda item: (item[1], item[0].updated_at), reverse=True)
        return scored[:limit]

    def export_group(self, group_id: str) -> list[dict[str, Any]]:
        return [
            asdict(record)
            for record in sorted(self.records.values(), key=lambda item: item.updated_at, reverse=True)
            if record.group_id == group_id
        ]

    def _rebuild_vectors(self) -> None:
        self.vectors = {record_id: embed_text(record.text()) for record_id, record in self.records.items()}


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def format_record(record: RelationRecord, with_id: bool = True) -> str:
    prefix = f"[{record.id}] " if with_id else ""
    note = f"（{record.note}）" if record.note else ""
    return f"{prefix}{record.subject} --{record.relation}--> {record.object}{note}"
