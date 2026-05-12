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


@dataclass
class GroupMemorySpace:
    id: str
    name: str = ""
    session_id: str = ""
    kind: str = "group"
    owner_user_id: str = ""
    owner_display_name: str = ""
    owner_evidence: str = ""
    owner_updated_at: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    message_count: int = 0


@dataclass
class UserProfile:
    id: str
    group_id: str
    user_id: str
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    group_role: str = "unknown"
    role_evidence: str = ""
    role_updated_at: int = 0
    facts: list[dict[str, Any]] = field(default_factory=list)
    message_count: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def text(self) -> str:
        fact_text = " ".join(str(item.get("fact") or "") for item in self.facts)
        parts = [self.display_name, " ".join(self.aliases), self.group_role, fact_text]
        return " ".join(part for part in parts if part).strip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def role_rank(role: str) -> int:
    return {"unknown": 0, "member": 1, "admin": 2, "owner": 3}.get(normalize_text(role), 0)


def profile_id(group_id: str, user_id: str) -> str:
    raw = f"{group_id}|{normalize_text(user_id)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


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
        self.groups: dict[str, GroupMemorySpace] = {}
        self.profiles: dict[str, UserProfile] = {}

    def load(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self.records = {}
            self.vectors = {}
            self.groups = {}
            self.profiles = {}
            return
        try:
            payload = json.loads(self.data_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.records = {}
            self.vectors = {}
            self.groups = {}
            self.profiles = {}
            return
        records = payload.get("relations", [])
        self.records = {}
        for item in records:
            record = self._coerce_record(item)
            if record:
                self.records[record.id] = record
        self.groups = {}
        for item in payload.get("groups", []):
            group = self._coerce_group(item)
            if group:
                self.groups[group.id] = group
        self.profiles = {}
        for item in payload.get("profiles", []):
            profile = self._coerce_profile(item)
            if profile:
                self.profiles[profile.id] = profile
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
        self._ensure_groups()

    def _coerce_group(self, item: Any) -> GroupMemorySpace | None:
        if not isinstance(item, dict) or not item.get("id"):
            return None
        allowed = {field_.name for field_ in fields(GroupMemorySpace)}
        payload = {key: value for key, value in item.items() if key in allowed}
        try:
            payload["created_at"] = int(payload.get("created_at") or time.time())
            payload["updated_at"] = int(payload.get("updated_at") or payload["created_at"])
            payload["message_count"] = int(payload.get("message_count") or 0)
            payload["owner_updated_at"] = int(payload.get("owner_updated_at") or 0)
            return GroupMemorySpace(**payload)
        except (TypeError, ValueError):
            return None

    def _coerce_profile(self, item: Any) -> UserProfile | None:
        if not isinstance(item, dict):
            return None
        required = ["id", "group_id", "user_id"]
        if any(not item.get(key) for key in required):
            return None
        allowed = {field_.name for field_ in fields(UserProfile)}
        payload = {key: value for key, value in item.items() if key in allowed}
        try:
            payload["aliases"] = [
                str(alias).strip()
                for alias in payload.get("aliases", [])
                if str(alias).strip()
            ][:12]
            payload["facts"] = [
                fact
                for fact in payload.get("facts", [])
                if isinstance(fact, dict) and str(fact.get("fact") or "").strip()
            ][:50]
            payload["message_count"] = int(payload.get("message_count") or 0)
            payload["created_at"] = int(payload.get("created_at") or time.time())
            payload["updated_at"] = int(payload.get("updated_at") or payload["created_at"])
            payload["role_updated_at"] = int(payload.get("role_updated_at") or 0)
            return UserProfile(**payload)
        except (TypeError, ValueError):
            return None

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
            "groups": [asdict(group) for group in self.groups.values()],
            "profiles": [asdict(profile) for profile in self.profiles.values()],
            "relations": [asdict(record) for record in self.records.values()],
            "vectors": self.vectors,
        }
        tmp_file = self.data_file.with_suffix(".json.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_file, self.data_file)

    def _ensure_groups(self) -> None:
        now = int(time.time())
        group_ids = {record.group_id for record in self.records.values()}
        group_ids.update(profile.group_id for profile in self.profiles.values())
        for group_id in group_ids:
            if group_id and group_id not in self.groups:
                self.groups[group_id] = GroupMemorySpace(
                    id=group_id,
                    name=group_id,
                    session_id=group_id,
                    kind="group",
                    created_at=now,
                    updated_at=now,
                )

    def touch_group(
        self,
        group_id: str,
        name: str = "",
        session_id: str = "",
        kind: str = "group",
    ) -> GroupMemorySpace:
        now = int(time.time())
        group = self.groups.get(group_id)
        if not group:
            group = GroupMemorySpace(
                id=group_id,
                name=name.strip(),
                session_id=session_id.strip(),
                kind=kind.strip() or "group",
                created_at=now,
                updated_at=now,
            )
            self.groups[group_id] = group
        else:
            if name.strip():
                group.name = name.strip()
            if session_id.strip():
                group.session_id = session_id.strip()
            if kind.strip():
                group.kind = kind.strip()
            group.updated_at = now
        group.message_count += 1
        self.save()
        return group

    def update_group(
        self,
        group_id: str,
        name: str | None = None,
        kind: str | None = None,
        owner_user_id: str | None = None,
        owner_display_name: str | None = None,
        owner_evidence: str | None = None,
    ) -> GroupMemorySpace | None:
        group = self.groups.get(group_id)
        if not group:
            return None
        if name is not None:
            group.name = name.strip()
        if kind is not None and kind.strip():
            group.kind = kind.strip()
        if owner_user_id is not None:
            group.owner_user_id = owner_user_id.strip()
            group.owner_display_name = (owner_display_name or "").strip()
            group.owner_evidence = (owner_evidence or "webui").strip()
            group.owner_updated_at = int(time.time()) if group.owner_user_id else 0
        group.updated_at = int(time.time())
        self.save()
        return group

    def set_group_owner(
        self,
        group_id: str,
        user_id: str,
        display_name: str = "",
        evidence: str = "",
    ) -> GroupMemorySpace | None:
        group = self.groups.get(group_id)
        if not group:
            return None
        group.owner_user_id = user_id.strip()
        group.owner_display_name = display_name.strip()
        group.owner_evidence = evidence.strip()
        group.owner_updated_at = int(time.time()) if group.owner_user_id else 0
        group.updated_at = int(time.time())
        self.save()
        return group

    def touch_profile(
        self,
        group_id: str,
        user_id: str,
        display_name: str = "",
        group_role: str = "",
        role_evidence: str = "",
    ) -> UserProfile:
        now = int(time.time())
        pid = profile_id(group_id, user_id)
        profile = self.profiles.get(pid)
        display_name = display_name.strip()
        group_role = group_role.strip().lower()
        role_evidence = role_evidence.strip()
        if not profile:
            profile = UserProfile(
                id=pid,
                group_id=group_id,
                user_id=str(user_id).strip(),
                display_name=display_name,
                aliases=[display_name] if display_name else [],
                group_role=group_role or "unknown",
                role_evidence=role_evidence,
                role_updated_at=now if group_role else 0,
                created_at=now,
                updated_at=now,
            )
            self.profiles[pid] = profile
        else:
            if display_name:
                profile.display_name = display_name
                if display_name not in profile.aliases:
                    profile.aliases.insert(0, display_name)
                    profile.aliases = profile.aliases[:12]
            if group_role and role_rank(group_role) > role_rank(profile.group_role):
                profile.group_role = group_role
                profile.role_evidence = role_evidence
                profile.role_updated_at = now
            profile.updated_at = now
        profile.message_count += 1
        self.save()
        return profile

    def remember_profile_fact(
        self,
        group_id: str,
        user_id: str,
        display_name: str,
        fact: str,
        note: str = "",
        source: str = "manual",
        confidence: float = 0.8,
    ) -> UserProfile:
        profile = self.touch_profile(group_id, user_id, display_name)
        fact = re.sub(r"\s+", " ", fact).strip()
        if not fact:
            return profile
        now = int(time.time())
        norm = normalize_text(fact)
        existing = next(
            (
                item
                for item in profile.facts
                if normalize_text(str(item.get("fact") or "")) == norm
            ),
            None,
        )
        if existing:
            existing["note"] = note.strip() or existing.get("note", "")
            existing["source"] = source or existing.get("source", "manual")
            existing["confidence"] = max(
                clamp_confidence(existing.get("confidence", 0.0)),
                clamp_confidence(confidence, 0.8),
            )
            existing["updated_at"] = now
        else:
            profile.facts.insert(
                0,
                {
                    "fact": fact[:160],
                    "note": note.strip()[:240],
                    "source": source,
                    "confidence": clamp_confidence(confidence, 0.8),
                    "created_at": now,
                    "updated_at": now,
                },
            )
            profile.facts = profile.facts[:50]
        profile.updated_at = now
        self.save()
        return profile

    def update_profile(
        self,
        profile_id_: str,
        group_id: str,
        display_name: str | None = None,
        aliases: list[str] | None = None,
        group_role: str | None = None,
        role_evidence: str | None = None,
    ) -> UserProfile | None:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return None
        if display_name is not None:
            profile.display_name = display_name.strip()
        if aliases is not None:
            clean_aliases = []
            for alias in aliases:
                alias = str(alias).strip()
                if alias and alias not in clean_aliases:
                    clean_aliases.append(alias)
            if profile.display_name and profile.display_name not in clean_aliases:
                clean_aliases.insert(0, profile.display_name)
            profile.aliases = clean_aliases[:12]
        if group_role is not None and group_role.strip():
            profile.group_role = group_role.strip().lower()
            profile.role_evidence = (role_evidence or "webui").strip()
            profile.role_updated_at = int(time.time())
        profile.updated_at = int(time.time())
        self.save()
        return profile

    def update_profile_fact(
        self,
        profile_id_: str,
        group_id: str,
        index: int,
        fact: str | None = None,
        note: str | None = None,
        confidence: float | None = None,
    ) -> UserProfile | None:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return None
        if index < 0 or index >= len(profile.facts):
            return None
        item = profile.facts[index]
        if fact is not None:
            fact = re.sub(r"\s+", " ", fact).strip()
            if not fact:
                return None
            item["fact"] = fact[:160]
        if note is not None:
            item["note"] = note.strip()[:240]
        if confidence is not None:
            item["confidence"] = clamp_confidence(confidence, 0.8)
        item["updated_at"] = int(time.time())
        profile.updated_at = int(time.time())
        self.save()
        return profile

    def delete_profile(self, profile_id_: str, group_id: str) -> bool:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return False
        del self.profiles[profile_id_]
        self.save()
        return True

    def delete_profile_facts(
        self,
        profile_id_: str,
        group_id: str,
        fact_query: str,
    ) -> int:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return 0
        query_norm = normalize_text(fact_query)
        if not query_norm:
            return 0
        old_count = len(profile.facts)
        profile.facts = [
            item
            for item in profile.facts
            if query_norm not in normalize_text(str(item.get("fact") or ""))
            and query_norm not in normalize_text(str(item.get("note") or ""))
        ]
        deleted = old_count - len(profile.facts)
        if deleted:
            profile.updated_at = int(time.time())
            self.save()
        return deleted

    def delete_profile_fact_index(
        self,
        profile_id_: str,
        group_id: str,
        index: int,
    ) -> bool:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return False
        if index < 0 or index >= len(profile.facts):
            return False
        profile.facts.pop(index)
        profile.updated_at = int(time.time())
        self.save()
        return True

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

    def get_profile(self, group_id: str, user_id: str) -> UserProfile | None:
        return self.profiles.get(profile_id(group_id, user_id))

    def find_profiles(
        self,
        group_id: str,
        query: str = "",
        limit: int = 8,
    ) -> list[UserProfile]:
        query_norm = normalize_text(query)
        matches = [
            profile
            for profile in self.profiles.values()
            if profile.group_id == group_id
            and (
                not query_norm
                or query_norm in normalize_text(profile.user_id)
                or query_norm in normalize_text(profile.display_name)
                or any(query_norm in normalize_text(alias) for alias in profile.aliases)
                or query_norm in normalize_text(profile.text())
            )
        ]
        return sorted(matches, key=lambda item: item.updated_at, reverse=True)[:limit]

    def find_user_groups(self, user_id: str) -> list[GroupMemorySpace]:
        user_id = str(user_id or "").strip()
        if not user_id:
            return []
        group_ids = {
            profile.group_id
            for profile in self.profiles.values()
            if profile.user_id == user_id
        }
        return sorted(
            [group for group_id, group in self.groups.items() if group_id in group_ids and group.kind == "group"],
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def sync_private_profile_to_user_groups(self, private_group_id: str, user_id: str) -> list[str]:
        source = self.get_profile(private_group_id, user_id)
        if not source:
            return []
        synced: list[str] = []
        for group in self.find_user_groups(user_id):
            target = self.get_profile(group.id, user_id)
            if not target:
                target = self.touch_profile(
                    group.id,
                    user_id,
                    source.display_name or user_id,
                    source.group_role,
                    "private sync",
                )
            if source.display_name and not target.display_name:
                target.display_name = source.display_name
            for alias in source.aliases:
                if alias and alias not in target.aliases:
                    target.aliases.append(alias)
            target.aliases = target.aliases[:12]
            existing = {normalize_text(str(item.get("fact") or "")) for item in target.facts}
            changed = False
            for item in source.facts:
                norm = normalize_text(str(item.get("fact") or ""))
                if not norm or norm in existing:
                    continue
                copied = dict(item)
                copied["source"] = f"private_sync:{item.get('source', '')}".rstrip(":")
                target.facts.insert(0, copied)
                existing.add(norm)
                changed = True
            if changed:
                target.facts = target.facts[:50]
                target.updated_at = int(time.time())
                synced.append(group.id)
        if synced:
            self.save()
        return synced

    def sync_private_relations_to_user_groups(self, private_group_id: str, user_id: str) -> list[str]:
        source_records = [
            record
            for record in self.records.values()
            if record.group_id == private_group_id
        ]
        if not source_records:
            return []
        synced: list[str] = []
        now = int(time.time())
        for group in self.find_user_groups(user_id):
            changed = False
            for source in source_records:
                target_id = relation_id(group.id, source.subject, source.relation, source.object)
                target = self.records.get(target_id)
                if target and target.updated_at >= source.updated_at:
                    continue
                source_label = f"private_sync:{source.source}".rstrip(":")
                if target:
                    target.note = source.note or target.note
                    target.source = source_label
                    target.confidence = max(target.confidence, source.confidence)
                    target.updated_at = now
                    target.embedding_provider_id = source.embedding_provider_id
                    target.embedding_dim = source.embedding_dim
                else:
                    target = RelationRecord(
                        id=target_id,
                        group_id=group.id,
                        subject=source.subject,
                        relation=source.relation,
                        object=source.object,
                        note=source.note,
                        source=source_label,
                        confidence=source.confidence,
                        created_at=now,
                        updated_at=now,
                        embedding_provider_id=source.embedding_provider_id,
                        embedding_dim=source.embedding_dim,
                    )
                    self.records[target_id] = target
                self.vectors[target_id] = list(self.vectors.get(source.id) or embed_text(target.text()))
                changed = True
            if changed:
                synced.append(group.id)
        if synced:
            self.save()
        return synced

    def export_profiles(self, group_id: str) -> list[dict[str, Any]]:
        return [
            asdict(profile)
            for profile in sorted(
                self.profiles.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )
            if profile.group_id == group_id
        ]

    def export_groups(self) -> list[dict[str, Any]]:
        return [
            asdict(group)
            for group in sorted(
                self.groups.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )
        ]

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


def format_profile(profile: UserProfile, max_facts: int = 5) -> str:
    name = profile.display_name or profile.user_id
    alias_text = f" aliases={','.join(profile.aliases[:4])}" if profile.aliases else ""
    role_text = f" role={profile.group_role}" if profile.group_role else ""
    facts = [
        str(item.get("fact") or "").strip()
        for item in profile.facts[:max_facts]
        if str(item.get("fact") or "").strip()
    ]
    fact_text = "；".join(facts) if facts else "暂无画像事实"
    return f"[{profile.user_id}] {name}{role_text}{alias_text}：{fact_text}"
