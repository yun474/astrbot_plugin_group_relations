from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

BASIC_PROFILE_FIELDS = {"likes", "dislikes", "traits", "notes"}


@dataclass
class RelationRecord:
    id: str
    group_id: str
    subject: str
    relation: str
    object: str
    subject_user_id: str = ""
    object_user_id: str = ""
    category: str = "relation"
    note: str = ""
    source: str = "manual"
    confidence: float = 1.0
    importance: float = 0.6
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def text(self) -> str:
        parts = [
            self.subject,
            self.subject_user_id,
            self.relation,
            self.object,
            self.object_user_id,
            self.category,
            self.note,
        ]
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
    member_directory_updated_at: int = 0
    member_directory_source: str = ""
    member_count: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    message_count: int = 0


@dataclass
class GroupMember:
    id: str
    group_id: str
    user_id: str
    display_name: str = ""
    card: str = ""
    nickname: str = ""
    recall_name_preference: str = ""
    role: str = "member"
    source: str = "event"
    active: bool = True
    first_seen_at: int = field(default_factory=lambda: int(time.time()))
    last_seen_at: int = field(default_factory=lambda: int(time.time()))
    verified_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class UserProfile:
    id: str
    group_id: str
    user_id: str
    display_name: str = ""
    preferred_name: str = ""
    aliases: list[str] = field(default_factory=list)
    basic_profile: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    group_role: str = "unknown"
    role_evidence: str = ""
    role_updated_at: int = 0
    facts: list[dict[str, Any]] = field(default_factory=list)
    message_count: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def text(self) -> str:
        fact_text = " ".join(str(item.get("fact") or "") for item in self.facts)
        basic_text = " ".join(
            str(item.get("value") or "")
            for items in self.basic_profile.values()
            for item in items
            if isinstance(item, dict)
        )
        parts = [
            self.display_name,
            self.preferred_name,
            " ".join(self.aliases),
            basic_text,
            self.group_role,
            fact_text,
        ]
        return " ".join(part for part in parts if part).strip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def role_rank(role: str) -> int:
    return {"unknown": 0, "member": 1, "admin": 2, "owner": 3}.get(normalize_text(role), 0)


def profile_id(group_id: str, user_id: str) -> str:
    raw = f"{group_id}|{normalize_text(user_id)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def member_id(group_id: str, user_id: str) -> str:
    raw = f"{group_id}|{normalize_text(user_id)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def relation_id(
    group_id: str,
    subject: str,
    relation: str,
    object_: str,
    subject_user_id: str = "",
    object_user_id: str = "",
) -> str:
    subject_key = normalize_text(subject_user_id) or normalize_text(subject)
    object_key = normalize_text(object_user_id) or normalize_text(object_)
    raw = f"{group_id}|{subject_key}|{normalize_text(relation)}|{object_key}"
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


def normalize_basic_profile_field(field: str) -> str:
    field = normalize_text(field)
    mapping = {
        "like": "likes",
        "likes": "likes",
        "hobby": "likes",
        "hobbies": "likes",
        "preference": "likes",
        "preferences": "likes",
        "爱好": "likes",
        "喜欢": "likes",
        "dislike": "dislikes",
        "dislikes": "dislikes",
        "hate": "dislikes",
        "hates": "dislikes",
        "讨厌": "dislikes",
        "不喜欢": "dislikes",
        "trait": "traits",
        "traits": "traits",
        "feature": "traits",
        "identity": "traits",
        "特点": "traits",
        "特征": "traits",
        "身份": "traits",
        "note": "notes",
        "notes": "notes",
        "备注": "notes",
    }
    return mapping.get(field, field)


def _coerce_basic_profile(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {field: [] for field in BASIC_PROFILE_FIELDS}
    now = int(time.time())
    result: dict[str, list[dict[str, Any]]] = {field: [] for field in BASIC_PROFILE_FIELDS}
    for raw_field, raw_items in raw.items():
        field = normalize_basic_profile_field(raw_field)
        if field not in BASIC_PROFILE_FIELDS:
            continue
        item_list = raw_items if isinstance(raw_items, list) else [raw_items]
        seen: set[tuple[str, str]] = set()
        for raw_item in item_list:
            if isinstance(raw_item, dict):
                item = dict(raw_item)
                value = str(item.get("value") or "").strip()
                key = str(item.get("key") or "").strip()
            else:
                value = str(raw_item or "").strip()
                key = ""
                item = {}
            if not value:
                continue
            key = key or normalize_text(value)[:40]
            dedupe_key = (normalize_text(key), normalize_text(value))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            result[field].append(
                {
                    "key": key[:40],
                    "value": re.sub(r"\s+", " ", value)[:120],
                    "note": str(item.get("note") or "").strip()[:240],
                    "source": str(item.get("source") or "manual").strip()[:40],
                    "confidence": clamp_confidence(item.get("confidence", 0.8), 0.8),
                    "importance": clamp_confidence(item.get("importance", 0.8), 0.8),
                    "created_at": int(item.get("created_at") or now),
                    "updated_at": int(item.get("updated_at") or item.get("created_at") or now),
                }
            )
        result[field] = sorted(
            result[field],
            key=lambda item: (item.get("importance", 0.0), item.get("updated_at", 0)),
            reverse=True,
        )[:24]
    return result


class RelationStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_file = data_dir / "relations.json"
        self.records: dict[str, RelationRecord] = {}
        self.groups: dict[str, GroupMemorySpace] = {}
        self.profiles: dict[str, UserProfile] = {}
        self.members: dict[str, GroupMember] = {}

    def load(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self.records = {}
            self.groups = {}
            self.profiles = {}
            self.members = {}
            return
        try:
            payload = json.loads(self.data_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.records = {}
            self.groups = {}
            self.profiles = {}
            self.members = {}
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
        self.members = {}
        for item in payload.get("members", []):
            member = self._coerce_member(item)
            if member:
                self.members[member.id] = member
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
            payload["member_directory_updated_at"] = int(payload.get("member_directory_updated_at") or 0)
            payload["member_count"] = int(payload.get("member_count") or 0)
            return GroupMemorySpace(**payload)
        except (TypeError, ValueError):
            return None

    def _coerce_member(self, item: Any) -> GroupMember | None:
        if not isinstance(item, dict):
            return None
        if not item.get("group_id") or not item.get("user_id"):
            return None
        allowed = {field_.name for field_ in fields(GroupMember)}
        payload = {key: value for key, value in item.items() if key in allowed}
        try:
            payload["id"] = payload.get("id") or member_id(payload["group_id"], payload["user_id"])
            payload["role"] = normalize_text(payload.get("role") or "member") or "member"
            payload["display_name"] = str(payload.get("display_name") or "").strip()
            payload["card"] = str(payload.get("card") or "").strip()
            payload["nickname"] = str(payload.get("nickname") or "").strip()
            payload["recall_name_preference"] = str(payload.get("recall_name_preference") or "").strip()
            payload["active"] = bool(payload.get("active", True))
            payload["first_seen_at"] = int(payload.get("first_seen_at") or time.time())
            payload["last_seen_at"] = int(payload.get("last_seen_at") or payload["first_seen_at"])
            payload["verified_at"] = int(payload.get("verified_at") or payload["last_seen_at"])
            return GroupMember(**payload)
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
            payload["preferred_name"] = str(payload.get("preferred_name") or "").strip()[:80]
            payload["aliases"] = [
                str(alias).strip()
                for alias in payload.get("aliases", [])
                if str(alias).strip()
            ][:12]
            payload["basic_profile"] = _coerce_basic_profile(payload.get("basic_profile", {}))
            facts = []
            for raw_fact in payload.get("facts", []):
                if not isinstance(raw_fact, dict) or not str(raw_fact.get("fact") or "").strip():
                    continue
                fact = dict(raw_fact)
                fact["category"] = str(fact.get("category") or "impression").strip()
                fact["importance"] = clamp_confidence(fact.get("importance", 0.6), 0.6)
                facts.append(fact)
            payload["facts"] = facts[:50]
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
            payload["importance"] = clamp_confidence(payload.get("importance", 0.6), 0.6)
            payload["created_at"] = int(payload.get("created_at") or time.time())
            payload["updated_at"] = int(payload.get("updated_at") or payload["created_at"])
            return RelationRecord(**payload)
        except (TypeError, ValueError):
            return None

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "groups": [asdict(group) for group in self.groups.values()],
            "members": [asdict(member) for member in self.members.values()],
            "profiles": [asdict(profile) for profile in self.profiles.values()],
            "relations": [asdict(record) for record in self.records.values()],
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
        group_ids.update(member.group_id for member in self.members.values())
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

    def get_member(self, group_id: str, user_id: str) -> GroupMember | None:
        return self.members.get(member_id(group_id, user_id))

    def has_member(self, group_id: str, user_id: str) -> bool:
        member = self.get_member(group_id, user_id)
        return bool(member and member.active)

    def member_directory_ready(self, group_id: str) -> bool:
        group = self.groups.get(group_id)
        return bool(group and group.member_directory_updated_at and group.member_count > 0)

    def upsert_group_member(
        self,
        group_id: str,
        user_id: str,
        display_name: str = "",
        card: str = "",
        nickname: str = "",
        role: str = "member",
        source: str = "event",
        recall_name_preference: str | None = None,
        active: bool = True,
        save: bool = True,
    ) -> GroupMember:
        now = int(time.time())
        mid = member_id(group_id, user_id)
        role = normalize_text(role) or "member"
        display_name = str(display_name or "").strip()
        card = str(card or "").strip()
        nickname = str(nickname or "").strip()
        member = self.members.get(mid)
        if not member:
            member = GroupMember(
                id=mid,
                group_id=group_id,
                user_id=str(user_id).strip(),
                display_name=display_name or card or nickname,
                card=card,
                nickname=nickname,
                recall_name_preference=str(recall_name_preference or "").strip(),
                role=role,
                source=source.strip() or "event",
                active=active,
                first_seen_at=now,
                last_seen_at=now,
                verified_at=now,
            )
            self.members[mid] = member
        else:
            if display_name:
                member.display_name = display_name
            if card:
                member.card = card
            if nickname:
                member.nickname = nickname
            if recall_name_preference is not None:
                member.recall_name_preference = str(recall_name_preference or "").strip()
            manual_role_locked = member.source in {"webui", "manual", "debug_command"} and source != "webui"
            if not manual_role_locked and role_rank(role) >= role_rank(member.role):
                member.role = role
                member.source = source.strip() or member.source
            member.active = active
            member.last_seen_at = now
            member.verified_at = now
        if save:
            self.save()
        return member

    def replace_group_members(
        self,
        group_id: str,
        members: list[dict[str, Any]],
        source: str = "platform",
    ) -> list[GroupMember]:
        now = int(time.time())
        seen: set[str] = set()
        updated: list[GroupMember] = []
        for item in members:
            user_id = str(item.get("user_id") or item.get("id") or "").strip()
            if not user_id:
                continue
            seen.add(user_id)
            card = str(item.get("card") or item.get("group_card") or "").strip()
            nickname = str(item.get("nickname") or item.get("nick") or item.get("name") or "").strip()
            updated.append(
                self.upsert_group_member(
                    group_id=group_id,
                    user_id=user_id,
                    display_name=str(item.get("display_name") or card or nickname).strip(),
                    card=card,
                    nickname=nickname,
                    role=str(item.get("role") or "member"),
                    source=source,
                    active=True,
                    save=False,
                )
            )
        for member in self.members.values():
            if member.group_id == group_id and member.user_id not in seen:
                member.active = False
        group = self.groups.get(group_id)
        if group:
            group.member_directory_updated_at = now
            group.member_directory_source = source
            group.member_count = len(seen)
            group.updated_at = now
        self.save()
        return updated

    def update_group_member_name_preference(
        self,
        group_id: str,
        user_id: str,
        recall_name_preference: str,
    ) -> GroupMember | None:
        member = self.get_member(group_id, user_id)
        if not member:
            return None
        now = int(time.time())
        member.recall_name_preference = str(recall_name_preference or "").strip()
        member.last_seen_at = now
        group = self.groups.get(group_id)
        if group:
            group.updated_at = now
        self.save()
        return member

    def update_group_member_role(
        self,
        group_id: str,
        user_id: str,
        role: str,
        source: str = "webui",
    ) -> GroupMember | None:
        member = self.get_member(group_id, user_id)
        if not member:
            return None
        role = normalize_text(role) or "member"
        if role == "manber":
            role = "member"
        if role not in {"owner", "admin", "member"}:
            return None
        now = int(time.time())
        member.role = role
        member.source = source.strip() or member.source
        member.active = True
        member.last_seen_at = now
        member.verified_at = now
        group = self.groups.get(group_id)
        if group:
            group.updated_at = now
        self.save()
        return member

    def refresh_member_directory_metadata(self, group_id: str, source: str = "event_fallback") -> None:
        group = self.groups.get(group_id)
        if not group:
            return
        now = int(time.time())
        group.member_directory_updated_at = now
        group.member_directory_source = source
        group.member_count = len(
            [
                member
                for member in self.members.values()
                if member.group_id == group_id and member.active
            ]
        )
        group.updated_at = now
        self.save()

    def export_members(self, group_id: str) -> list[dict[str, Any]]:
        return [
            asdict(member)
            for member in sorted(
                self.members.values(),
                key=lambda item: (role_rank(item.role), item.last_seen_at),
                reverse=True,
            )
            if member.group_id == group_id and member.active
        ]

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
                preferred_name="",
                aliases=[display_name] if display_name else [],
                basic_profile=_coerce_basic_profile({}),
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
        category: str = "impression",
        importance: float = 0.6,
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
            existing["category"] = category.strip() or existing.get("category", "impression")
            existing["importance"] = max(
                clamp_confidence(existing.get("importance", 0.0), 0.6),
                clamp_confidence(importance, 0.6),
            )
            existing["updated_at"] = now
        else:
            profile.facts.insert(
                0,
                {
                    "fact": fact[:160],
                    "category": category.strip() or "impression",
                    "note": note.strip()[:240],
                    "source": source,
                    "confidence": clamp_confidence(confidence, 0.8),
                    "importance": clamp_confidence(importance, 0.6),
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
        preferred_name: str | None = None,
        aliases: list[str] | None = None,
        group_role: str | None = None,
        role_evidence: str | None = None,
    ) -> UserProfile | None:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return None
        if display_name is not None:
            profile.display_name = display_name.strip()
        if preferred_name is not None:
            profile.preferred_name = preferred_name.strip()[:80]
            if profile.preferred_name and profile.preferred_name not in profile.aliases:
                profile.aliases.insert(0, profile.preferred_name)
        if aliases is not None:
            clean_aliases = []
            for raw_alias in aliases:
                alias = str(raw_alias).strip()
                if alias and alias not in clean_aliases:
                    clean_aliases.append(alias)
            if profile.preferred_name and profile.preferred_name not in clean_aliases:
                clean_aliases.insert(0, profile.preferred_name)
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

    def upsert_profile_basic(
        self,
        group_id: str,
        user_id: str,
        display_name: str,
        field: str,
        value: str,
        key: str = "",
        note: str = "",
        source: str = "manual",
        confidence: float = 0.8,
        importance: float = 0.8,
    ) -> UserProfile | None:
        field = normalize_basic_profile_field(field)
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if not value:
            return None
        profile = self.touch_profile(group_id, user_id, display_name)
        now = int(time.time())
        if field in {"preferred_name", "nickname", "nicknames", "alias", "aliases"}:
            if field == "preferred_name":
                profile.preferred_name = value[:80]
            if value not in profile.aliases:
                profile.aliases.insert(0, value[:80])
                profile.aliases = profile.aliases[:12]
            profile.updated_at = now
            self.save()
            return profile
        if field not in BASIC_PROFILE_FIELDS:
            return None
        profile.basic_profile = _coerce_basic_profile(profile.basic_profile)
        key = re.sub(r"\s+", " ", str(key or "").strip())[:40] or normalize_text(value)[:40]
        items = profile.basic_profile.setdefault(field, [])
        key_norm = normalize_text(key)
        value_norm = normalize_text(value)
        existing = next(
            (
                item
                for item in items
                if normalize_text(str(item.get("key") or "")) == key_norm
                or normalize_text(str(item.get("value") or "")) == value_norm
            ),
            None,
        )
        if existing:
            existing["key"] = key
            existing["value"] = value[:120]
            existing["note"] = note.strip()[:240] or existing.get("note", "")
            existing["source"] = source or existing.get("source", "manual")
            existing["confidence"] = max(
                clamp_confidence(existing.get("confidence", 0.0), 0.8),
                clamp_confidence(confidence, 0.8),
            )
            existing["importance"] = max(
                clamp_confidence(existing.get("importance", 0.0), 0.8),
                clamp_confidence(importance, 0.8),
            )
            existing["updated_at"] = now
        else:
            items.insert(
                0,
                {
                    "key": key,
                    "value": value[:120],
                    "note": note.strip()[:240],
                    "source": source,
                    "confidence": clamp_confidence(confidence, 0.8),
                    "importance": clamp_confidence(importance, 0.8),
                    "created_at": now,
                    "updated_at": now,
                },
            )
        profile.basic_profile[field] = sorted(
            items,
            key=lambda item: (item.get("importance", 0.0), item.get("updated_at", 0)),
            reverse=True,
        )[:24]
        profile.updated_at = now
        self.save()
        return profile

    def delete_profile_basic(
        self,
        profile_id_: str,
        group_id: str,
        field: str,
        value: str = "",
        key: str = "",
    ) -> bool:
        profile = self.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return False
        field = normalize_basic_profile_field(field)
        value_norm = normalize_text(value)
        key_norm = normalize_text(key)
        changed = False
        if field in {"preferred_name", "nickname", "nicknames", "alias", "aliases"}:
            if field == "preferred_name" and (not value_norm or normalize_text(profile.preferred_name) == value_norm):
                profile.preferred_name = ""
                changed = True
            old_aliases = list(profile.aliases)
            profile.aliases = [
                alias
                for alias in profile.aliases
                if value_norm and normalize_text(alias) != value_norm
            ] if value_norm else profile.aliases
            changed = changed or old_aliases != profile.aliases
        elif field in BASIC_PROFILE_FIELDS:
            profile.basic_profile = _coerce_basic_profile(profile.basic_profile)
            old_items = profile.basic_profile.get(field, [])
            profile.basic_profile[field] = [
                item
                for item in old_items
                if not (
                    (key_norm and normalize_text(str(item.get("key") or "")) == key_norm)
                    or (value_norm and normalize_text(str(item.get("value") or "")) == value_norm)
                )
            ]
            changed = len(profile.basic_profile[field]) != len(old_items)
        if changed:
            profile.updated_at = int(time.time())
            self.save()
        return changed

    def update_profile_fact(
        self,
        profile_id_: str,
        group_id: str,
        index: int,
        fact: str | None = None,
        note: str | None = None,
        confidence: float | None = None,
        category: str | None = None,
        importance: float | None = None,
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
        if category is not None and category.strip():
            item["category"] = category.strip()
        if importance is not None:
            item["importance"] = clamp_confidence(importance, 0.6)
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
        subject_user_id: str = "",
        object_user_id: str = "",
        category: str = "relation",
        note: str = "",
        source: str = "manual",
        confidence: float = 1.0,
        importance: float = 0.6,
    ) -> RelationRecord:
        subject_user_id = str(subject_user_id or "").strip()
        object_user_id = str(object_user_id or "").strip()
        category = str(category or "relation").strip()
        rid = relation_id(group_id, subject, relation, object_, subject_user_id, object_user_id)
        now = int(time.time())
        existing = self.records.get(rid)
        if existing:
            existing.subject_user_id = subject_user_id or existing.subject_user_id
            existing.object_user_id = object_user_id or existing.object_user_id
            existing.category = category or existing.category
            existing.note = note or existing.note
            existing.source = source or existing.source
            existing.confidence = max(existing.confidence, clamp_confidence(confidence))
            existing.importance = max(existing.importance, clamp_confidence(importance, 0.6))
            existing.updated_at = now
            record = existing
        else:
            record = RelationRecord(
                id=rid,
                group_id=group_id,
                subject=subject.strip(),
                relation=relation.strip(),
                object=object_.strip(),
                subject_user_id=subject_user_id,
                object_user_id=object_user_id,
                category=category,
                note=note.strip(),
                source=source,
                confidence=clamp_confidence(confidence),
                importance=clamp_confidence(importance, 0.6),
            )
            self.records[rid] = record
        self.save()
        return record

    def delete(self, relation_id_: str, group_id: str | None = None) -> bool:
        record = self.records.get(relation_id_)
        if not record or (group_id and record.group_id != group_id):
            return False
        del self.records[relation_id_]
        self.save()
        return True

    def update(
        self,
        relation_id_: str,
        group_id: str,
        subject: str | None = None,
        relation: str | None = None,
        object_: str | None = None,
        subject_user_id: str | None = None,
        object_user_id: str | None = None,
        category: str | None = None,
        note: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
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
        if subject_user_id is not None:
            record.subject_user_id = subject_user_id.strip()
        if object_user_id is not None:
            record.object_user_id = object_user_id.strip()
        if category is not None and category.strip():
            record.category = category.strip()
        if note is not None:
            record.note = note.strip()
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
        if importance is not None:
            record.importance = clamp_confidence(importance, 0.6)
        record.updated_at = int(time.time())
        new_id = relation_id(
            record.group_id,
            record.subject,
            record.relation,
            record.object,
            record.subject_user_id,
            record.object_user_id,
        )
        if new_id != relation_id_:
            self.records.pop(relation_id_, None)
            record.id = new_id
            if new_id in self.records:
                existing = self.records[new_id]
                existing.note = record.note or existing.note
                existing.source = record.source or existing.source
                existing.confidence = max(existing.confidence, record.confidence)
                existing.importance = max(existing.importance, record.importance)
                existing.subject_user_id = record.subject_user_id or existing.subject_user_id
                existing.object_user_id = record.object_user_id or existing.object_user_id
                existing.category = record.category or existing.category
                existing.updated_at = record.updated_at
                record = existing
            self.records[new_id] = record
        self.save()
        return record

    def by_person(self, group_id: str, person: str, limit: int = 8) -> list[RelationRecord]:
        needle = normalize_text(person)
        matches = [
            record
            for record in self.records.values()
            if record.group_id == group_id
            and (
                needle in normalize_text(record.subject)
                or needle in normalize_text(record.object)
                or needle in normalize_text(record.subject_user_id)
                or needle in normalize_text(record.object_user_id)
            )
        ]
        return sorted(matches, key=lambda item: item.updated_at, reverse=True)[:limit]

    def by_user_id(self, group_id: str, user_id: str, limit: int = 8) -> list[RelationRecord]:
        needle = normalize_text(user_id)
        if not needle:
            return self.recent(group_id, limit=limit)
        matches = [
            record
            for record in self.records.values()
            if record.group_id == group_id
            and (
                normalize_text(record.subject_user_id) == needle
                or normalize_text(record.object_user_id) == needle
                or normalize_text(record.subject) == needle
                or normalize_text(record.object) == needle
            )
        ]
        return sorted(
            matches,
            key=lambda item: (item.importance, item.updated_at),
            reverse=True,
        )[:limit]

    def get_profile(self, group_id: str, user_id: str) -> UserProfile | None:
        return self.profiles.get(profile_id(group_id, user_id))

    def find_profile_by_user_id(self, group_id: str, user_id: str) -> UserProfile | None:
        return self.get_profile(group_id, user_id)

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
            changed = False
            if source.display_name and not target.display_name:
                target.display_name = source.display_name
                changed = True
            if source.preferred_name and not target.preferred_name:
                target.preferred_name = source.preferred_name
                changed = True
            for alias in source.aliases:
                if alias and alias not in target.aliases:
                    target.aliases.append(alias)
                    changed = True
            target.aliases = target.aliases[:12]
            target.basic_profile = _coerce_basic_profile(target.basic_profile)
            source_basic = _coerce_basic_profile(source.basic_profile)
            for profile_field, items in source_basic.items():
                existing_basic = {
                    (normalize_text(str(item.get("key") or "")), normalize_text(str(item.get("value") or "")))
                    for item in target.basic_profile.get(profile_field, [])
                }
                for item in items:
                    dedupe = (
                        normalize_text(str(item.get("key") or "")),
                        normalize_text(str(item.get("value") or "")),
                    )
                    if dedupe in existing_basic:
                        continue
                    copied = dict(item)
                    copied["source"] = f"private_sync:{item.get('source', '')}".rstrip(":")
                    target.basic_profile.setdefault(profile_field, []).insert(0, copied)
                    existing_basic.add(dedupe)
                    changed = True
                target.basic_profile[profile_field] = target.basic_profile.get(profile_field, [])[:24]
            existing = {normalize_text(str(item.get("fact") or "")) for item in target.facts}
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
                target_id = relation_id(
                    group.id,
                    source.subject,
                    source.relation,
                    source.object,
                    source.subject_user_id,
                    source.object_user_id,
                )
                target = self.records.get(target_id)
                if target and target.updated_at >= source.updated_at:
                    continue
                source_label = f"private_sync:{source.source}".rstrip(":")
                if target:
                    target.note = source.note or target.note
                    target.source = source_label
                    target.confidence = max(target.confidence, source.confidence)
                    target.updated_at = now
                    target.subject_user_id = source.subject_user_id
                    target.object_user_id = source.object_user_id
                    target.category = source.category
                    target.importance = source.importance
                else:
                    target = RelationRecord(
                        id=target_id,
                        group_id=group.id,
                        subject=source.subject,
                        relation=source.relation,
                        object=source.object,
                        subject_user_id=source.subject_user_id,
                        object_user_id=source.object_user_id,
                        category=source.category,
                        note=source.note,
                        source=source_label,
                        confidence=source.confidence,
                        importance=source.importance,
                        created_at=now,
                        updated_at=now,
                    )
                    self.records[target_id] = target
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
            if query_norm in {
                normalize_text(record.subject_user_id),
                normalize_text(record.object_user_id),
            }:
                scored.append((record, 1.0))
                continue
            if query_norm in record_norm:
                scored.append((record, 0.25))
                continue
            record_tokens = set(_tokens(record_norm))
            if not query_tokens or not record_tokens:
                continue
            overlap = len(query_tokens & record_tokens) / max(len(query_tokens), 1)
            if overlap >= 0.15:
                scored.append((record, overlap))
        scored.sort(key=lambda item: (item[1], item[0].importance, item[0].updated_at), reverse=True)
        return scored[:limit]

    def export_group(self, group_id: str) -> list[dict[str, Any]]:
        return [
            asdict(record)
            for record in sorted(self.records.values(), key=lambda item: item.updated_at, reverse=True)
            if record.group_id == group_id
        ]

def format_record(record: RelationRecord, with_id: bool = True) -> str:
    prefix = f"[{record.id}] " if with_id else ""
    subject = f"{record.subject}({record.subject_user_id})" if record.subject_user_id else record.subject
    object_ = f"{record.object}({record.object_user_id})" if record.object_user_id else record.object
    category = f"[{record.category}] " if record.category and record.category != "relation" else ""
    note = f"（{record.note}）" if record.note else ""
    return f"{prefix}{category}{subject} --{record.relation}--> {object_}{note}"


def format_basic_profile(profile: UserProfile, max_items: int = 6) -> str:
    labels = {
        "likes": "喜欢",
        "dislikes": "讨厌",
        "traits": "特征",
        "notes": "备注",
    }
    parts = []
    if profile.preferred_name:
        parts.append(f"称呼: {profile.preferred_name}")
    for profile_field in ("likes", "dislikes", "traits", "notes"):
        values = [
            str(item.get("value") or "").strip()
            for item in profile.basic_profile.get(profile_field, [])[:max_items]
            if isinstance(item, dict) and str(item.get("value") or "").strip()
        ]
        if values:
            parts.append(f"{labels[profile_field]}: {'、'.join(values)}")
    return "；".join(parts)


def format_profile(profile: UserProfile, max_facts: int = 5) -> str:
    name = profile.preferred_name or profile.display_name or profile.user_id
    alias_text = f" aliases={','.join(profile.aliases[:4])}" if profile.aliases else ""
    role_text = f" role={profile.group_role}" if profile.group_role else ""
    basic_text = format_basic_profile(profile, max_items=4)
    facts = [
        str(item.get("fact") or "").strip()
        for item in profile.facts[:max_facts]
        if str(item.get("fact") or "").strip()
    ]
    fact_text = "；".join(facts) if facts else "暂无画像事实"
    detail = "；".join(part for part in [basic_text, fact_text] if part)
    return f"[{profile.user_id}] {name}{role_text}{alias_text}：{detail}"
