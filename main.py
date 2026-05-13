from __future__ import annotations

import asyncio
import json
from pathlib import Path

from quart import jsonify, request

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.provider import Provider
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .relation_store import (
    RelationRecord,
    RelationStore,
    format_basic_profile,
    format_profile,
    format_record,
)


PLUGIN_NAME = "astrbot_plugin_group_relations"


def _group_id(event: AstrMessageEvent) -> str:
    message_obj = getattr(event, "message_obj", None)
    group_id = getattr(message_obj, "group_id", "") if message_obj else ""
    return str(group_id or "")


def _group_name(event: AstrMessageEvent) -> str:
    group = getattr(getattr(event, "message_obj", None), "group", None)
    return str(getattr(group, "group_name", "") or "")


def _is_group_event(event: AstrMessageEvent) -> bool:
    message_obj = getattr(event, "message_obj", None)
    return bool(getattr(message_obj, "group_id", "") if message_obj else False)


def _sender_name(event: AstrMessageEvent) -> str:
    return str(event.get_sender_name() or event.get_sender_id() or "未知成员")


def _sender_id(event: AstrMessageEvent) -> str:
    return str(event.get_sender_id() or _sender_name(event) or "unknown")


def _split_config_list(value) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str):
        return {item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()}
    return set()


def normalize_role(value) -> str:
    role = str(value or "").strip().lower()
    if role in {"owner", "群主", "super_admin", "superadmin"}:
        return "owner"
    if role in {"admin", "administrator", "管理员", "manage", "manager"}:
        return "admin"
    if role in {"member", "manber", "群友", "成员", "user", "normal"}:
        return "member"
    return role


class GroupRelationsPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or getattr(self, "config", {}) or {}
        data_dir = Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        self.store = RelationStore(data_dir)
        self.store.load()
        self._warned_summary_provider_ids: set[str] = set()
        self._warned_no_summary_provider = False
        self._summary_buffers: dict[str, list[dict[str, str]]] = {}
        self._register_web_apis()

    def _cfg(self, primary: str, fallback: str | None = None, default=None):
        value = self.config.get(primary, None)
        if value is not None and value != "":
            return value
        if fallback:
            value = self.config.get(fallback, None)
            if value is not None and value != "":
                return value
        return default

    def _cfg_int(self, primary: str, fallback: str | None = None, default: int = 0) -> int:
        try:
            return int(self._cfg(primary, fallback, default))
        except (TypeError, ValueError):
            return default

    def _cfg_float(self, primary: str, fallback: str | None = None, default: float = 0.0) -> float:
        try:
            return float(self._cfg(primary, fallback, default))
        except (TypeError, ValueError):
            return default

    def _cfg_bool(self, primary: str, fallback: str | None = None, default: bool = False) -> bool:
        value = self._cfg(primary, fallback, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "开启", "是"}
        return bool(value)

    @filter.command_group("关系")
    def relations(self):
        """群关系记忆调试。"""
        pass

    @relations.command("help")
    async def help(self, event: AstrMessageEvent):
        """查看调试指令。"""
        yield event.plain_result(
            "\n".join(
                [
                    "群关系插件主要通过 LLM 自动注入和工具调用工作。",
                    "调试指令：",
                    "/关系 状态",
                    "/关系 群",
                    "/关系 用户 [用户ID或昵称]",
                    "/关系 画像 [用户ID或昵称]",
                    "/关系 删除画像 <用户ID或昵称> [画像关键词]",
                    "/关系 刷新目录",
                    "/关系 更新成员列表",
                    "/关系 调试 <查询词>",
                    "/关系 最近",
                ]
            )
        )

    @relations.command("状态")
    async def status(self, event: AstrMessageEvent):
        """查看插件状态。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        self._touch_event_scope(event)
        scope_id = self._scope_id(event)
        yield event.plain_result(
            "\n".join(
                [
                    f"自动注入：{bool(self.config.get('enable_context_injection', True))}",
                    f"自动总结：{bool(self.config.get('enable_dialogue_summary', False))}",
                    f"总结轮数：{self._cfg_int('自动总结_触发轮数', 'summary_trigger_rounds', 6)}",
                    f"注入条数：{self._cfg_int('记忆管理_每轮注入关系数量', 'injection_top_k', 5)}",
                    f"人物画像：{bool(self.config.get('enable_person_profile', True))}",
                    f"工具读取：{bool(self.config.get('enable_tool_read', True))}",
                    f"工具写入：{bool(self.config.get('enable_tool_write', False))}",
                    f"工具修改/删除：{bool(self.config.get('enable_tool_update', False))}",
                    "记忆隔离：群聊按群空间隔离，私聊按会话独立",
                    "召回方式：user_id 优先 + 文本匹配",
                    f"总结 Provider：{self._cfg('自动总结_模型Provider', 'summary_provider_id', '') or '当前会话模型'}",
                    f"总结参考人格：{self._summary_persona_label()}",
                    f"已记录群空间：{len(self.store.groups)}",
                    f"当前群关系数：{len(self.store.export_group(scope_id))}",
                    f"当前群用户画像数：{len(self.store.export_profiles(scope_id))}",
                ]
            )
        )

    @relations.command("群")
    async def group_status(self, event: AstrMessageEvent):
        """查看当前群空间概况。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        group = self._touch_event_scope(event)
        scope_id = self._scope_id(event)
        profiles = self.store.find_profiles(scope_id, limit=5)
        lines = [
            f"群空间：{group.name or group.id}",
            f"空间ID：{group.id}",
            f"类型：{group.kind}",
            f"群主：{group.owner_display_name or group.owner_user_id or '未知'}",
            f"成员目录：{group.member_count} 人 / {group.member_directory_source or '未初始化'}",
            f"消息触达：{group.message_count}",
            f"关系数：{len(self.store.export_group(scope_id))}",
            f"用户画像数：{len(self.store.export_profiles(scope_id))}",
        ]
        if profiles:
            lines.extend(["", "最近活跃用户画像："])
            lines.extend(format_profile(profile, max_facts=2) for profile in profiles)
        yield event.plain_result("\n".join(lines))

    @relations.command("刷新目录")
    async def refresh_directory(self, event: AstrMessageEvent):
        """刷新当前群成员目录。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        if not _is_group_event(event):
            yield event.plain_result("当前不是群聊，不能刷新群成员目录。")
            return
        group = self._touch_event_scope(event)
        await self._refresh_group_directory(event, force=True)
        group = self.store.groups.get(group.id) or group
        yield event.plain_result(
            "\n".join(
                [
                    f"群成员目录已刷新：{group.name or group.id}",
                    f"成员数：{group.member_count}",
                    f"来源：{group.member_directory_source or '未知'}",
                    f"群主：{group.owner_display_name or group.owner_user_id or '未知'}",
                ]
            )
        )

    @relations.command("更新成员列表")
    async def update_member_list(self, event: AstrMessageEvent):
        """管理员手动更新当前群成员列表。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        if not _is_group_event(event):
            yield event.plain_result("当前不是群聊，不能更新群成员列表。")
            return
        group = self._touch_event_scope(event)
        await self._refresh_group_directory(event, force=True)
        group = self.store.groups.get(group.id) or group
        yield event.plain_result(
            "\n".join(
                [
                    f"群成员列表已更新：{group.name or group.id}",
                    f"成员数：{group.member_count}",
                    f"来源：{group.member_directory_source or '未知'}",
                    f"群主：{group.owner_display_name or group.owner_user_id or '未知'}",
                    "WebUI 中手动修正过的成员身份不会被本次平台刷新覆盖。",
                ]
            )
        )

    @relations.command("用户")
    async def user_profile(self, event: AstrMessageEvent, query: str = ""):
        """查看当前群内用户画像。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        yield event.plain_result(self._format_profile_query_result(event, query))

    @relations.command("画像")
    async def profile(self, event: AstrMessageEvent, query: str = ""):
        """查看当前群内用户画像。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        yield event.plain_result(self._format_profile_query_result(event, query))

    @relations.command("删除画像")
    async def delete_profile(self, event: AstrMessageEvent, query: str = "", fact_query: str = ""):
        """删除当前群内用户画像或画像事实。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        self._touch_event_scope(event)
        query = query.strip()
        fact_query = fact_query.strip()
        if not query:
            yield event.plain_result("用法：/关系 删除画像 <用户ID或昵称> [画像关键词]")
            return
        profiles = self.store.find_profiles(
            self._scope_id(event),
            query=query,
            limit=self._cfg_int("记忆管理_画像查询返回人数", "max_profile_results", 5),
        )
        if not profiles:
            yield event.plain_result(f"没有找到「{query}」的群内画像。")
            return
        if len(profiles) > 1 and not any(profile.user_id == query or profile.id == query for profile in profiles):
            lines = ["匹配到多个画像，请用更准确的用户ID或昵称："]
            lines.extend(format_profile(profile, max_facts=2) for profile in profiles)
            yield event.plain_result("\n".join(lines))
            return
        profile = next(
            (item for item in profiles if item.user_id == query or item.id == query),
            profiles[0],
        )
        if fact_query:
            deleted = self.store.delete_profile_facts(profile.id, self._scope_id(event), fact_query)
            if not deleted:
                yield event.plain_result(f"没有删掉画像事实：{profile.display_name or profile.user_id} / {fact_query}")
                return
            yield event.plain_result(f"已删除 {profile.display_name or profile.user_id} 的 {deleted} 条画像事实。")
            return
        ok = self.store.delete_profile(profile.id, self._scope_id(event))
        yield event.plain_result(
            f"已删除 {profile.display_name or profile.user_id} 的群内画像。"
            if ok
            else "删除失败，画像可能已经不存在。"
        )

    @relations.command("调试")
    async def debug_search(self, event: AstrMessageEvent, query: str = ""):
        """调试自动召回结果。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        query = query or self._build_injection_query(event, None)
        matches = await self._search_relations(event, query)
        if not matches:
            yield event.plain_result(f"没有搜到和「{query}」接近的关系。")
            return
        lines = [
            f"query={query}",
            "",
            *[
                f"{format_record(record)}  score={score:.2f}"
                if bool(self.config.get("debug_include_scores", True))
                else format_record(record)
                for record, score in matches
            ],
        ]
        yield event.plain_result("\n".join(lines))

    @relations.command("最近")
    async def recent(self, event: AstrMessageEvent):
        """查看最近关系记录。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        records = self.store.recent(
            self._scope_id(event),
            limit=int(self.config.get("max_query_results", 8)),
        )
        yield event.plain_result("\n".join(format_record(record) for record in records) or "当前会话还没有关系记录。")

    def _register_web_apis(self) -> None:
        register = getattr(self.context, "register_web_api", None)
        if not callable(register):
            logger.warning("group relation WebUI disabled: context.register_web_api is unavailable.")
            return
        routes = [
            ("memory", self.web_memory, ["GET"], "Group relation memory overview"),
            ("group-save", self.web_group_save, ["POST"], "Update group memory space"),
            ("member-save", self.web_member_save, ["POST"], "Update group member role"),
            ("relation-save", self.web_relation_save, ["POST"], "Create or update group relation"),
            ("relation-delete", self.web_relation_delete, ["POST"], "Delete group relation"),
            ("profile-save", self.web_profile_save, ["POST"], "Create or update group user profile"),
            ("profile-basic-save", self.web_profile_basic_save, ["POST"], "Create or update stable group user profile field"),
            ("profile-basic-delete", self.web_profile_basic_delete, ["POST"], "Delete stable group user profile field"),
            ("profile-fact-save", self.web_profile_fact_save, ["POST"], "Create or update group user profile fact"),
            ("profile-fact-delete", self.web_profile_fact_delete, ["POST"], "Delete group user profile fact"),
            ("profile-delete", self.web_profile_delete, ["POST"], "Delete group user profile"),
        ]
        for endpoint, handler, methods, description in routes:
            try:
                register(f"/{PLUGIN_NAME}/{endpoint}", handler, methods, description)
            except Exception as exc:
                logger.warning(f"group relation failed to register WebUI API `{endpoint}`: {exc}")

    async def _web_request_json(self) -> dict:
        payload = await request.get_json(silent=True)
        return payload if isinstance(payload, dict) else {}

    def _web_ok(self, **payload):
        payload["ok"] = True
        return jsonify(payload)

    def _web_error(self, message: str, status: int = 400):
        return jsonify({"ok": False, "error": message}), status

    def _web_group_payload(self, group_id: str) -> dict:
        group = self.store.groups.get(group_id)
        relation_count = len(self.store.export_group(group_id))
        profile_count = len(self.store.export_profiles(group_id))
        payload = {
            "id": group_id,
            "name": group.name if group else group_id,
            "session_id": group.session_id if group else group_id,
            "kind": group.kind if group else "group",
            "owner_user_id": group.owner_user_id if group else "",
            "owner_display_name": group.owner_display_name if group else "",
            "owner_evidence": group.owner_evidence if group else "",
            "owner_updated_at": group.owner_updated_at if group else 0,
            "member_directory_updated_at": group.member_directory_updated_at if group else 0,
            "member_directory_source": group.member_directory_source if group else "",
            "member_count": group.member_count if group else 0,
            "created_at": group.created_at if group else 0,
            "updated_at": group.updated_at if group else 0,
            "message_count": group.message_count if group else 0,
            "relation_count": relation_count,
            "profile_count": profile_count,
        }
        return payload

    def _web_profile_payload(self, profile: dict) -> dict:
        profile = dict(profile)
        facts = []
        for index, item in enumerate(profile.get("facts", [])):
            if not isinstance(item, dict):
                continue
            fact = dict(item)
            fact["index"] = index
            facts.append(fact)
        profile["facts"] = facts
        return profile

    def _web_member_payload(self, member: dict) -> dict:
        return dict(member)

    def _web_memory_payload(self, selected_group_id: str = "") -> dict:
        self.store._ensure_groups()
        groups = [self._web_group_payload(group["id"]) for group in self.store.export_groups()]
        groups.sort(key=lambda item: (item["updated_at"], item["message_count"]), reverse=True)
        if not selected_group_id and groups:
            selected_group_id = groups[0]["id"]
        selected = self._web_group_payload(selected_group_id) if selected_group_id else None
        relations = self.store.export_group(selected_group_id) if selected_group_id else []
        profiles = [
            self._web_profile_payload(profile)
            for profile in self.store.export_profiles(selected_group_id)
        ] if selected_group_id else []
        members = [
            self._web_member_payload(member)
            for member in self.store.export_members(selected_group_id)
        ] if selected_group_id else []
        return {
            "plugin": PLUGIN_NAME,
            "groups": groups,
            "selected_group_id": selected_group_id,
            "selected_group": selected,
            "relations": relations,
            "profiles": profiles,
            "members": members,
        }

    async def web_memory(self):
        group_id = str(request.args.get("group_id", "") or "").strip()
        return jsonify(self._web_memory_payload(group_id))

    async def web_group_save(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        if not group_id:
            return self._web_error("missing group_id")
        group = self.store.update_group(
            group_id=group_id,
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "") or "group"),
            owner_user_id=str(data.get("owner_user_id", "")),
            owner_display_name=str(data.get("owner_display_name", "")),
            owner_evidence=str(data.get("owner_evidence", "webui")),
        )
        if not group:
            return self._web_error("group not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_relation_save(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        if not group_id:
            return self._web_error("missing group_id")
        relation_id_ = str(data.get("id") or data.get("relation_id") or "").strip()
        subject = str(data.get("subject") or "").strip()
        relation = str(data.get("relation") or "").strip()
        object_ = str(data.get("object") or data.get("object_") or "").strip()
        subject_user_id = str(data.get("subject_user_id") or "").strip()
        object_user_id = str(data.get("object_user_id") or "").strip()
        category = str(data.get("category") or "relation").strip()
        note = str(data.get("note") or "").strip()
        try:
            confidence = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        try:
            importance = float(data.get("importance", 0.6))
        except (TypeError, ValueError):
            importance = 0.6
        text = " ".join(part for part in [subject, subject_user_id, relation, object_, object_user_id, category, note] if part)
        if not text:
            return self._web_error("relation content is empty")
        if (
            self._cfg_bool("记忆写入_关系必须包含群成员ID", None, True)
            and self.store.groups.get(group_id)
            and self.store.groups[group_id].kind == "group"
            and not (subject_user_id or object_user_id)
        ):
            return self._web_error("relation requires subject_user_id or object_user_id in group spaces")
        if subject_user_id and not self._web_can_use_group_user(group_id, subject_user_id):
            return self._web_error(f"subject_user_id `{subject_user_id}` is not in this group member directory")
        if object_user_id and not self._web_can_use_group_user(group_id, object_user_id):
            return self._web_error(f"object_user_id `{object_user_id}` is not in this group member directory")
        if relation_id_:
            updated = self.store.update(
                relation_id_,
                group_id=group_id,
                subject=subject or None,
                relation=relation or None,
                object_=object_ or None,
                subject_user_id=subject_user_id,
                object_user_id=object_user_id,
                category=category,
                note=note,
                confidence=confidence,
                importance=importance,
            )
            if not updated:
                return self._web_error("relation not found", 404)
        else:
            if not subject or not relation or not object_:
                return self._web_error("subject, relation and object are required")
            self.store.upsert(
                group_id=group_id,
                subject=subject,
                relation=relation,
                object_=object_,
                subject_user_id=subject_user_id,
                object_user_id=object_user_id,
                category=category,
                note=note,
                source="webui",
                confidence=confidence,
                importance=importance,
            )
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_relation_delete(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        relation_id_ = str(data.get("id") or data.get("relation_id") or "").strip()
        if not group_id or not relation_id_:
            return self._web_error("missing group_id or relation_id")
        if not self.store.delete(relation_id_, group_id=group_id):
            return self._web_error("relation not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    def _web_can_use_group_user(self, group_id: str, user_id: str) -> bool:
        group = self.store.groups.get(group_id)
        if not group or group.kind != "group":
            return True
        if not self._cfg_bool("群成员目录_写入时校验", None, True):
            return True
        if not self.store.member_directory_ready(group_id):
            return False
        return self.store.has_member(group_id, user_id)

    async def web_member_save(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        user_id = str(data.get("user_id") or "").strip()
        role = normalize_role(data.get("role") or "member")
        if not group_id or not user_id:
            return self._web_error("missing group_id or user_id")
        if role not in {"owner", "admin", "member"}:
            return self._web_error("role must be owner, admin or member")
        member = self._apply_member_role_override(group_id, user_id, role, "webui")
        if not member:
            return self._web_error("member not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_profile_save(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        user_id = str(data.get("user_id") or "").strip()
        profile_id_ = str(data.get("id") or data.get("profile_id") or "").strip()
        display_name = str(data.get("display_name") or "").strip()
        preferred_name = str(data.get("preferred_name") or "").strip()
        group_role = str(data.get("group_role") or "").strip()
        role_evidence = str(data.get("role_evidence") or "webui").strip()
        aliases_raw = data.get("aliases", [])
        if isinstance(aliases_raw, str):
            aliases = [item.strip() for item in aliases_raw.replace(",", "\n").splitlines() if item.strip()]
        elif isinstance(aliases_raw, list):
            aliases = [str(item).strip() for item in aliases_raw if str(item).strip()]
        else:
            aliases = []
        if not group_id:
            return self._web_error("missing group_id")
        if profile_id_:
            profile = self.store.update_profile(
                profile_id_,
                group_id,
                display_name=display_name,
                preferred_name=preferred_name,
                aliases=aliases,
                group_role=group_role,
                role_evidence=role_evidence,
            )
            if not profile:
                return self._web_error("profile not found", 404)
        else:
            if not user_id:
                return self._web_error("user_id is required")
            if not self._web_can_use_group_user(group_id, user_id):
                return self._web_error(f"user_id `{user_id}` is not in this group member directory")
            profile = self.store.touch_profile(group_id, user_id, display_name or user_id)
            self.store.update_profile(
                profile.id,
                group_id,
                display_name=display_name or user_id,
                preferred_name=preferred_name,
                aliases=aliases,
                group_role=group_role or "unknown",
                role_evidence=role_evidence,
            )
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_profile_basic_save(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        profile_id_ = str(data.get("profile_id") or "").strip()
        field = str(data.get("field") or "").strip()
        key = str(data.get("key") or "").strip()
        value = str(data.get("value") or "").strip()
        note = str(data.get("note") or "").strip()
        try:
            confidence = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        try:
            importance = float(data.get("importance", 0.8))
        except (TypeError, ValueError):
            importance = 0.8
        if not group_id or not profile_id_:
            return self._web_error("missing group_id or profile_id")
        profile = self.store.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return self._web_error("profile not found", 404)
        if not field or not value:
            return self._web_error("field and value are required")
        updated = self.store.upsert_profile_basic(
            group_id=group_id,
            user_id=profile.user_id,
            display_name=profile.display_name,
            field=field,
            key=key,
            value=value,
            note=note,
            source="webui",
            confidence=confidence,
            importance=importance,
        )
        if not updated:
            return self._web_error("unsupported basic profile field")
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_profile_basic_delete(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        profile_id_ = str(data.get("profile_id") or "").strip()
        field = str(data.get("field") or "").strip()
        key = str(data.get("key") or "").strip()
        value = str(data.get("value") or "").strip()
        if not group_id or not profile_id_:
            return self._web_error("missing group_id or profile_id")
        if not self.store.delete_profile_basic(profile_id_, group_id, field, value=value, key=key):
            return self._web_error("basic profile item not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_profile_fact_save(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        profile_id_ = str(data.get("profile_id") or "").strip()
        fact = str(data.get("fact") or "").strip()
        note = str(data.get("note") or "").strip()
        category = str(data.get("category") or "impression").strip()
        try:
            confidence = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        try:
            importance = float(data.get("importance", 0.6))
        except (TypeError, ValueError):
            importance = 0.6
        if not group_id or not profile_id_:
            return self._web_error("missing group_id or profile_id")
        profile = self.store.profiles.get(profile_id_)
        if not profile or profile.group_id != group_id:
            return self._web_error("profile not found", 404)
        index_raw = data.get("index")
        if index_raw is None or str(index_raw) == "":
            if not fact:
                return self._web_error("fact is required")
            profile = self.store.remember_profile_fact(
                group_id=group_id,
                user_id=profile.user_id,
                display_name=profile.display_name,
                fact=fact,
                note=note,
                source="webui",
                confidence=confidence,
                category=category,
                importance=importance,
            )
            self._trim_profile_facts(profile)
        else:
            try:
                index = int(index_raw)
            except (TypeError, ValueError):
                return self._web_error("invalid fact index")
            if not self.store.update_profile_fact(
                profile_id_,
                group_id,
                index,
                fact=fact,
                note=note,
                confidence=confidence,
                category=category,
                importance=importance,
            ):
                return self._web_error("fact not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_profile_fact_delete(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        profile_id_ = str(data.get("profile_id") or "").strip()
        try:
            index = int(data.get("index"))
        except (TypeError, ValueError):
            return self._web_error("invalid fact index")
        if not group_id or not profile_id_:
            return self._web_error("missing group_id or profile_id")
        if not self.store.delete_profile_fact_index(profile_id_, group_id, index):
            return self._web_error("fact not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    async def web_profile_delete(self):
        data = await self._web_request_json()
        group_id = str(data.get("group_id") or "").strip()
        profile_id_ = str(data.get("profile_id") or data.get("id") or "").strip()
        if not group_id or not profile_id_:
            return self._web_error("missing group_id or profile_id")
        if not self.store.delete_profile(profile_id_, group_id):
            return self._web_error("profile not found", 404)
        return self._web_ok(memory=self._web_memory_payload(group_id))

    @filter.on_llm_request()
    async def inject_group_relations(self, event: AstrMessageEvent, req: ProviderRequest):
        """在每次 LLM 请求前注入少量当前群关系上下文。"""
        self._touch_event_scope(event)
        if event.get_extra("group_relation_skip_injection", False):
            return
        if not bool(self.config.get("enable_context_injection", True)):
            return
        query = self._build_injection_query(event, req)
        top_k = max(0, self._cfg_int("记忆管理_每轮注入关系数量", "injection_top_k", 5))
        matches = []
        if top_k > 0:
            matches = await self._search_relations(event, query, user_id=_sender_id(event), limit=top_k)
        if not matches and not self.store.get_profile(self._scope_id(event), _sender_id(event)):
            return
        text = self._build_injection_text(event, matches)
        max_length = int(self.config.get("max_injected_text_length", 1200))
        if max_length > 0 and len(text) > max_length:
            text = text[:max_length].rstrip() + "\n[已截断]"
        req.extra_user_content_parts.append(TextPart(text=text).mark_as_temp())

    @filter.on_llm_response()
    async def summarize_dialogue_relations(self, event: AstrMessageEvent, resp: LLMResponse):
        """每隔几轮对话，总结抽取当前会话的人物关系。"""
        self._touch_event_scope(event)
        if not bool(self.config.get("enable_dialogue_summary", False)):
            return
        if event.get_extra("group_relation_skip_summary", False):
            return
        user_text = (event.message_str or "").strip()
        assistant_text = (getattr(resp, "completion_text", "") or "").strip()
        if not user_text and not assistant_text:
            return
        scope_id = self._scope_id(event)
        buffer = self._summary_buffers.setdefault(scope_id, [])
        sender = f"{_sender_name(event)}({_sender_id(event)})"
        buffer.append({"role": "user", "name": sender, "content": user_text})
        buffer.append({"role": "assistant", "content": assistant_text})
        trigger_rounds = max(1, self._cfg_int("自动总结_触发轮数", "summary_trigger_rounds", 6))
        if len(buffer) < trigger_rounds * 2:
            return
        dialogue = self._format_summary_dialogue(buffer)
        max_chars = self._cfg_int("自动总结_最大对话长度", "summary_max_dialogue_chars", 4000)
        if max_chars > 0 and len(dialogue) > max_chars:
            dialogue = dialogue[-max_chars:]
        try:
            provider_id = await self._get_summary_provider_id(event)
            if not provider_id:
                logger.warning(
                    "group relation dialogue summary skipped: no summary provider available. "
                    "Configure summary_provider_id or set a current chat provider."
                )
                return
            event.set_extra("group_relation_skip_injection", True)
            event.set_extra("group_relation_skip_summary", True)
            summary_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=self._summary_extract_prompt(event, dialogue),
            )
            memories = self._parse_extracted_memories(summary_resp.completion_text)
            op_limit = max(1, self._cfg_int("记忆管理_每轮记忆更新操作上限", None, 6))
            memories = memories[:op_limit]
        except Exception as exc:
            logger.warning(f"group relation dialogue summary failed: {exc}")
            return
        finally:
            event.set_extra("group_relation_skip_injection", False)
            event.set_extra("group_relation_skip_summary", False)
        for item in memories:
            await self._remember_memory_item(event, item, source="summary", default_confidence=0.65)
        self._sync_private_profile_if_needed(event)
        self._summary_buffers[scope_id] = []

    @filter.llm_tool(name="group_user_context_lookup")
    async def group_user_context_lookup(self, event: AstrMessageEvent, user_id: str = "", query: str = ""):
        """一次性查询某个群成员在当前群环境里的身份、画像、关系和跨群补充。

        使用场景：
        - 回答前需要知道当前正在和谁聊天、对方在本群是什么身份、有哪些印象和人物关系。
        - 用户问“这个人是谁”“我和某人什么关系”“群主/管理员是谁”等环境问题。
        - 本轮注入的环境上下文不足，需要按 user_id 查完整人物上下文。

        注意：
        - 知道用户 ID 时必须传 user_id；query 只用于不知道 ID 时按昵称/关键词查找。
        - 当前群结果优先；跨群结果会标注来源群，只能作为补充背景。

        Args:
            user_id(string): 要查询的群成员用户 ID；留空时默认当前发言人。
            query(string): 不知道用户 ID 时用于搜索昵称、群名片或关键词。
        """
        if not bool(self.config.get("enable_tool_read", True)):
            yield event.plain_result("群人物上下文查询工具当前未启用。")
            return
        self._touch_event_scope(event)
        lookup_user_id = str(user_id or "").strip()
        if not lookup_user_id and query.strip():
            matched_profiles = self.store.find_profiles(
                self._scope_id(event),
                query=query,
                limit=1,
            )
            if matched_profiles:
                lookup_user_id = matched_profiles[0].user_id
            else:
                matches = await self._search_relations(event, query)
                lines = [f"没有找到「{query}」对应的群成员画像。"]
                if matches:
                    lines.extend(["按文本找到的关系："])
                    lines.extend(self._format_tool_record(event, record) for record, _score in matches)
                else:
                    lines.append("也没有找到相关关系；请提供准确 user_id 后再查。")
                yield event.plain_result("\n".join(lines))
                return
        lookup_user_id = lookup_user_id or _sender_id(event)
        text = self._format_user_context_lookup(event, lookup_user_id, query)
        yield event.plain_result(text)

    @filter.llm_tool(name="group_relation_save")
    async def group_relation_save(
        self,
        event: AstrMessageEvent,
        subject: str,
        relation: str,
        object_: str,
        subject_user_id: str = "",
        object_user_id: str = "",
        category: str = "relation",
        note: str = "",
        confidence: float = 0.8,
        importance: float = 0.6,
    ):
        """写入当前群空间内一条明确、稳定、可复用的人物关系上下文。

        只有当用户明确表达或明确纠正关系时才调用。适合记录：
        - A 是 B 的朋友/同学/同事/亲属/管理员/常一起玩的对象。
        - A 属于某组织、负责某职责、和某群体有稳定关系。
        - 用户明确纠正了之前的关系事实。

        不要记录玩笑、辱骂、临时情绪、暧昧猜测、敏感隐私或模型推断。

        Args:
            subject(string): 关系主体，例如人名、昵称、群成员、组织。
            relation(string): 关系类型，例如朋友、同学、管理员、常一起玩、合作、敌对。
            object_(string): 关系客体，例如另一个人名、昵称、群成员、组织。
            subject_user_id(string): 主体是群成员时填写其用户 ID；不知道可留空。
            object_user_id(string): 客体是群成员时填写其用户 ID；不知道可留空。
            category(string): 关系类别，如 relationship / role / preference / memory / correction。
            note(string): 补充说明或证据摘要，没有可以留空。
            confidence(number): 关系可信度，0 到 1 之间。
            importance(number): 长期重要度，0 到 1 之间；普通小事不要高于 0.5。
        """
        if not bool(self.config.get("enable_tool_write", False)):
            yield event.plain_result("群关系记忆写入工具当前未启用。")
            return
        self._touch_event_scope(event)
        confidence = max(0.0, min(1.0, float(confidence)))
        importance = max(0.0, min(1.0, float(importance)))
        ok, reason = await self._validate_relation_users_for_write(
            event,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
        )
        if not ok:
            yield event.plain_result(reason)
            return
        if not self._is_relation_worth_storing(subject, relation, object_, note, confidence, importance, "tool"):
            yield event.plain_result("这条关系太像临时闲聊或低价值小事，已拒绝写入。")
            return
        record = await self._remember_relation(
            group_id=self._scope_id(event),
            subject=subject,
            relation=relation,
            object_=object_,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            category=category,
            note=note,
            source="tool",
            confidence=confidence,
            importance=importance,
        )
        self._sync_private_profile_if_needed(event)
        yield event.plain_result(f"已写入群关系上下文：{format_record(record)}")

    @filter.llm_tool(name="group_user_profile_save")
    async def group_user_profile_save(
        self,
        event: AstrMessageEvent,
        user_id: str,
        fact: str,
        display_name: str = "",
        note: str = "",
        confidence: float = 0.8,
        category: str = "impression",
        importance: float = 0.6,
    ):
        """写入当前群空间内某个用户的稳定画像事实，用于帮助理解群环境和人物关系。

        只有当用户本人自述、管理员确认、或群聊中明确确认时才调用。适合记录：
        - 用户ID对应的昵称/群名片、常用别名。
        - 稳定身份、偏好、长期兴趣、常一起互动的人。
        - 对之后问答有帮助的非敏感背景。

        不要记录一次性闲聊、攻击性评价、隐私敏感信息、未经确认的猜测。

        Args:
            user_id(string): 用户 ID；记录当前发言人时传当前发言人 ID。
            fact(string): 明确、稳定、对之后聊天有帮助的用户画像事实。
            display_name(string): 用户昵称或群名片，不知道可留空。
            note(string): 证据摘要，没有可以留空。
            confidence(number): 可信度，0 到 1 之间。
            category(string): 画像类别，如 identity / preference / impression / memory / correction。
            importance(number): 长期重要度，0 到 1 之间；普通小事不要高于 0.5。
        """
        if not bool(self.config.get("enable_tool_write", False)):
            yield event.plain_result("群用户画像写入工具当前未启用。")
            return
        self._touch_event_scope(event)
        target_user_id = str(user_id or _sender_id(event))
        ok, reason = await self._validate_group_user_for_write(event, target_user_id, display_name or _sender_name(event))
        if not ok:
            yield event.plain_result(reason)
            return
        if not self._is_profile_fact_worth_storing(fact, note, confidence, importance, "tool"):
            yield event.plain_result("这条画像太像临时闲聊或低价值小事，已拒绝写入。")
            return
        profile = self.store.remember_profile_fact(
            group_id=self._scope_id(event),
            user_id=target_user_id,
            display_name=display_name or _sender_name(event),
            fact=fact,
            note=note,
            source="tool",
            confidence=max(0.0, min(1.0, float(confidence))),
            category=category,
            importance=max(0.0, min(1.0, float(importance))),
        )
        self._trim_profile_facts(profile)
        self._sync_private_profile_if_needed(event)
        yield event.plain_result(f"已写入群用户画像上下文：{format_profile(profile)}")

    @filter.llm_tool(name="group_user_basic_profile_update")
    async def group_user_basic_profile_update(
        self,
        event: AstrMessageEvent,
        user_id: str,
        operation: str,
        field: str,
        value: str = "",
        key: str = "",
        old_value: str = "",
        display_name: str = "",
        note: str = "",
        confidence: float = 0.8,
        importance: float = 0.8,
    ):
        """按用户明确说法增删改稳定基础画像字段。

        适合记录和维护昵称、希望被如何称呼、爱好、讨厌的东西、稳定特征和长期备注。
        删除或替换必须来自用户明确否定、纠正或要求不要再这样称呼/记录。

        Args:
            user_id(string): 目标用户 ID；记录当前发言人时传当前发言人 ID。
            operation(string): upsert / delete / replace。
            field(string): preferred_name / aliases / likes / dislikes / traits / notes。
            value(string): 新值；delete 时是要删除的值。
            key(string): 可选分类键，如 game / food / nickname。
            old_value(string): replace 时要删除的旧值。
            display_name(string): 目标用户昵称或群名片。
            note(string): 用户原话或证据摘要。
            confidence(number): 可信度，0 到 1。
            importance(number): 长期重要度，0 到 1。
        """
        operation = str(operation or "upsert").strip().lower()
        if operation in {"delete", "remove"}:
            if not bool(self.config.get("enable_tool_update", False)):
                yield event.plain_result("群用户基础画像删除工具当前未启用。")
                return
        elif not bool(self.config.get("enable_tool_write", False)):
            yield event.plain_result("群用户基础画像写入工具当前未启用。")
            return
        self._touch_event_scope(event)
        target_user_id = str(user_id or _sender_id(event)).strip()
        ok, reason = await self._validate_group_user_for_write(event, target_user_id, display_name or _sender_name(event))
        if not ok:
            yield event.plain_result(reason)
            return
        profile = self.store.get_profile(self._scope_id(event), target_user_id)
        if operation in {"delete", "remove", "replace"}:
            if not profile:
                yield event.plain_result("没有找到这个用户的画像，无法删除基础字段。")
                return
            delete_value = old_value if operation == "replace" and old_value else value
            if delete_value:
                self.store.delete_profile_basic(profile.id, self._scope_id(event), field, value=delete_value, key=key)
        if operation not in {"delete", "remove"}:
            if not value:
                yield event.plain_result("缺少基础画像新值。")
                return
            profile = self.store.upsert_profile_basic(
                group_id=self._scope_id(event),
                user_id=target_user_id,
                display_name=display_name or _sender_name(event),
                field=field,
                key=key,
                value=value,
                note=note,
                source="tool",
                confidence=max(0.0, min(1.0, float(confidence))),
                importance=max(0.0, min(1.0, float(importance))),
            )
        self._sync_private_profile_if_needed(event)
        profile = profile or self.store.get_profile(self._scope_id(event), target_user_id)
        yield event.plain_result(
            f"已更新基础画像：{format_basic_profile(profile)}" if profile else "基础画像已更新。"
        )

    @filter.llm_tool(name="group_relation_update")
    async def group_relation_update(
        self,
        event: AstrMessageEvent,
        relation_id: str,
        subject: str = "",
        relation: str = "",
        object_: str = "",
        subject_user_id: str = "",
        object_user_id: str = "",
        category: str = "",
        note: str = "",
        confidence: float = -1.0,
        importance: float = -1.0,
    ):
        """根据用户明确纠正，修改一条当前群空间内的人物关系记忆。

        只有当用户明确指出旧记忆错误或给出更准确说法时才调用。
        修改前应尽量先通过 group_user_context_lookup 找到 relation_id。

        Args:
            relation_id(string): 要修改的关系 ID。
            subject(string): 新主体，不改则留空。
            relation(string): 新关系，不改则留空。
            object_(string): 新客体，不改则留空。
            subject_user_id(string): 新主体用户 ID，不改则留空。
            object_user_id(string): 新客体用户 ID，不改则留空。
            category(string): 新类别，不改则留空。
            note(string): 新补充说明，不改则留空。
            confidence(number): 新可信度，0 到 1；不改传 -1。
            importance(number): 新长期重要度，0 到 1；不改传 -1。
        """
        if not bool(self.config.get("enable_tool_update", False)):
            yield event.plain_result("群关系记忆修改工具当前未启用。")
            return
        self._touch_event_scope(event)
        old = self.store.records.get(relation_id)
        if not old or old.group_id != self._scope_id(event):
            yield event.plain_result("没有找到这条关系，或它不属于当前群。")
            return
        new_subject_user_id = subject_user_id or old.subject_user_id
        new_object_user_id = object_user_id or old.object_user_id
        ok, reason = await self._validate_relation_users_for_write(
            event,
            subject_user_id=new_subject_user_id,
            object_user_id=new_object_user_id,
        )
        if not ok:
            yield event.plain_result(reason)
            return
        updated = self.store.update(
            relation_id,
            group_id=self._scope_id(event),
            subject=subject or None,
            relation=relation or None,
            object_=object_ or None,
            subject_user_id=subject_user_id if subject_user_id else None,
            object_user_id=object_user_id if object_user_id else None,
            category=category if category else None,
            note=note if note else None,
            confidence=None if confidence < 0 else confidence,
            importance=None if importance < 0 else importance,
        )
        yield event.plain_result(f"已更新：{format_record(updated)}" if updated else "更新失败。")

    @filter.llm_tool(name="group_relation_delete")
    async def group_relation_delete(self, event: AstrMessageEvent, relation_id: str, reason: str = ""):
        """根据用户明确纠正，删除一条当前群空间内的人物关系记忆。

        只有当用户明确表示某条关系是错的、过期的或不应记录时才调用。
        删除前应尽量先通过 group_user_context_lookup 找到 relation_id。

        Args:
            relation_id(string): 要删除的关系 ID。
            reason(string): 删除原因摘要。
        """
        if not bool(self.config.get("enable_tool_update", False)):
            yield event.plain_result("群关系记忆删除工具当前未启用。")
            return
        self._touch_event_scope(event)
        ok = self.store.delete(relation_id, group_id=self._scope_id(event))
        yield event.plain_result("已删除。" if ok else "没有找到这条关系，或它不属于当前群。")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def auto_extract(self, event: AstrMessageEvent):
        """从群聊中自动抽取关系，默认关闭。"""
        self._touch_event_scope(event)
        if not bool(self.config.get("enable_auto_extract", False)):
            return
        message = event.message_str.strip()
        min_length = int(self.config.get("auto_extract_min_length", 12))
        if len(message) < min_length or message.startswith("/"):
            return
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            event.set_extra("group_relation_skip_injection", True)
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=self._extract_prompt(event, message),
            )
            memories = self._parse_extracted_memories(response.completion_text)
            op_limit = max(1, self._cfg_int("记忆管理_每轮记忆更新操作上限", None, 6))
            memories = memories[:op_limit]
        except Exception as exc:
            logger.warning(f"group relation auto extraction failed: {exc}")
            return
        finally:
            event.set_extra("group_relation_skip_injection", False)
        for item in memories:
            await self._remember_memory_item(event, item, source="auto", default_confidence=0.6)
        self._sync_private_profile_if_needed(event)

    async def _remember_relation(
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
        return self.store.upsert(
            group_id=group_id,
            subject=subject,
            relation=relation,
            object_=object_,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            category=category,
            note=note,
            source=source,
            confidence=confidence,
            importance=importance,
        )

    def _debug_denied_message(self) -> str:
        return "这个调试指令会暴露群关系记忆，当前只允许配置里的管理员使用。"

    def _apply_member_role_override(self, group_id: str, user_id: str, role: str, source: str):
        role = normalize_role(role)
        if role not in {"owner", "admin", "member"}:
            return None
        member = self.store.update_group_member_role(group_id, user_id, role, source=source)
        if not member:
            return None
        evidence = f"{source} member role override"
        profile = self.store.get_profile(group_id, user_id)
        if not profile:
            profile = self.store.touch_profile(group_id, user_id, member.display_name or user_id, role, evidence)
        self.store.update_profile(
            profile.id,
            group_id,
            display_name=profile.display_name or member.display_name or user_id,
            group_role=role,
            role_evidence=evidence,
        )
        group = self.store.groups.get(group_id)
        if role == "owner":
            self.store.set_group_owner(group_id, user_id, member.display_name or user_id, evidence)
        elif group and group.owner_user_id == user_id:
            self.store.set_group_owner(group_id, "", "", evidence)
        return member

    def _is_low_value_text(self, text: str) -> bool:
        text = str(text or "").strip()
        if len(text) < self._cfg_int("记忆写入_最小事实长度", None, 8):
            return True
        lowered = text.lower()
        low_value_markers = [
            "哈哈",
            "笑死",
            "草",
            "在吗",
            "谢谢",
            "好的",
            "今天",
            "刚才",
            "现在",
            "一会儿",
            "吃饭",
            "睡觉",
            "上线",
            "下线",
        ]
        return any(marker in lowered for marker in low_value_markers) and len(text) < 18

    def _is_profile_fact_worth_storing(
        self,
        fact: str,
        note: str,
        confidence: float,
        importance: float,
        source: str,
    ) -> bool:
        if source in {"manual", "webui"}:
            return bool(str(fact or "").strip())
        threshold = self._cfg_float("记忆写入_写入重要度阈值", None, 0.55)
        if confidence < self._cfg_float("记忆管理_写入可信度阈值", None, 0.65):
            return False
        if importance < threshold:
            return False
        text = " ".join(part for part in [fact, note] if part)
        return not self._is_low_value_text(text)

    def _is_relation_worth_storing(
        self,
        subject: str,
        relation: str,
        object_: str,
        note: str,
        confidence: float,
        importance: float,
        source: str,
    ) -> bool:
        if source in {"manual", "webui"}:
            return bool(subject and relation and object_)
        threshold = self._cfg_float("记忆写入_写入重要度阈值", None, 0.55)
        if confidence < self._cfg_float("记忆管理_写入可信度阈值", None, 0.65):
            return False
        if importance < threshold:
            return False
        text = " ".join(part for part in [subject, relation, object_, note] if part)
        return not self._is_low_value_text(text)

    def _format_profile_query_result(self, event: AstrMessageEvent, query: str = "") -> str:
        self._touch_event_scope(event)
        scope_id = self._scope_id(event)
        query = query.strip() or _sender_id(event)
        profiles = self.store.find_profiles(
            scope_id,
            query=query,
            limit=self._cfg_int("记忆管理_画像查询返回人数", "max_profile_results", 5),
        )
        if not profiles:
            return f"没有找到「{query}」的群内画像。"
        return "\n".join(format_profile(profile) for profile in profiles)

    def _can_use_debug_commands(self, event: AstrMessageEvent) -> bool:
        if bool(self.config.get("allow_public_debug_commands", False)):
            return True
        sender_id = str(event.get_sender_id() or "").strip()
        admin_ids = _split_config_list(self.config.get("relation_admin_user_ids", ""))
        if sender_id and sender_id in admin_ids:
            return True
        for attr_name in ("is_admin", "is_group_admin", "is_super_admin"):
            attr = getattr(event, attr_name, None)
            try:
                if callable(attr) and bool(attr()):
                    return True
                if attr is not None and not callable(attr) and bool(attr):
                    return True
            except Exception:
                continue
        return False

    def _touch_event_scope(self, event: AstrMessageEvent):
        scope_id = self._scope_id(event)
        if event.get_extra("group_relation_scope_touched", False):
            group = self.store.groups.get(scope_id)
            if group:
                return group
        group_name = _group_name(event)
        kind = "group" if _is_group_event(event) else "private"
        was_new_group = scope_id not in self.store.groups
        existing_profile = self.store.get_profile(scope_id, _sender_id(event))
        group_role = ""
        role_evidence = ""
        should_init_role = (
            self._cfg_bool("群身份_初始化扫描", None, True)
            and kind == "group"
            and (not existing_profile or not existing_profile.group_role or existing_profile.group_role == "unknown")
        )
        if should_init_role:
            group_role, role_evidence = self._resolve_sender_group_role(event)
        group = self.store.touch_group(
            group_id=scope_id,
            name=group_name or self._session_label(event),
            session_id=event.unified_msg_origin,
            kind=kind,
        )
        self.store.touch_profile(
            group_id=scope_id,
            user_id=_sender_id(event),
            display_name=_sender_name(event),
            group_role=group_role,
            role_evidence=role_evidence,
        )
        if kind == "group":
            self.store.upsert_group_member(
                scope_id,
                _sender_id(event),
                _sender_name(event),
                group_role or "member",
                role_evidence or "event message",
            )
        if group_role == "owner" and not group.owner_user_id:
            self.store.set_group_owner(
                scope_id,
                _sender_id(event),
                _sender_name(event),
                role_evidence or "group role initialization",
            )
        if kind == "group" and (
            was_new_group
            or (not group.member_directory_updated_at and self._cfg_bool("群成员目录_初始化获取", None, True))
        ):
            self._schedule_group_directory_refresh(event)
        event.set_extra("group_relation_scope_touched", True)
        return group

    def _resolve_sender_group_role(self, event: AstrMessageEvent) -> tuple[str, str]:
        role_value = normalize_role(getattr(event, "role", ""))
        if role_value in {"owner", "admin", "member"}:
            return role_value, "event.role"
        for attr_name, role in (
            ("is_super_admin", "owner"),
            ("is_group_owner", "owner"),
            ("is_owner", "owner"),
            ("is_admin", "admin"),
            ("is_group_admin", "admin"),
        ):
            attr = getattr(event, attr_name, None)
            try:
                matched = bool(attr()) if callable(attr) else bool(attr)
            except Exception:
                matched = False
            if matched:
                return role, f"event.{attr_name}"
        return "member", "group message initialization"

    def _schedule_group_directory_refresh(self, event: AstrMessageEvent) -> None:
        if event.get_extra("group_relation_directory_refresh_scheduled", False):
            return
        event.set_extra("group_relation_directory_refresh_scheduled", True)
        try:
            asyncio.create_task(self._refresh_group_directory(event, force=False))
        except RuntimeError:
            logger.warning("group relation cannot schedule group member directory refresh: no running event loop")

    async def _validate_group_user_for_write(
        self,
        event: AstrMessageEvent,
        user_id: str,
        display_name: str = "",
    ) -> tuple[bool, str]:
        user_id = str(user_id or "").strip()
        if not user_id:
            return False, "缺少 user_id，不能写入群用户画像。"
        if not _is_group_event(event) or not self._cfg_bool("群成员目录_写入时校验", None, True):
            return True, ""
        group_id = self._scope_id(event)
        if self.store.has_member(group_id, user_id):
            return True, ""
        await self._refresh_group_directory(event, force=True)
        if self.store.has_member(group_id, user_id):
            return True, ""
        if user_id == _sender_id(event):
            self.store.upsert_group_member(group_id, user_id, display_name or _sender_name(event), "member", "event fallback")
            return True, ""
        return False, f"user_id `{user_id}` 不在当前群成员目录里，刷新目录后仍未找到，已拒绝写入。"

    async def _validate_relation_users_for_write(
        self,
        event: AstrMessageEvent,
        subject_user_id: str = "",
        object_user_id: str = "",
    ) -> tuple[bool, str]:
        user_ids = [value for value in [subject_user_id.strip(), object_user_id.strip()] if value]
        if (
            _is_group_event(event)
            and self._cfg_bool("记忆写入_关系必须包含群成员ID", None, True)
            and not user_ids
        ):
            return False, "关系记忆至少要包含一个群成员 user_id，避免把非本群人物写进群空间。"
        for user_id in user_ids:
            ok, reason = await self._validate_group_user_for_write(event, user_id)
            if not ok:
                return ok, reason
        return True, ""

    async def _refresh_group_directory(self, event: AstrMessageEvent, force: bool = False) -> None:
        if not _is_group_event(event):
            return
        group_id = self._scope_id(event)
        group = self.store.groups.get(group_id)
        if (
            group
            and group.member_directory_updated_at
            and not force
            and int(group.member_directory_updated_at) > 0
        ):
            return
        members = await self._fetch_group_member_directory(event)
        if members:
            self.store.replace_group_members(group_id, members, source="platform")
            for item in members:
                role = normalize_role(item.get("role", "member"))
                if role in {"owner", "admin"}:
                    self.store.touch_profile(
                        group_id,
                        str(item.get("user_id") or ""),
                        str(item.get("display_name") or item.get("nickname") or item.get("card") or ""),
                        role,
                        "platform member directory",
                    )
            owner = next((item for item in members if normalize_role(item.get("role", "")) == "owner"), None)
            if owner:
                self.store.set_group_owner(
                    group_id,
                    str(owner.get("user_id") or ""),
                    str(owner.get("display_name") or owner.get("nickname") or owner.get("card") or ""),
                    "platform member directory",
                )
            return
        self.store.upsert_group_member(
            group_id,
            _sender_id(event),
            _sender_name(event),
            self._resolve_sender_group_role(event)[0],
            "event fallback",
        )
        self.store.refresh_member_directory_metadata(group_id, "event_fallback_seen_only")

    async def _fetch_group_member_directory(self, event: AstrMessageEvent) -> list[dict]:
        group_id = _group_id(event)
        if not group_id:
            return []
        raw = await self._call_platform_action(event, "get_group_member_list", {"group_id": group_id})
        items = self._extract_action_data(raw)
        if not isinstance(items, list):
            return []
        members = []
        for item in items:
            if not isinstance(item, dict):
                continue
            user_id = str(item.get("user_id") or item.get("id") or "").strip()
            if not user_id:
                continue
            display_name = str(item.get("card") or item.get("nickname") or item.get("name") or user_id).strip()
            members.append(
                {
                    "user_id": user_id,
                    "display_name": display_name,
                    "role": normalize_role(item.get("role", "member")),
                }
            )
        return members

    async def _call_platform_action(self, event: AstrMessageEvent, action: str, params: dict):
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None)
        call_action = getattr(api, "call_action", None)
        if not callable(call_action):
            return None
        group_id = params.get("group_id")
        candidates = [params]
        if isinstance(group_id, str) and group_id.isdigit():
            converted = dict(params)
            converted["group_id"] = int(group_id)
            candidates.append(converted)
        for candidate in candidates:
            for mode in ("kwargs", "dict"):
                try:
                    if mode == "kwargs":
                        result = call_action(action, **candidate)
                    else:
                        result = call_action(action, candidate)
                    if hasattr(result, "__await__"):
                        result = await result
                    if result is not None:
                        return result
                except Exception as exc:
                    logger.debug(f"group relation platform action `{action}` failed via {mode}: {exc}")
        return None

    def _extract_action_data(self, raw):
        if isinstance(raw, dict):
            if "data" in raw:
                return raw["data"]
            if "retcode" in raw and "result" in raw:
                return raw["result"]
        return raw

    def _sync_private_profile_if_needed(self, event: AstrMessageEvent) -> None:
        if _is_group_event(event):
            return
        if not self._cfg_bool("私聊_同步到所属群空间", None, True):
            return
        profile_synced = self.store.sync_private_profile_to_user_groups(
            self._scope_id(event),
            _sender_id(event),
        )
        relation_synced = self.store.sync_private_relations_to_user_groups(
            self._scope_id(event),
            _sender_id(event),
        )
        synced = sorted(set(profile_synced + relation_synced))
        if synced:
            logger.info(
                f"group relation synced private memory for user `{_sender_id(event)}` "
                f"to groups: {', '.join(synced)}"
            )

    async def _get_summary_provider_id(self, event: AstrMessageEvent) -> str:
        provider_id = str(self._cfg("自动总结_模型Provider", "summary_provider_id", "")).strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if isinstance(provider, Provider):
                return provider_id
            if provider_id not in self._warned_summary_provider_ids:
                logger.warning(
                    f"group relation summary provider `{provider_id}` not found or invalid; "
                    "falling back to current chat provider."
                )
                self._warned_summary_provider_ids.add(provider_id)
        try:
            current_provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin
            )
        except Exception as exc:
            if not self._warned_no_summary_provider:
                logger.warning(
                    "group relation cannot resolve current chat provider for dialogue summary. "
                    f"Reason: {exc}"
                )
                self._warned_no_summary_provider = True
            return ""
        provider = self.context.get_provider_by_id(current_provider_id)
        if isinstance(provider, Provider):
            return current_provider_id
        if not self._warned_no_summary_provider:
            logger.warning(
                f"group relation current chat provider `{current_provider_id}` is unavailable "
                "or not a chat Provider; dialogue summary disabled until provider is ready."
            )
            self._warned_no_summary_provider = True
        return ""

    def _format_summary_dialogue(self, buffer: list[dict[str, str]]) -> str:
        lines = []
        for item in buffer:
            if item["role"] == "user":
                role = f"用户 {item.get('name', '').strip()}".strip()
            else:
                role = "机器人"
            content = item["content"].strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _search_relations(
        self,
        event: AstrMessageEvent,
        query: str,
        user_id: str = "",
        limit: int | None = None,
    ) -> list[tuple[RelationRecord, float]]:
        limit = limit or int(self.config.get("max_query_results", 8))
        lookup_user_id = str(user_id or "").strip()
        if lookup_user_id:
            matches = [(record, 1.0) for record in self.store.by_user_id(self._scope_id(event), lookup_user_id, limit=limit)]
            return self._cross_group_relation_fallback(
                event,
                query or lookup_user_id,
                self._cap_speaker_relations(event, matches),
                limit,
                user_id=lookup_user_id,
            )
        matches = self.store.search_by_text(self._scope_id(event), query, limit=limit)
        return self._cross_group_relation_fallback(event, query, self._cap_speaker_relations(event, matches), limit)

    def _cross_group_relation_fallback(
        self,
        event: AstrMessageEvent,
        query: str,
        matches: list[tuple[RelationRecord, float]],
        limit: int,
        user_id: str = "",
    ) -> list[tuple[RelationRecord, float]]:
        if not self._cfg_bool("记忆隔离_允许跨群召回", None, True):
            return matches
        if len(matches) >= limit:
            return matches[:limit]
        seen = {record.id for record, _score in matches}
        merged = list(matches)
        for group in self.store.find_user_groups(user_id or _sender_id(event)):
            if group.id == self._scope_id(event):
                continue
            cross_records = (
                [(record, 1.0) for record in self.store.by_user_id(group.id, user_id, limit=limit)]
                if user_id
                else self.store.search_by_text(group.id, query, limit=limit)
            )
            for record, score in cross_records:
                if record.id in seen:
                    continue
                seen.add(record.id)
                merged.append((record, score))
                if len(merged) >= limit:
                    return merged[:limit]
        return merged

    def _cross_group_profiles(self, user_id: str, query: str, profiles=None, current_group_id: str = ""):
        limit = self._cfg_int("记忆管理_画像查询返回人数", "max_profile_results", 5)
        merged = list(profiles or [])
        if len(merged) >= limit:
            return merged[:limit]
        seen = {profile.id for profile in merged}
        for group in self.store.find_user_groups(user_id):
            if current_group_id and group.id == current_group_id:
                continue
            for profile in self.store.find_profiles(group.id, query=query, limit=limit):
                if profile.id in seen:
                    continue
                seen.add(profile.id)
                merged.append(profile)
                if len(merged) >= limit:
                    return merged[:limit]
        return merged

    def _format_user_context_lookup(self, event: AstrMessageEvent, user_id: str, query: str = "") -> str:
        scope_id = self._scope_id(event)
        group = self.store.groups.get(scope_id)
        member = self.store.get_member(scope_id, user_id)
        profile = self.store.get_profile(scope_id, user_id)
        relations = self.store.by_user_id(scope_id, user_id, limit=int(self.config.get("max_query_results", 8)))
        if not profile and query:
            matches = self.store.find_profiles(scope_id, query=query, limit=1)
            if matches:
                profile = matches[0]
                user_id = profile.user_id
                member = self.store.get_member(scope_id, user_id)
                relations = self.store.by_user_id(scope_id, user_id, limit=int(self.config.get("max_query_results", 8)))

        lines = [
            "<group_user_context>",
            f"当前群: {group.name if group else self._session_label(event)}",
            f"群空间ID: {scope_id}",
            f"查询用户ID: {user_id}",
        ]
        if member:
            lines.extend(
                [
                    "群成员目录:",
                    f"- 昵称/群名片: {member.display_name or user_id}",
                    f"- 群身份: {member.role}",
                    f"- 目录来源: {member.source}",
                ]
            )
        else:
            lines.append("群成员目录: 当前群目录中未找到该用户。")
        if profile:
            lines.extend(
                [
                    "用户画像:",
                    f"- 显示名: {profile.display_name or user_id}",
                    f"- 群身份: {profile.group_role or 'unknown'}",
                    f"- 身份来源: {profile.role_evidence or '未记录'}",
                    f"- 首选称呼: {profile.preferred_name or '未记录'}",
                    f"- 别名: {', '.join(profile.aliases) if profile.aliases else '无'}",
                    f"- 触达次数: {profile.message_count}",
                ]
            )
            basic_text = format_basic_profile(profile, max_items=6)
            if basic_text:
                lines.append(f"- 基础画像: {basic_text}")
            if profile.facts:
                lines.append("- 画像事实:")
                for item in profile.facts[: self._cfg_int("记忆管理_当前发言人画像条数", "person_profile_max_items", 4)]:
                    fact = str(item.get("fact") or "").strip()
                    if fact:
                        category = str(item.get("category") or "impression")
                        lines.append(f"  - [{category}] {fact}")
        else:
            lines.append("用户画像: 暂无。")
        if relations:
            lines.append("当前群关系:")
            lines.extend(f"- {format_record(record)}" for record in relations)
        else:
            lines.append("当前群关系: 暂无。")

        if self._cfg_bool("记忆隔离_允许跨群召回", None, True):
            cross_lines = []
            for other_group in self.store.find_user_groups(user_id):
                if other_group.id == scope_id:
                    continue
                other_profile = self.store.get_profile(other_group.id, user_id)
                other_relations = self.store.by_user_id(other_group.id, user_id, limit=4)
                if not other_profile and not other_relations:
                    continue
                cross_lines.append(f"- 来自群 {other_group.name or other_group.id}:")
                if other_profile:
                    cross_lines.append(f"  画像: {format_profile(other_profile, max_facts=3)}")
                for record in other_relations:
                    cross_lines.append(f"  关系: {format_record(record)}")
            if cross_lines:
                lines.append("跨群补充:")
                lines.extend(cross_lines)
        lines.append("</group_user_context>")
        return "\n".join(lines)

    def _group_label_for_memory(self, group_id: str) -> str:
        group = self.store.groups.get(group_id)
        return group.name or group_id if group else group_id

    def _format_tool_record(self, event: AstrMessageEvent, record: RelationRecord) -> str:
        text = format_record(record)
        if record.group_id != self._scope_id(event):
            return f"[来自群:{self._group_label_for_memory(record.group_id)}] {text}"
        return text

    def _format_tool_profile(self, event: AstrMessageEvent, profile) -> str:
        text = format_profile(profile)
        if profile.group_id != self._scope_id(event):
            return f"[来自群:{self._group_label_for_memory(profile.group_id)}] {text}"
        return text

    def _cap_speaker_relations(
        self,
        event: AstrMessageEvent,
        matches: list[tuple[RelationRecord, float]],
    ) -> list[tuple[RelationRecord, float]]:
        cap = self._cfg_int("记忆管理_每人关系召回上限", None, 8)
        if cap <= 0:
            return matches
        speaker_needles = {
            needle
            for needle in (_sender_id(event), _sender_name(event))
            if needle
        }
        speaker_related_count = 0
        capped = []
        for record, score in matches:
            text = f"{record.subject} {record.object}"
            is_speaker_related = any(needle in text for needle in speaker_needles)
            if is_speaker_related:
                speaker_related_count += 1
                if speaker_related_count > cap:
                    continue
            capped.append((record, score))
        return capped

    def _scope_id(self, event: AstrMessageEvent) -> str:
        if _is_group_event(event):
            return _group_id(event) or event.unified_msg_origin
        return event.unified_msg_origin

    def _bot_aliases(self) -> list[str]:
        aliases = _split_config_list(self.config.get("bot_relation_aliases", ""))
        if not aliases:
            aliases = {"机器人", "AI", "助手", "本机器人", "AstrBot"}
        return sorted(aliases)

    def _session_label(self, event: AstrMessageEvent) -> str:
        group_name = _group_name(event)
        group_id = _group_id(event)
        if _is_group_event(event):
            if group_name:
                return f"群聊「{group_name}」({group_id})"
            return f"群聊({group_id})"
        sender = _sender_name(event)
        return f"私聊/单人会话「{sender}」({event.unified_msg_origin})"

    def _build_injection_query(self, event: AstrMessageEvent, req: ProviderRequest | None) -> str:
        group_name = _group_name(event)
        sender_name = _sender_name(event)
        prompt = getattr(req, "prompt", None) or ""
        message = event.message_str or prompt
        return "\n".join(
            part
            for part in [
                f"群聊名称: {group_name}" if group_name else "当前不是群聊，而是私聊/单人会话",
                f"会话记忆范围ID: {self._scope_id(event)}",
                f"平台会话ID: {event.unified_msg_origin}",
                f"群号: {_group_id(event)}" if _group_id(event) else "",
                f"发言人: {sender_name}",
                f"发言人ID: {event.get_sender_id()}",
                f"机器人关系称呼: {', '.join(self._bot_aliases())}",
                f"消息: {message}",
            ]
            if part
        )

    def _build_injection_text(
        self,
        event: AstrMessageEvent,
        matches: list[tuple[RelationRecord, float]],
    ) -> str:
        sender_name = _sender_name(event)
        sender_id = _sender_id(event)
        scope_id = self._scope_id(event)
        session_label = self._session_label(event)
        group = self.store.groups.get(scope_id)
        profile = self.store.get_profile(scope_id, sender_id)
        related_profiles = self._find_related_profiles(scope_id, sender_id, matches)
        lines = [
            "<group_relation_context>",
            "以下是当前群空间的临时记忆上下文，仅供本轮回答参考。",
            "请用这些信息理解当前在哪个群、正在和哪个群友聊天、相关人物是谁，以及他们和当前发言人的关系。",
            "不要把未列出的关系、身份、偏好或隐私当作事实。",
        ]
        if bool(self.config.get("enable_session_identity_injection", True)):
            lines.extend(
                [
                    f"当前会话: {session_label}",
                    f"当前群空间ID: {scope_id}",
                    f"当前发言人: {sender_name}({sender_id})",
                    f"机器人在关系记忆中的可能称呼: {', '.join(self._bot_aliases())}",
                    "回答时优先把当前发言人理解为本轮对话对象，关系和画像都限定在当前群空间内。",
                ]
            )
        if group:
            owner = group.owner_display_name or group.owner_user_id or "未知"
            lines.extend(
                [
                    "",
                    "群空间状态:",
                    f"- 类型: {group.kind}",
                    f"- 群主: {owner}",
                    f"- 群主来源: {group.owner_evidence or '未记录'}",
                    f"- 群成员目录: {group.member_count} 人，来源 {group.member_directory_source or '未初始化'}",
                    f"- 已触达消息数: {group.message_count}",
                    f"- 已记录关系数: {len(self.store.export_group(scope_id))}",
                    f"- 已记录用户画像数: {len(self.store.export_profiles(scope_id))}",
                ]
            )
        if bool(self.config.get("enable_person_profile", True)):
            profile_text = self._build_person_profile(sender_name, sender_id, profile, matches)
            if profile_text:
                lines.extend(["", "当前发言人画像:", profile_text])
        if related_profiles:
            lines.extend(["", "相关人物画像:"])
            lines.extend(f"- {format_profile(item, max_facts=3)}" for item in related_profiles)
        if matches:
            lines.extend(["", "相关关系:"])
            for record, score in matches:
                lines.append(f"- {format_record(record)}  score={score:.2f}")
        lines.extend(
            [
                "",
                "如果用户询问群友身份、画像或人物关系但以上信息不足，可以主动调用 group_user_context_lookup；知道用户ID时必须传 user_id。",
                "只有当用户明确提供稳定环境事实、人物关系或明确纠正时，才在开关允许时写入/修改/删除。",
                "昵称、爱好、讨厌的东西和稳定特征优先使用 group_user_basic_profile_update；普通画像事实使用 group_user_profile_save；关系写入/修改/删除使用 group_relation_save / group_relation_update / group_relation_delete。",
                "</group_relation_context>",
            ]
        )
        return "\n".join(lines)

    def _summary_persona_label(self) -> str:
        persona_id = str(self.config.get("自动总结_人格选择", "") or "").strip()
        return persona_id or "未选择"

    def _summary_persona_prompt(self) -> str:
        persona_id = str(self.config.get("自动总结_人格选择", "") or "").strip()
        if not persona_id:
            return ""
        persona_manager = getattr(self.context, "persona_manager", None)
        get_persona = getattr(persona_manager, "get_persona", None)
        if not callable(get_persona):
            logger.warning("group relation summary persona skipped: persona_manager is unavailable.")
            return ""
        try:
            persona = get_persona(persona_id)
        except Exception as exc:
            logger.warning(f"group relation summary persona `{persona_id}` cannot be loaded: {exc}")
            return ""
        system_prompt = str(getattr(persona, "system_prompt", "") or "").strip()
        if not system_prompt:
            logger.warning(f"group relation summary persona `{persona_id}` has empty system_prompt.")
        return system_prompt

    def _find_related_profiles(
        self,
        scope_id: str,
        sender_id: str,
        matches: list[tuple[RelationRecord, float]],
    ):
        limit = self._cfg_int("记忆管理_相关人物画像注入人数", "related_profile_max_items", 4)
        related = []
        seen = {sender_id}
        for record, _score in matches:
            for key in (record.subject_user_id, record.object_user_id, record.subject, record.object):
                for profile in self.store.find_profiles(scope_id, query=key, limit=2):
                    if profile.user_id in seen or profile.id in seen:
                        continue
                    related.append(profile)
                    seen.add(profile.user_id)
                    seen.add(profile.id)
                    if len(related) >= limit:
                        return related
        return related

    def _build_person_profile(
        self,
        person: str,
        user_id: str,
        profile,
        matches: list[tuple[RelationRecord, float]],
    ) -> str:
        limit = self._cfg_int("记忆管理_当前发言人画像条数", "person_profile_max_items", 4)
        person_norm = person.strip().lower()
        facts = []
        if profile:
            if profile.group_role:
                facts.append(f"群身份: {profile.group_role}")
            basic_text = format_basic_profile(profile, max_items=4)
            if basic_text:
                facts.append(basic_text)
            facts.extend(
                str(item.get("fact") or "").strip()
                for item in profile.facts[:limit]
                if str(item.get("fact") or "").strip()
            )
            if profile.preferred_name:
                facts.insert(0, f"首选称呼: {profile.preferred_name}")
            if profile.aliases:
                facts.insert(0, f"常用称呼: {', '.join(profile.aliases[:4])}")
        for record, _score in matches:
            if len(facts) >= limit:
                break
            record_subject = record.subject.lower()
            record_object = record.object.lower()
            record_subject_user_id = record.subject_user_id.lower()
            record_object_user_id = record.object_user_id.lower()
            if (person_norm and (
                person_norm in record.subject.lower() or person_norm in record.object.lower()
            )) or (user_id and (
                user_id in record_subject
                or user_id in record_object
                or user_id in record_subject_user_id
                or user_id in record_object_user_id
            )):
                facts.append(format_record(record, with_id=False))
        return "；".join(facts)

    async def _remember_memory_item(
        self,
        event: AstrMessageEvent,
        item: dict,
        source: str,
        default_confidence: float,
    ) -> None:
        item_type = str(item.get("type") or "relation").strip().lower()
        confidence = float(item.get("confidence", default_confidence))
        importance = float(item.get("importance", 0.6))
        threshold = self._cfg_float("记忆管理_写入可信度阈值", None, 0.65)
        if confidence < threshold:
            return
        if item_type in {"profile_basic", "basic_profile", "profile_field"}:
            user_id = str(item.get("user_id") or _sender_id(event)).strip()
            display_name = str(item.get("display_name") or item.get("subject") or _sender_name(event)).strip()
            field = str(item.get("field") or "").strip()
            value = str(item.get("value") or "").strip()
            old_value = str(item.get("old_value") or "").strip()
            key = str(item.get("key") or "").strip()
            operation = str(item.get("operation") or "upsert").strip().lower()
            if not field:
                return
            ok, reason = await self._validate_group_user_for_write(event, user_id, display_name)
            if not ok:
                logger.info(f"group relation skipped basic profile memory: {reason}")
                return
            profile = self.store.get_profile(self._scope_id(event), user_id)
            if operation in {"delete", "remove", "replace"} and profile:
                delete_value = old_value if operation == "replace" and old_value else value
                if delete_value:
                    self.store.delete_profile_basic(profile.id, self._scope_id(event), field, value=delete_value, key=key)
            if operation in {"delete", "remove"}:
                return
            if not value:
                return
            self.store.upsert_profile_basic(
                group_id=self._scope_id(event),
                user_id=user_id,
                display_name=display_name,
                field=field,
                key=key,
                value=value,
                note=str(item.get("note") or ""),
                source=source,
                confidence=confidence,
                importance=importance,
            )
            return
        if item_type == "profile":
            user_id = str(item.get("user_id") or _sender_id(event)).strip()
            display_name = str(item.get("display_name") or item.get("subject") or _sender_name(event)).strip()
            fact = str(item.get("fact") or "").strip()
            category = str(item.get("category") or "impression").strip()
            if not fact:
                relation = str(item.get("relation") or "").strip()
                object_ = str(item.get("object") or "").strip()
                fact = " ".join(part for part in [relation, object_] if part).strip()
            if not fact:
                return
            ok, reason = await self._validate_group_user_for_write(event, user_id, display_name)
            if not ok:
                logger.info(f"group relation skipped profile memory: {reason}")
                return
            if not self._is_profile_fact_worth_storing(
                fact,
                str(item.get("note") or ""),
                confidence,
                importance,
                source,
            ):
                return
            profile = self.store.remember_profile_fact(
                group_id=self._scope_id(event),
                user_id=user_id,
                display_name=display_name,
                fact=fact,
                note=str(item.get("note") or ""),
                source=source,
                confidence=confidence,
                category=category,
                importance=importance,
            )
            self._trim_profile_facts(profile)
            return
        subject_user_id = str(item.get("subject_user_id") or "").strip()
        object_user_id = str(item.get("object_user_id") or "").strip()
        ok, reason = await self._validate_relation_users_for_write(
            event,
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
        )
        if not ok:
            logger.info(f"group relation skipped relation memory: {reason}")
            return
        if not self._is_relation_worth_storing(
            str(item.get("subject") or ""),
            str(item.get("relation") or ""),
            str(item.get("object") or ""),
            str(item.get("note") or ""),
            confidence,
            importance,
            source,
        ):
            return
        await self._remember_relation(
            group_id=self._scope_id(event),
            subject=item["subject"],
            relation=item["relation"],
            object_=item["object"],
            subject_user_id=subject_user_id,
            object_user_id=object_user_id,
            category=str(item.get("category") or "relation"),
            note=item.get("note", ""),
            source=source,
            confidence=confidence,
            importance=importance,
        )

    def _trim_profile_facts(self, profile) -> None:
        limit = max(1, self._cfg_int("记忆管理_每人记忆上限", None, 20))
        if len(profile.facts) <= limit:
            return
        profile.facts = profile.facts[:limit]
        self.store.save()

    def _extract_prompt(self, event: AstrMessageEvent, message: str) -> str:
        return f"""
你是群关系记忆抽取器。请从单条群聊消息中抽取“明确、稳定、之后聊天有用”的记忆。
只输出 JSON 数组，不要输出解释、Markdown 或多余文本。

当前群空间：{self._session_label(event)}
当前发言人：{_sender_name(event)}({_sender_id(event)})

可输出三类元素。

1. 关系记忆：用于描述群成员之间的稳定关系，至少要有一个群成员 user_id。
{{"type":"relation","subject":"人物A","subject_user_id":"群成员A的ID或空","relation":"关系","object":"人物B","object_user_id":"群成员B的ID或空","category":"relationship/role/preference/memory/correction","note":"证据摘要","confidence":0.0到1.0,"importance":0.0到1.0}}

2. 基础画像：用于维护昵称/首选称呼/爱好/讨厌的东西/稳定特征/长期备注。优先使用此类型，不要把这些内容写成普通画像事实。
{{"type":"profile_basic","operation":"upsert/delete/replace","user_id":"{_sender_id(event)}","display_name":"{_sender_name(event)}","field":"preferred_name/aliases/likes/dislikes/traits/notes","key":"可选分类键，如 game/food/nickname","value":"新值或要删除的值","old_value":"replace 时的旧值或空","note":"证据摘要","confidence":0.0到1.0,"importance":0.0到1.0}}

3. 普通用户画像：只用于不适合归入基础画像、但长期有用的稳定事实。
{{"type":"profile","user_id":"{_sender_id(event)}","display_name":"{_sender_name(event)}","category":"identity/preference/impression/memory/correction","fact":"稳定画像事实","note":"证据摘要","confidence":0.0到1.0,"importance":0.0到1.0}}

抽取规则：
1. 只能抽取消息字面明确表达、当前发言人自述、或被明确确认的事实。
2. 对“我/本人/咱”这类指代，若指当前发言人，画像必须使用当前发言人的 user_id 和 display_name。
3. 关系若涉及当前发言人，必须填写当前发言人的 user_id；不能确定群成员 ID 时不要输出该关系。
4. 昵称/首选称呼输出 field=preferred_name 或 aliases；爱好输出 likes；讨厌/雷点输出 dislikes；稳定性格/长期身份特征输出 traits。
5. 用户明确否定、纠正或要求别记时，才输出 delete/replace；普通新增用 upsert。
6. 只保留长期有用信息：群身份、稳定关系、明确偏好、长期习惯、用户要求记住的内容、纠错后的事实。
7. 不记录一次性闲聊、玩笑、辱骂、情绪评价、当天行程、临时状态、敏感隐私、猜测、反问或未确认传闻。
8. importance 表示之后水群时是否经常有用；小事低于 0.55，低于 0.55 的内容不要输出。
9. 不能确定时输出 []。

消息：
{message}
""".strip()

    def _summary_extract_prompt(self, event: AstrMessageEvent, dialogue: str) -> str:
        personality_prompt = self._summary_persona_prompt()
        personality_block = ""
        if personality_prompt:
            personality_block = f"""
已选择的 AstrBot 人格 system_prompt：
{personality_prompt}

该人格来自 AstrBot 已配置人格，仅用于帮助你理解机器人在群内的称呼、说话风格、角色边界和上下文语气。
不要把人格内容本身当作用户事实、人物关系或用户画像写入记忆。
""".strip()
        return f"""
你是 AstrBot 的群关系与用户画像记忆整理器。请从最近几轮对话中抽取“明确、稳定、之后聊天有用”的记忆。
只输出 JSON 数组，不要输出解释、Markdown 或多余文本。

当前会话：{self._session_label(event)}
当前最后发言人：{_sender_name(event)}({_sender_id(event)})

{personality_block}

可输出三类元素。

1. 关系记忆：描述群成员之间或群成员与组织/群体之间的稳定关系，至少要有一个群成员 user_id。
{{"type":"relation","subject":"人物A","subject_user_id":"群成员A的ID或空","relation":"关系","object":"人物B/群/组织","object_user_id":"群成员B的ID或空","category":"relationship/role/preference/memory/correction","note":"证据摘要","confidence":0.0到1.0,"importance":0.0到1.0}}

2. 基础画像：维护昵称/首选称呼/爱好/讨厌的东西/稳定特征/长期备注。优先使用此类型合并同类信息，不要把这些内容拆成多条普通画像事实。
{{"type":"profile_basic","operation":"upsert/delete/replace","user_id":"用户ID","display_name":"昵称或群名片","field":"preferred_name/aliases/likes/dislikes/traits/notes","key":"可选分类键，如 game/food/nickname","value":"新值或要删除的值","old_value":"replace 时的旧值或空","note":"证据摘要","confidence":0.0到1.0,"importance":0.0到1.0}}

3. 普通用户画像：描述无法归入基础画像、但长期有用的稳定事实。
{{"type":"profile","user_id":"用户ID","display_name":"昵称或群名片","category":"identity/preference/impression/memory/correction","fact":"稳定画像事实","note":"证据摘要","confidence":0.0到1.0,"importance":0.0到1.0}}

抽取规则：
1. 只抽取对话明确表达、用户自述、管理员确认或用户纠正后的事实。
2. 用户纠正机器人时，以用户纠正为准；可以输出新的正确事实，但不要保留被纠正的旧说法。
3. 对“我/本人/咱/楼上/他”等指代，只有能从对话行里的昵称和ID确定对象时才抽取。
4. 用户画像必须填写 user_id；无法确定 user_id 时不要输出 profile。
5. 关系记忆至少填写 subject_user_id 或 object_user_id；无法确认任一群成员 ID 时不要输出 relation。
6. 昵称/首选称呼输出 field=preferred_name 或 aliases；爱好输出 likes；讨厌/雷点输出 dislikes；稳定性格/长期身份特征输出 traits。
7. 用户明确否定、纠正或要求别记时，才输出 delete/replace；普通新增用 upsert。
8. 只保留长期有用信息：群身份、稳定关系、明确偏好、长期习惯、用户要求记住的内容、纠错后的事实。
9. 不记录一次性闲聊、玩笑、攻击性评价、当天行程、临时状态、敏感隐私、推测、谣言、未确认关系。
10. importance 表示之后水群时是否经常有用；小事低于 0.55，低于 0.55 的内容不要输出。
11. 没有值得记忆的内容时输出 []。

最近对话：
{dialogue}
""".strip()

    def _parse_extracted_memories(self, text: str) -> list[dict]:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        memories = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or item.get("kind") or "relation").strip().lower()
            note = str(item.get("note") or "").strip()
            try:
                confidence = float(item.get("confidence", 0.65))
            except (TypeError, ValueError):
                confidence = 0.65
            try:
                importance = float(item.get("importance", 0.6))
            except (TypeError, ValueError):
                importance = 0.6
            if item_type in {"profile_basic", "basic_profile", "profile_field"}:
                field = str(item.get("field") or "").strip()[:40]
                value = str(item.get("value") or "").strip()[:120]
                operation = str(item.get("operation") or "upsert").strip().lower()[:20]
                if not field or (operation not in {"delete", "remove"} and not value):
                    continue
                memories.append(
                    {
                        "type": "profile_basic",
                        "operation": operation,
                        "user_id": str(item.get("user_id") or "").strip()[:80],
                        "display_name": str(item.get("display_name") or item.get("subject") or "").strip()[:80],
                        "field": field,
                        "key": str(item.get("key") or "").strip()[:40],
                        "value": value,
                        "old_value": str(item.get("old_value") or "").strip()[:120],
                        "note": note[:240],
                        "confidence": max(0.0, min(1.0, confidence)),
                        "importance": max(0.0, min(1.0, importance)),
                    }
                )
                continue
            if item_type == "profile":
                fact = str(item.get("fact") or "").strip()
                if not fact:
                    relation = str(item.get("relation") or "").strip()
                    object_ = str(item.get("object") or "").strip()
                    fact = " ".join(part for part in [relation, object_] if part).strip()
                if not fact:
                    continue
                memories.append(
                    {
                        "type": "profile",
                        "user_id": str(item.get("user_id") or "").strip()[:80],
                        "display_name": str(item.get("display_name") or item.get("subject") or "").strip()[:80],
                        "category": str(item.get("category") or "impression").strip()[:40],
                        "fact": fact[:160],
                        "note": note[:240],
                        "confidence": max(0.0, min(1.0, confidence)),
                        "importance": max(0.0, min(1.0, importance)),
                    }
                )
                continue
            subject = str(item.get("subject") or "").strip()
            relation = str(item.get("relation") or "").strip()
            object_ = str(item.get("object") or "").strip()
            subject_user_id = str(item.get("subject_user_id") or "").strip()
            object_user_id = str(item.get("object_user_id") or "").strip()
            if not subject or not relation or not object_:
                continue
            memories.append(
                {
                    "type": "relation",
                    "subject": subject[:80],
                    "subject_user_id": subject_user_id[:80],
                    "relation": relation[:80],
                    "object": object_[:80],
                    "object_user_id": object_user_id[:80],
                    "category": str(item.get("category") or "relation").strip()[:40],
                    "note": note[:240],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "importance": max(0.0, min(1.0, importance)),
                }
            )
        return memories

    async def terminate(self):
        self.store.save()
