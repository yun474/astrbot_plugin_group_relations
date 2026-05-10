from __future__ import annotations

import asyncio
import json
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.provider import EmbeddingProvider, Provider
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .relation_store import (
    RelationRecord,
    RelationStore,
    embed_text,
    format_record,
    normalize_vector,
)


PLUGIN_NAME = "astrbot_plugin_group_relations"


def _group_id(event: AstrMessageEvent) -> str:
    message_obj = getattr(event, "message_obj", None)
    group_id = getattr(message_obj, "group_id", "") if message_obj else ""
    return str(group_id or getattr(message_obj, "session_id", "") or event.unified_msg_origin)


def _group_name(event: AstrMessageEvent) -> str:
    group = getattr(getattr(event, "message_obj", None), "group", None)
    return str(getattr(group, "group_name", "") or "")


def _sender_name(event: AstrMessageEvent) -> str:
    return str(event.get_sender_name() or event.get_sender_id() or "未知成员")


def _split_config_list(value) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str):
        return {item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()}
    return set()


class GroupRelationsPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or getattr(self, "config", {}) or {}
        data_dir = Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        self.store = RelationStore(data_dir)
        self.store.load()
        self._reembed_started = False
        self._warned_embedding_provider_ids: set[str] = set()
        self._warned_no_embedding_provider = False
        self._warned_local_embedding_fallback = False
        self._warned_summary_provider_ids: set[str] = set()
        self._warned_no_summary_provider = False
        self._provider_options_task: asyncio.Task | None = None
        self._summary_buffers: dict[str, list[dict[str, str]]] = {}
        self._refresh_embedding_provider_options()
        try:
            self._provider_options_task = asyncio.create_task(
                self._refresh_embedding_provider_options_later()
            )
        except RuntimeError:
            self._provider_options_task = None

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
                    "/关系 调试 <查询词>",
                    "/关系 最近",
                    "/关系 向量",
                ]
            )
        )

    @relations.command("状态")
    async def status(self, event: AstrMessageEvent):
        """查看插件状态。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        provider = self._get_embedding_provider(silent=True)
        yield event.plain_result(
            "\n".join(
                [
                    f"自动注入：{bool(self.config.get('enable_context_injection', True))}",
                    f"自动总结：{bool(self.config.get('enable_dialogue_summary', False))}",
                    f"总结轮数：{int(self.config.get('summary_trigger_rounds', 6))}",
                    f"注入条数：{int(self.config.get('injection_top_k', 5))}",
                    f"人物画像：{bool(self.config.get('enable_person_profile', True))}",
                    f"工具读取：{bool(self.config.get('enable_tool_read', True))}",
                    f"工具写入：{bool(self.config.get('enable_tool_write', False))}",
                    f"工具修改/删除：{bool(self.config.get('enable_tool_update', False))}",
                    f"记忆隔离：{self.config.get('memory_scope', 'session')}",
                    f"Embedding Provider：{provider.meta().id if provider else '不可用'}",
                    f"向量本地回退：{bool(self.config.get('enable_local_embedding_fallback', False))}",
                    f"总结 Provider：{self.config.get('summary_provider_id', '') or '当前会话模型'}",
                    f"当前会话关系数：{len(self.store.export_group(self._scope_id(event)))}",
                ]
            )
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

    @relations.command("向量")
    async def vector_status(self, event: AstrMessageEvent):
        """查看向量化配置。"""
        if not self._can_use_debug_commands(event):
            yield event.plain_result(self._debug_denied_message())
            return
        self._refresh_embedding_provider_options()
        await self._maybe_reembed_records()
        provider_id = str(self.config.get("embedding_provider_id", "")).strip()
        all_providers = self._list_embedding_providers()
        if not provider_id:
            provider = self._get_embedding_provider(silent=True)
            if provider:
                yield event.plain_result(
                    f"当前未指定 Provider，自动使用：{provider.meta().id}，维度：{provider.get_dim()}"
                )
            else:
                fallback = "开启" if bool(self.config.get("enable_local_embedding_fallback", False)) else "关闭"
                yield event.plain_result(
                    "当前没有可用的 AstrBot Embedding Provider；"
                    f"本地哈希回退：{fallback}。\n"
                    "请在 AstrBot 服务提供商中配置 Embedding Provider，或临时开启本地回退。"
                )
            return
        provider = self._get_embedding_provider()
        if not provider:
            available = ", ".join(provider.meta().id for provider in all_providers) or "无"
            yield event.plain_result(
                f"未找到可用的 Embedding Provider：{provider_id}\n当前可选：{available}"
            )
            return
        yield event.plain_result(
            f"当前使用 AstrBot Embedding Provider：{provider_id}，维度：{provider.get_dim()}"
        )

    @filter.on_llm_request()
    async def inject_group_relations(self, event: AstrMessageEvent, req: ProviderRequest):
        """在每次 LLM 请求前注入少量当前群关系上下文。"""
        if event.get_extra("group_relation_skip_injection", False):
            return
        if not bool(self.config.get("enable_context_injection", True)):
            return
        query = self._build_injection_query(event, req)
        top_k = max(0, int(self.config.get("injection_top_k", 5)))
        if top_k <= 0:
            return
        matches = await self._search_relations(event, query, limit=top_k)
        if not matches:
            return
        text = self._build_injection_text(event, matches)
        max_length = int(self.config.get("max_injected_text_length", 1200))
        if max_length > 0 and len(text) > max_length:
            text = text[:max_length].rstrip() + "\n[已截断]"
        req.extra_user_content_parts.append(TextPart(text=text).mark_as_temp())

    @filter.on_llm_response()
    async def summarize_dialogue_relations(self, event: AstrMessageEvent, resp: LLMResponse):
        """每隔几轮对话，总结抽取当前会话的人物关系。"""
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
        buffer.append({"role": "user", "content": user_text})
        buffer.append({"role": "assistant", "content": assistant_text})
        trigger_rounds = max(1, int(self.config.get("summary_trigger_rounds", 6)))
        if len(buffer) < trigger_rounds * 2:
            return
        dialogue = self._format_summary_dialogue(buffer)
        max_chars = int(self.config.get("summary_max_dialogue_chars", 4000))
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
            relations = self._parse_extracted_relations(summary_resp.completion_text)
        except Exception as exc:
            logger.warning(f"group relation dialogue summary failed: {exc}")
            return
        finally:
            event.set_extra("group_relation_skip_injection", False)
            event.set_extra("group_relation_skip_summary", False)
        for item in relations:
            await self._remember_relation(
                group_id=scope_id,
                subject=item["subject"],
                relation=item["relation"],
                object_=item["object"],
                note=item.get("note", ""),
                source="summary",
                confidence=float(item.get("confidence", 0.65)),
            )
        self._summary_buffers[scope_id] = []

    @filter.llm_tool(name="group_relation_search")
    async def group_relation_search(self, event: AstrMessageEvent, query: str):
        """查询当前群的人物关系记忆。

        Args:
            query(string): 要查询的人名、昵称、关系、事件或自然语言问题。
        """
        if not bool(self.config.get("enable_tool_read", True)):
            yield event.plain_result("群关系记忆查询工具当前未启用。")
            return
        matches = await self._search_relations(event, query)
        if not matches:
            yield event.plain_result(f"没有找到和「{query}」相关的群关系记忆。")
            return
        lines = [format_record(record) for record, _score in matches]
        yield event.plain_result("\n".join(lines))

    @filter.llm_tool(name="group_relation_remember")
    async def group_relation_remember(
        self,
        event: AstrMessageEvent,
        subject: str,
        relation: str,
        object_: str,
        note: str = "",
        confidence: float = 0.8,
    ):
        """写入当前群的一条明确人物关系记忆。

        Args:
            subject(string): 关系主体，例如人名、昵称、群成员、组织。
            relation(string): 关系类型，例如朋友、同学、管理员、常一起玩、合作、敌对。
            object_(string): 关系客体，例如另一个人名、昵称、群成员、组织。
            note(string): 补充说明或证据摘要，没有可以留空。
            confidence(number): 关系可信度，0 到 1 之间。
        """
        if not bool(self.config.get("enable_tool_write", False)):
            yield event.plain_result("群关系记忆写入工具当前未启用。")
            return
        confidence = max(0.0, min(1.0, float(confidence)))
        record = await self._remember_relation(
            group_id=self._scope_id(event),
            subject=subject,
            relation=relation,
            object_=object_,
            note=note,
            source="tool",
            confidence=confidence,
        )
        yield event.plain_result(f"已写入群关系记忆：{format_record(record)}")

    @filter.llm_tool(name="group_relation_update")
    async def group_relation_update(
        self,
        event: AstrMessageEvent,
        relation_id: str,
        subject: str = "",
        relation: str = "",
        object_: str = "",
        note: str = "",
        confidence: float = -1.0,
    ):
        """根据用户明确纠正，修改一条当前群关系记忆。

        Args:
            relation_id(string): 要修改的关系 ID。
            subject(string): 新主体，不改则留空。
            relation(string): 新关系，不改则留空。
            object_(string): 新客体，不改则留空。
            note(string): 新补充说明，不改则留空。
            confidence(number): 新可信度，0 到 1；不改传 -1。
        """
        if not bool(self.config.get("enable_tool_update", False)):
            yield event.plain_result("群关系记忆修改工具当前未启用。")
            return
        old = self.store.records.get(relation_id)
        if not old or old.group_id != self._scope_id(event):
            yield event.plain_result("没有找到这条关系，或它不属于当前群。")
            return
        new_subject = subject or old.subject
        new_relation = relation or old.relation
        new_object = object_ or old.object
        new_note = note if note else old.note
        text = " ".join(part for part in [new_subject, new_relation, new_object, new_note] if part)
        vector, provider_id = await self._embed(text)
        updated = self.store.update(
            relation_id,
            group_id=self._scope_id(event),
            subject=subject or None,
            relation=relation or None,
            object_=object_ or None,
            note=note if note else None,
            confidence=None if confidence < 0 else confidence,
            vector=vector,
            embedding_provider_id=provider_id,
        )
        yield event.plain_result(f"已更新：{format_record(updated)}" if updated else "更新失败。")

    @filter.llm_tool(name="group_relation_delete")
    async def group_relation_delete(self, event: AstrMessageEvent, relation_id: str, reason: str = ""):
        """根据用户明确纠正，删除一条当前群关系记忆。

        Args:
            relation_id(string): 要删除的关系 ID。
            reason(string): 删除原因摘要。
        """
        if not bool(self.config.get("enable_tool_update", False)):
            yield event.plain_result("群关系记忆删除工具当前未启用。")
            return
        ok = self.store.delete(relation_id, group_id=self._scope_id(event))
        yield event.plain_result("已删除。" if ok else "没有找到这条关系，或它不属于当前群。")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def auto_extract(self, event: AstrMessageEvent):
        """从群聊中自动抽取关系，默认关闭。"""
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
                prompt=self._extract_prompt(message),
            )
            relations = self._parse_extracted_relations(response.completion_text)
        except Exception as exc:
            logger.warning(f"group relation auto extraction failed: {exc}")
            return
        finally:
            event.set_extra("group_relation_skip_injection", False)
        for item in relations:
            await self._remember_relation(
                group_id=self._scope_id(event),
                subject=item["subject"],
                relation=item["relation"],
                object_=item["object"],
                note=item.get("note", ""),
                source="auto",
                confidence=float(item.get("confidence", 0.6)),
            )

    async def _remember_relation(
        self,
        group_id: str,
        subject: str,
        relation: str,
        object_: str,
        note: str = "",
        source: str = "manual",
        confidence: float = 1.0,
    ) -> RelationRecord:
        await self._maybe_reembed_records()
        text = " ".join(part for part in [subject, relation, object_, note] if part)
        vector, provider_id = await self._embed(text)
        return self.store.upsert(
            group_id=group_id,
            subject=subject,
            relation=relation,
            object_=object_,
            note=note,
            source=source,
            confidence=confidence,
            vector=vector,
            embedding_provider_id=provider_id,
        )

    def _debug_denied_message(self) -> str:
        return "这个调试指令会暴露群关系记忆，当前只允许配置里的管理员使用。"

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

    async def _get_summary_provider_id(self, event: AstrMessageEvent) -> str:
        provider_id = str(self.config.get("summary_provider_id", "")).strip()
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
            role = "用户" if item["role"] == "user" else "机器人"
            content = item["content"].strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _search_relations(
        self,
        event: AstrMessageEvent,
        query: str,
        limit: int | None = None,
    ) -> list[tuple[RelationRecord, float]]:
        await self._maybe_reembed_records()
        limit = limit or int(self.config.get("max_query_results", 8))
        threshold = float(self.config.get("similarity_threshold", 0.12))
        vector, _provider_id = await self._embed(query)
        if not vector:
            return self.store.search_by_text(self._scope_id(event), query, limit=limit)
        return self.store.search_by_vector(
            self._scope_id(event),
            query,
            vector,
            limit=limit,
            threshold=threshold,
        )

    def _scope_id(self, event: AstrMessageEvent) -> str:
        scope = str(self.config.get("memory_scope", "session")).strip().lower()
        if scope == "global":
            return "global"
        if scope == "group":
            return _group_id(event) or event.unified_msg_origin
        return event.unified_msg_origin

    def _session_label(self, event: AstrMessageEvent) -> str:
        group_name = _group_name(event)
        group_id = _group_id(event)
        if group_id:
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
        session_label = self._session_label(event)
        lines = [
            "<group_relation_context>",
            "以下是当前群关系记忆的检索结果，仅供本轮回答参考；不要编造未列出的关系。",
        ]
        if bool(self.config.get("enable_session_identity_injection", True)):
            lines.extend(
                [
                    f"当前会话: {session_label}",
                    f"当前发言人: {sender_name}",
                    "你正在这个会话里聊天；回答时要把这些关系理解为这个会话内的人际背景。",
                ]
            )
        if bool(self.config.get("enable_person_profile", True)):
            profile = self._build_person_profile(sender_name, matches)
            if profile:
                lines.extend(["", "当前发言人简易画像:", profile])
        lines.extend(["", "相关关系:"])
        for record, score in matches:
            lines.append(f"- {format_record(record)}  score={score:.2f}")
        lines.extend(
            [
                "",
                "如果用户询问人物关系但以上信息不足，可以主动调用 group_relation_search。",
                "如果用户明确纠正关系，可在开关允许时调用 group_relation_remember / group_relation_update / group_relation_delete。",
                "</group_relation_context>",
            ]
        )
        return "\n".join(lines)

    def _build_person_profile(
        self,
        person: str,
        matches: list[tuple[RelationRecord, float]],
    ) -> str:
        limit = int(self.config.get("person_profile_max_items", 4))
        person_norm = person.strip().lower()
        facts = []
        for record, _score in matches:
            if person_norm and (
                person_norm in record.subject.lower() or person_norm in record.object.lower()
            ):
                facts.append(format_record(record, with_id=False))
            if len(facts) >= limit:
                break
        return "；".join(facts)

    async def _embed(self, text: str) -> tuple[list[float] | None, str]:
        max_length = int(self.config.get("max_embedding_text_length", 2000))
        if max_length > 0 and len(text) > max_length:
            logger.warning(
                f"group relation embedding text too long ({len(text)} chars), "
                f"truncated to {max_length}"
            )
            text = text[:max_length]
        provider = self._get_embedding_provider()
        if provider:
            try:
                return normalize_vector(await provider.get_embedding(text)), provider.meta().id
            except Exception as exc:
                logger.warning(
                    f"group relation AstrBot embedding failed with provider `{provider.meta().id}`: {exc}"
                )
        if bool(self.config.get("enable_local_embedding_fallback", False)):
            if not self._warned_local_embedding_fallback:
                logger.warning(
                    "group relation falling back to local hash embedding because AstrBot embedding is unavailable."
                )
                self._warned_local_embedding_fallback = True
            return embed_text(text), "local_hash"
        if not self._warned_no_embedding_provider:
            logger.warning(
                "group relation embedding unavailable: no valid AstrBot Embedding Provider resolved "
                "and local fallback is disabled. Configure embedding_provider_id or add an Embedding Provider."
            )
            self._warned_no_embedding_provider = True
        return None, ""

    def _get_embedding_provider(self, silent: bool = False) -> EmbeddingProvider | None:
        provider_id = str(self.config.get("embedding_provider_id", "")).strip()
        if not provider_id:
            providers = self._list_embedding_providers()
            if providers:
                return providers[0]
            if not silent and not self._warned_no_embedding_provider:
                logger.warning(
                    "group relation found no configured AstrBot Embedding Provider. "
                    "Vector search/write will wait for a provider, unless local fallback is enabled."
                )
                self._warned_no_embedding_provider = True
            return None
        provider = self.context.get_provider_by_id(provider_id)
        if isinstance(provider, EmbeddingProvider):
            return provider
        if not silent and provider_id not in self._warned_embedding_provider_ids:
            logger.warning(f"group relation embedding provider `{provider_id}` not found or invalid")
            self._warned_embedding_provider_ids.add(provider_id)
        return None

    def _list_embedding_providers(self) -> list[EmbeddingProvider]:
        try:
            return list(self.context.get_all_embedding_providers())
        except Exception as exc:
            logger.warning(f"group relation failed to list embedding providers: {exc}")
            return []

    def _refresh_embedding_provider_options(self) -> None:
        schema = getattr(self.config, "schema", None)
        if not isinstance(schema, dict):
            return
        item = schema.get("embedding_provider_id")
        if not isinstance(item, dict):
            return
        providers = self._list_embedding_providers()
        options = [""]
        labels = ["自动选择第一个可用 Embedding Provider"]
        for provider in providers:
            meta = provider.meta()
            provider_id = meta.id
            if not provider_id or provider_id in options:
                continue
            model = provider.get_model() or provider.provider_config.get("embedding_model", "")
            dim = provider.get_dim()
            label_parts = [provider_id]
            if model:
                label_parts.append(str(model))
            if dim:
                label_parts.append(f"{dim}维")
            options.append(provider_id)
            labels.append(" / ".join(label_parts))
        item["options"] = options
        item["labels"] = labels

    async def _refresh_embedding_provider_options_later(self) -> None:
        for _ in range(30):
            self._refresh_embedding_provider_options()
            if self._list_embedding_providers():
                return
            await asyncio.sleep(2)

    async def _maybe_reembed_records(self) -> None:
        if self._reembed_started:
            return
        self._reembed_started = True
        if not bool(self.config.get("reembed_on_provider_change", False)):
            return
        provider = self._get_embedding_provider()
        if not provider:
            return
        provider_id = provider.meta().id
        records = [
            record
            for record in self.store.records.values()
            if record.embedding_provider_id != provider_id
        ]
        if not records:
            return
        try:
            texts = [record.text() for record in records]
            vectors = await provider.get_embeddings(texts)
        except Exception as exc:
            logger.warning(f"group relation re-embedding failed: {exc}")
            return
        if len(vectors) != len(records):
            logger.warning(
                "group relation re-embedding skipped: vector count mismatch "
                f"expected={len(records)} actual={len(vectors)}"
            )
            return
        for record, vector in zip(records, vectors):
            record.embedding_provider_id = provider_id
            record.embedding_dim = len(vector)
            self.store.vectors[record.id] = normalize_vector(vector)
        self.store.save()

    def _extract_prompt(self, message: str) -> str:
        return f"""
从下面这条群聊消息中抽取明确的人物关系。只输出 JSON 数组，不要输出解释。
每个元素格式：
{{"subject":"人物A","relation":"关系","object":"人物B","note":"补充信息","confidence":0.0到1.0}}

要求：
1. 只抽取消息明确表达的关系，不要猜测。
2. 没有明确关系时输出 []。
3. 人名、昵称、组织名都可以作为 subject 或 object。

消息：
{message}
""".strip()

    def _summary_extract_prompt(self, event: AstrMessageEvent, dialogue: str) -> str:
        return f"""
你正在为 AstrBot 的当前会话整理人物关系记忆。
当前会话：{self._session_label(event)}
当前发言人：{_sender_name(event)}

请从最近几轮对话中抽取明确、稳定、对之后聊天有帮助的人物关系或人物画像事实。
只输出 JSON 数组，不要输出解释。

每个元素格式：
{{"subject":"人物A或当前发言人","relation":"关系或画像属性","object":"人物B/群/兴趣/身份/偏好","note":"证据摘要","confidence":0.0到1.0}}

要求：
1. 只抽取对话明确表达或强烈确认的事实，不要猜测。
2. 用户纠正机器人时，以用户纠正为准。
3. 可以记录人物画像，例如昵称、身份、偏好、常一起玩的对象，但不要记录一次性闲聊废话。
4. 没有值得记忆的内容时输出 []。
5. 不要输出隐私敏感、攻击性或不确定的关系。

最近对话：
{dialogue}
""".strip()

    def _parse_extracted_relations(self, text: str) -> list[dict]:
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
        relations = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or "").strip()
            relation = str(item.get("relation") or "").strip()
            object_ = str(item.get("object") or "").strip()
            if not subject or not relation or not object_:
                continue
            note = str(item.get("note") or "").strip()
            try:
                confidence = float(item.get("confidence", 0.65))
            except (TypeError, ValueError):
                confidence = 0.65
            relations.append(
                {
                    "subject": subject[:80],
                    "relation": relation[:80],
                    "object": object_[:80],
                    "note": note[:240],
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return relations

    async def terminate(self):
        if self._provider_options_task and not self._provider_options_task.done():
            self._provider_options_task.cancel()
        self.store.save()
