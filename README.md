# astrbot_plugin_group_relations

给 AstrBot 使用的群关系上下文插件。它不是靠用户日常输入指令来工作，而是在每次 LLM 请求前，自动根据当前群、当前发言人和本轮消息召回少量人物关系，并作为临时上下文注入给模型。

## 核心逻辑

1. 写入关系时保存结构化三元组：`主体 --关系--> 客体`。
2. 每条关系使用 AstrBot 已配置的 Embedding Provider 向量化。
3. 每次 LLM 请求前，用 `群名/私聊标识 + 会话 ID + 发言人昵称/群名片 + 发言人 ID + 本轮消息` 组成查询文本。
4. 默认按会话隔离记忆，群聊和私聊都会拥有独立关系上下文。
5. 召回到的少量关系、当前会话身份说明和当前发言人的简易画像会通过 `req.extra_user_content_parts` 临时注入，不写入会话历史。
6. LLM 需要更多信息时，可以主动调用工具搜索关系。
7. 用户明确纠正时，LLM 可以在配置允许的情况下写入、修改或删除关系。
8. 可选开启每隔几轮对话自动总结，用指定 LLM Provider 抽取关系和人物画像事实。

## 调试指令

```text
/关系 状态
/关系 调试 <查询词>
/关系 最近
/关系 向量
```

这些指令只用于开关检查和 debug，不是主要交互入口。默认只有配置里的管理员或平台可识别的管理员可以使用，避免把群关系记忆暴露给普通成员。

## LLM 工具

- `group_relation_search`：主动搜索当前群人物关系。
- `group_relation_remember`：写入当前群人物关系，受 `enable_tool_write` 控制。
- `group_relation_update`：修改当前群人物关系，受 `enable_tool_update` 控制。
- `group_relation_delete`：删除当前群人物关系，受 `enable_tool_update` 控制。

## 重要配置

- `enable_context_injection`：是否每轮自动注入关系上下文，默认开启。
- `enable_dialogue_summary`：是否每隔几轮对话自动总结并写入关系，默认关闭。
- `summary_trigger_rounds`：自动总结触发轮数，默认 6。
- `summary_provider_id`：自动总结使用的 LLM Provider，留空使用当前会话模型。
- `memory_scope`：记忆隔离范围，默认 `session`，可选 `session` / `group` / `global`。
- `enable_session_identity_injection`：是否明确注入“当前在哪个群聊/私聊”的会话身份说明，默认开启。
- `relation_admin_user_ids`：允许使用 `/关系` 调试指令的用户 ID，多个 ID 可用英文逗号或换行分隔。
- `allow_public_debug_commands`：是否允许所有人使用 `/关系` 调试指令，默认关闭，只建议测试群临时开启。
- `injection_top_k`：每轮注入关系数量，默认 5。
- `enable_person_profile`：是否生成当前发言人的简易画像，默认开启。
- `embedding_provider_id`：AstrBot Embedding Provider。留空时自动使用第一个可用 Provider。
- `enable_tool_read`：是否允许 LLM 主动搜索关系，默认开启。
- `enable_tool_write`：是否允许 LLM 主动写入关系，默认关闭。
- `enable_tool_update`：是否允许 LLM 主动修改/删除关系，默认关闭。
- `enable_auto_extract`：是否自动从群聊消息抽取关系，默认关闭。

建议先开启自动注入和工具搜索；写入、修改、删除能力等你确认模型行为稳定后再打开。没有 AstrBot Embedding Provider 时，插件会保留轻量本地检索能力，避免关系被写入后完全搜不到；如果要获得更好的召回效果，仍建议配置正式的 Embedding Provider。
