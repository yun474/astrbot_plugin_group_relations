const plugin = window.AstrBotPluginPage;

const state = {
  memory: null,
  groupId: "",
  filter: "",
  collapsedPanels: {
    group: false,
    relations: false,
    members: true,
    profiles: false,
  },
};

const $ = (selector) => document.querySelector(selector);

const basicProfileFields = [
  ["likes", "喜欢 / 爱好"],
  ["dislikes", "讨厌 / 雷点"],
  ["traits", "稳定特征"],
  ["notes", "长期备注"],
];

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    el.hidden = true;
  }, 2600);
}

function field(label, value, name, type = "text") {
  return `
    <label>
      <span>${label}</span>
      <input name="${name}" type="${type}" value="${escapeHtml(value ?? "")}" />
    </label>
  `;
}

function textarea(label, value, name) {
  return `
    <label>
      <span>${label}</span>
      <textarea name="${name}">${escapeHtml(value ?? "")}</textarea>
    </label>
  `;
}

function roleSelect(value) {
  const current = value || "member";
  return `
    <label>
      <span>成员身份</span>
      <select name="role">
        ${["owner", "admin", "member"]
          .map((role) => `<option value="${role}" ${role === current ? "selected" : ""}>${role}</option>`)
          .join("")}
      </select>
    </label>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formData(node) {
  return Object.fromEntries(new FormData(node).entries());
}

function setMemberRefreshState(message = "", busy = false, tone = "") {
  const status = $("#memberRefreshStatus");
  if (status) {
    status.textContent = message;
    status.dataset.tone = tone;
  }
  document.querySelectorAll("[data-action='refresh-members-card'], [data-action='refresh-members-nickname']").forEach((button) => {
    button.disabled = busy;
  });
}

async function apiGet(endpoint, params = {}) {
  const result = await plugin.apiGet(endpoint, params);
  return result.data ?? result;
}

async function apiPost(endpoint, payload = {}) {
  const result = await plugin.apiPost(endpoint, payload);
  const body = result.data ?? result;
  if (body && body.ok === false) {
    throw new Error(body.error || "操作失败");
  }
  return body;
}

async function load(groupId = state.groupId) {
  state.memory = await apiGet("memory", groupId ? { group_id: groupId } : {});
  state.groupId = state.memory.selected_group_id || "";
  render();
}

function render() {
  if (!state.memory) return;
  $("#pluginName").textContent = state.memory.plugin || "astrbot_plugin_group_relations";
  renderGroups();
  renderHeader();
  renderGroupForm();
  renderRelations();
  renderMembers();
  renderProfiles();
  applyPanelStates();
}

function applyPanelStates() {
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    const key = panel.dataset.panel;
    const collapsed = Boolean(state.collapsedPanels[key]);
    const body = panel.querySelector(".panel-body");
    const toggle = panel.querySelector("[data-action='toggle-panel']");
    panel.classList.toggle("is-collapsed", collapsed);
    if (body) body.hidden = collapsed;
    if (toggle) {
      toggle.setAttribute("aria-expanded", String(!collapsed));
      const chevron = toggle.querySelector(".chevron");
      if (chevron) chevron.textContent = collapsed ? "›" : "⌄";
    }
  });
}

function togglePanel(button) {
  const panel = button.closest("[data-panel]");
  if (!panel) return;
  const key = panel.dataset.panel;
  state.collapsedPanels[key] = !state.collapsedPanels[key];
  applyPanelStates();
}

function renderGroups() {
  const list = $("#groupList");
  const filter = state.filter.trim().toLowerCase();
  const groups = (state.memory.groups || []).filter((group) => {
    if (!filter) return true;
    return `${group.name} ${group.id}`.toLowerCase().includes(filter);
  });
  if (!groups.length) {
    list.innerHTML = `<div class="empty">暂无群空间</div>`;
    return;
  }
  list.innerHTML = groups
    .map(
      (group) => `
        <button class="group-item" data-group-id="${escapeHtml(group.id)}" aria-selected="${group.id === state.groupId}">
          <strong>${escapeHtml(group.name || group.id)}</strong>
          <span>${escapeHtml(group.id)}</span>
          <span>${group.relation_count || 0} 关系 · ${group.profile_count || 0} 画像 · ${group.member_count || 0} 成员</span>
        </button>
      `,
    )
    .join("");
}

function renderHeader() {
  const group = state.memory.selected_group;
  $("#groupTitle").textContent = group ? group.name || group.id : "未选择";
  $("#stats").innerHTML = group
    ? `
      <div class="stat"><span class="meta">关系</span><strong>${group.relation_count || 0}</strong></div>
      <div class="stat"><span class="meta">画像</span><strong>${group.profile_count || 0}</strong></div>
      <div class="stat"><span class="meta">成员</span><strong>${group.member_count || 0}</strong></div>
      <div class="stat"><span class="meta">消息</span><strong>${group.message_count || 0}</strong></div>
    `
    : "";
}

function renderGroupForm() {
  const group = state.memory.selected_group;
  $("#groupName").value = group?.name || "";
  $("#groupKind").value = group?.kind || "group";
  $("#groupOwnerId").value = group?.owner_user_id || "";
  $("#groupOwnerName").value = group?.owner_display_name || "";
  $("#groupOwnerEvidence").value = group?.owner_evidence || "";
  $("#groupPanel").hidden = !group;
}

function renderRelations() {
  const table = $("#relationTable");
  const relations = state.memory.relations || [];
  const rows = [
    relationRow({
      id: "",
      subject: "",
      subject_user_id: "",
      relation: "",
      object: "",
      object_user_id: "",
      category: "relation",
      note: "",
      confidence: 0.8,
      importance: 0.6,
      isNew: true,
    }),
    ...relations.map((item) => relationRow(item)),
  ];
  table.innerHTML = rows.length ? rows.join("") : `<div class="empty">暂无关系</div>`;
}

function relationRow(item) {
  const idAttr = item.id ? `data-id="${escapeHtml(item.id)}"` : "";
  return `
    <form class="relation-row" ${idAttr}>
      ${field("主体", item.subject, "subject")}
      ${field("主体ID", item.subject_user_id, "subject_user_id")}
      ${field("关系", item.relation, "relation")}
      ${field("客体", item.object, "object")}
      ${field("客体ID", item.object_user_id, "object_user_id")}
      ${field("类别", item.category || "relation", "category")}
      ${field("备注", item.note, "note")}
      ${field("可信度", item.confidence ?? 0.8, "confidence", "number")}
      ${field("重要度", item.importance ?? 0.6, "importance", "number")}
      <div class="row-actions">
        <button class="primary" data-action="save-relation" type="submit">${item.id ? "保存" : "新增"}</button>
        ${item.id ? `<button class="danger" data-action="delete-relation" type="button">删除</button>` : ""}
      </div>
    </form>
  `;
}

function renderMembers() {
  const list = $("#memberList");
  const members = state.memory.members || [];
  if (!members.length) {
    list.innerHTML = `<div class="empty">暂无成员目录；新群首次触达后会尝试自动获取。</div>`;
    return;
  }
  list.innerHTML = members
    .map(
      (member) => `
        <form class="member" data-user-id="${escapeHtml(member.user_id)}">
          <strong>${escapeHtml(member.display_name || member.user_id)}</strong>
          <span>${escapeHtml(member.user_id)}</span>
          ${roleSelect(member.role || "member")}
          <span>${escapeHtml(member.source || "unknown")}</span>
          <button class="secondary" data-action="save-member" type="submit">保存身份</button>
        </form>
      `,
    )
    .join("");
}

function renderProfiles() {
  const list = $("#profileList");
  const profiles = state.memory.profiles || [];
  const createPanel = profilePanel({
    id: "",
    user_id: "",
    display_name: "",
    preferred_name: "",
    group_role: "unknown",
    role_evidence: "",
    aliases: [],
    basic_profile: {},
    facts: [],
    isNew: true,
  });
  list.innerHTML = [createPanel, ...profiles.map(profilePanel)].join("");
}

function profilePanel(profile) {
  const aliases = (profile.aliases || []).join("\n");
  const basics = profile.id ? basicProfile(profile) : "";
  const facts = (profile.facts || []).map((fact) => factRow(profile, fact)).join("");
  return `
    <div class="profile" data-profile-id="${escapeHtml(profile.id || "")}">
      <form class="profile-title">
        ${field("用户 ID", profile.user_id, "user_id")}
        ${field("显示名称", profile.display_name, "display_name")}
        ${field("首选称呼", profile.preferred_name || "", "preferred_name")}
        ${field("群身份", profile.group_role || "unknown", "group_role")}
        ${field("身份来源", profile.role_evidence || "", "role_evidence")}
        ${textarea("别名", aliases, "aliases")}
        <div class="row-actions">
          <button class="primary" data-action="save-profile" type="submit">${profile.id ? "保存画像" : "新增画像"}</button>
          ${profile.id ? `<button class="danger" data-action="delete-profile" type="button">删除画像</button>` : ""}
        </div>
      </form>
      ${
        profile.id
          ? `
            ${basics}
            <div class="facts">
              <div class="subsection-title">普通画像事实</div>
              ${factRow(profile, { index: "", fact: "", note: "", confidence: 0.8, isNew: true })}
              ${facts || `<div class="empty">暂无画像事实</div>`}
            </div>
          `
          : ""
      }
    </div>
  `;
}

function basicProfile(profile) {
  const basics = profile.basic_profile || {};
  return `
    <div class="basic-profile">
      <div class="subsection-title">基础画像</div>
      ${basicProfileFields
        .map(([fieldName, label]) => {
          const items = basics[fieldName] || [];
          return `
            <div class="basic-group">
              <div class="basic-group-title">${label}</div>
              ${basicRow(profile, fieldName, { key: "", value: "", note: "", confidence: 0.8, importance: 0.8, isNew: true })}
              ${items.map((item) => basicRow(profile, fieldName, item)).join("") || `<div class="empty compact">暂无记录</div>`}
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function basicRow(profile, fieldName, item) {
  return `
    <form class="basic-row" data-profile-id="${escapeHtml(profile.id || "")}" data-field="${escapeHtml(fieldName)}" data-key="${escapeHtml(item.key || "")}" data-value="${escapeHtml(item.value || "")}">
      ${field("分类键", item.key, "key")}
      ${field("内容", item.value, "value")}
      ${field("证据备注", item.note, "note")}
      ${field("可信度", item.confidence ?? 0.8, "confidence", "number")}
      ${field("重要度", item.importance ?? 0.8, "importance", "number")}
      <div class="fact-actions">
        <button class="secondary" data-action="save-basic" type="submit">${item.isNew ? "新增" : "保存"}</button>
        ${item.isNew ? "" : `<button class="danger" data-action="delete-basic" type="button">删除</button>`}
      </div>
    </form>
  `;
}

function factRow(profile, fact) {
  return `
    <form class="fact-row" data-profile-id="${escapeHtml(profile.id || "")}" data-index="${escapeHtml(fact.index ?? "")}">
      ${field("画像事实", fact.fact, "fact")}
      ${field("类别", fact.category || "impression", "category")}
      ${field("证据备注", fact.note, "note")}
      ${field("可信度", fact.confidence ?? 0.8, "confidence", "number")}
      ${field("重要度", fact.importance ?? 0.6, "importance", "number")}
      <div class="fact-actions">
        <button class="secondary" data-action="save-fact" type="submit">${fact.isNew ? "新增事实" : "保存事实"}</button>
        ${fact.isNew ? "" : `<button class="danger" data-action="delete-fact" type="button">删除事实</button>`}
      </div>
    </form>
  `;
}

async function saveGroup() {
  if (!state.groupId) return;
  const body = {
    group_id: state.groupId,
    name: $("#groupName").value,
    kind: $("#groupKind").value,
    owner_user_id: $("#groupOwnerId").value,
    owner_display_name: $("#groupOwnerName").value,
    owner_evidence: $("#groupOwnerEvidence").value,
  };
  const result = await apiPost("group-save", body);
  state.memory = result.memory;
  toast("群信息已保存");
  render();
}

async function saveRelation(form) {
  const data = formData(form);
  const result = await apiPost("relation-save", {
    ...data,
    id: form.dataset.id || "",
    group_id: state.groupId,
  });
  state.memory = result.memory;
  toast("关系已保存");
  render();
}

async function deleteRelation(form) {
  const result = await apiPost("relation-delete", {
    group_id: state.groupId,
    relation_id: form.dataset.id,
  });
  state.memory = result.memory;
  toast("关系已删除");
  render();
}

async function saveMember(form) {
  const data = formData(form);
  const result = await apiPost("member-save", {
    group_id: state.groupId,
    user_id: form.dataset.userId,
    role: data.role,
  });
  state.memory = result.memory;
  toast("成员身份已保存");
  render();
}

async function refreshMembers(namePreference) {
  if (!state.groupId) return;
  const label = namePreference === "nickname" ? "QQ 名称" : "群名片";
  setMemberRefreshState(`正在按${label}获取群成员目录...`, true);
  try {
    const result = await apiPost("member-refresh", {
      group_id: state.groupId,
      name_preference: namePreference,
    });
    state.memory = result.memory;
    const count = result.refreshed_count ?? state.memory?.selected_group?.member_count ?? 0;
    toast(`成员目录已按${label}重新获取`);
    render();
    setMemberRefreshState(`已按${label}获取 ${count} 个成员`, false, "ok");
  } catch (error) {
    setMemberRefreshState(error.message || "成员目录重新获取失败", false, "error");
    throw error;
  }
}

async function saveProfile(panel, form) {
  const data = formData(form);
  const result = await apiPost("profile-save", {
    ...data,
    id: panel.dataset.profileId || "",
    aliases: data.aliases || "",
    group_id: state.groupId,
  });
  state.memory = result.memory;
  toast("画像已保存");
  render();
}

async function saveBasic(form) {
  const result = await apiPost("profile-basic-save", {
    ...formData(form),
    group_id: state.groupId,
    profile_id: form.dataset.profileId,
    field: form.dataset.field,
  });
  state.memory = result.memory;
  toast("基础画像已保存");
  render();
}

async function deleteBasic(form) {
  const result = await apiPost("profile-basic-delete", {
    group_id: state.groupId,
    profile_id: form.dataset.profileId,
    field: form.dataset.field,
    key: form.dataset.key,
    value: form.dataset.value,
  });
  state.memory = result.memory;
  toast("基础画像已删除");
  render();
}

async function deleteProfile(panel) {
  const result = await apiPost("profile-delete", {
    group_id: state.groupId,
    profile_id: panel.dataset.profileId,
  });
  state.memory = result.memory;
  toast("画像已删除");
  render();
}

async function saveFact(form) {
  const result = await apiPost("profile-fact-save", {
    ...formData(form),
    group_id: state.groupId,
    profile_id: form.dataset.profileId,
    index: form.dataset.index,
  });
  state.memory = result.memory;
  toast("画像事实已保存");
  render();
}

async function deleteFact(form) {
  const result = await apiPost("profile-fact-delete", {
    group_id: state.groupId,
    profile_id: form.dataset.profileId,
    index: form.dataset.index,
  });
  state.memory = result.memory;
  toast("画像事实已删除");
  render();
}

function bindEvents() {
  $("#refreshButton").addEventListener("click", () => load());
  $("#groupFilter").addEventListener("input", (event) => {
    state.filter = event.target.value;
    renderGroups();
  });
  $("#groupList").addEventListener("click", (event) => {
    const button = event.target.closest(".group-item");
    if (button) load(button.dataset.groupId);
  });
  document.querySelector(".content").addEventListener("click", (event) => {
    const refreshButton = event.target.closest("[data-action='refresh-members-card'], [data-action='refresh-members-nickname']");
    if (refreshButton) {
      const preference = refreshButton.dataset.action === "refresh-members-nickname" ? "nickname" : "card";
      run(() => refreshMembers(preference));
      return;
    }
    const button = event.target.closest("[data-action='toggle-panel']");
    if (button) togglePanel(button);
  });
  $("#saveGroupButton").addEventListener("click", () => run(saveGroup));
  $("#relationTable").addEventListener("submit", (event) => {
    event.preventDefault();
    run(() => saveRelation(event.target));
  });
  $("#relationTable").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action='delete-relation']");
    if (button) run(() => deleteRelation(button.closest(".relation-row")));
  });
  $("#memberList").addEventListener("submit", (event) => {
    event.preventDefault();
    run(() => saveMember(event.target));
  });
  $("#profileList").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    if (form.classList.contains("profile-title")) {
      run(() => saveProfile(form.closest(".profile"), form));
      return;
    }
    if (form.classList.contains("fact-row")) {
      run(() => saveFact(form));
      return;
    }
    if (form.classList.contains("basic-row")) {
      run(() => saveBasic(form));
    }
  });
  $("#profileList").addEventListener("click", (event) => {
    const profileButton = event.target.closest("[data-action='delete-profile']");
    if (profileButton) {
      run(() => deleteProfile(profileButton.closest(".profile")));
      return;
    }
    const basicButton = event.target.closest("[data-action='delete-basic']");
    if (basicButton) {
      run(() => deleteBasic(basicButton.closest(".basic-row")));
      return;
    }
    const factButton = event.target.closest("[data-action='delete-fact']");
    if (factButton) {
      run(() => deleteFact(factButton.closest(".fact-row")));
    }
  });
}

async function run(fn) {
  try {
    await fn();
  } catch (error) {
    toast(error.message || "操作失败");
  }
}

async function init() {
  if (!plugin) {
    toast("未找到 AstrBot 插件页面桥接对象");
    return;
  }
  await plugin.ready();
  bindEvents();
  await load();
}

init();
