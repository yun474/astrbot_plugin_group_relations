const plugin = window.AstrBotPluginPage;

const state = {
  memory: null,
  groupId: "",
  filter: "",
};

const $ = (selector) => document.querySelector(selector);

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
  renderProfiles();
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
          <span>${group.relation_count || 0} 关系 · ${group.profile_count || 0} 画像</span>
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
      <div class="stat"><span class="meta">消息</span><strong>${group.message_count || 0}</strong></div>
    `
    : "";
}

function renderGroupForm() {
  const group = state.memory.selected_group;
  $("#groupName").value = group?.name || "";
  $("#groupKind").value = group?.kind || "group";
  $("#groupPanel").hidden = !group;
}

function renderRelations() {
  const table = $("#relationTable");
  const relations = state.memory.relations || [];
  const rows = [
    relationRow({
      id: "",
      subject: "",
      relation: "",
      object: "",
      note: "",
      confidence: 0.8,
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
      ${field("关系", item.relation, "relation")}
      ${field("客体", item.object, "object")}
      ${field("备注", item.note, "note")}
      ${field("可信度", item.confidence ?? 0.8, "confidence", "number")}
      <div class="row-actions">
        <button class="primary" data-action="save-relation" type="submit">${item.id ? "保存" : "新增"}</button>
        ${item.id ? `<button class="danger" data-action="delete-relation" type="button">删除</button>` : ""}
      </div>
    </form>
  `;
}

function renderProfiles() {
  const list = $("#profileList");
  const profiles = state.memory.profiles || [];
  const createPanel = profilePanel({
    id: "",
    user_id: "",
    display_name: "",
    group_role: "unknown",
    role_evidence: "",
    aliases: [],
    facts: [],
    isNew: true,
  });
  list.innerHTML = [createPanel, ...profiles.map(profilePanel)].join("");
}

function profilePanel(profile) {
  const aliases = (profile.aliases || []).join("\n");
  const facts = (profile.facts || []).map((fact) => factRow(profile, fact)).join("");
  return `
    <div class="profile" data-profile-id="${escapeHtml(profile.id || "")}">
      <form class="profile-title">
        ${field("用户 ID", profile.user_id, "user_id")}
        ${field("显示名称", profile.display_name, "display_name")}
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
            <div class="facts">
              ${factRow(profile, { index: "", fact: "", note: "", confidence: 0.8, isNew: true })}
              ${facts || `<div class="empty">暂无画像事实</div>`}
            </div>
          `
          : ""
      }
    </div>
  `;
}

function factRow(profile, fact) {
  return `
    <form class="fact-row" data-profile-id="${escapeHtml(profile.id || "")}" data-index="${escapeHtml(fact.index ?? "")}">
      ${field("画像事实", fact.fact, "fact")}
      ${field("证据备注", fact.note, "note")}
      ${field("可信度", fact.confidence ?? 0.8, "confidence", "number")}
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
  $("#saveGroupButton").addEventListener("click", () => run(saveGroup));
  $("#relationTable").addEventListener("submit", (event) => {
    event.preventDefault();
    run(() => saveRelation(event.target));
  });
  $("#relationTable").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action='delete-relation']");
    if (button) run(() => deleteRelation(button.closest(".relation-row")));
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
    }
  });
  $("#profileList").addEventListener("click", (event) => {
    const profileButton = event.target.closest("[data-action='delete-profile']");
    if (profileButton) {
      run(() => deleteProfile(profileButton.closest(".profile")));
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
