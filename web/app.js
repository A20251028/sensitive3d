import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const jsonOpts = (method, data) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data ?? {}) });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "—");

const STEPS = ["import", "preview", "detect", "review", "repair", "package"];
const STEP_LABEL = { import: "导入检查", preview: "原始预览", detect: "只检测", review: "人工审核", repair: "执行修复", package: "下载" };
const ST_LABEL = { pending: "未开始", queued: "排队中", running: "进行中", done: "完成", failed: "失败", cancelled: "已取消", stopped: "已停止" };
const JOB_LABEL = { queued: "排队中", running: "处理中", done: "完成", failed: "失败", stopped: "已停止", ready: "等待操作" };
const MOUNT = { pole: "立杆", wall: "墙面", free: "悬挂/其他", unknown: "未知" };

// ---------------------------------------------------------------------------
// new job: upload / local path / demo
// ---------------------------------------------------------------------------
let selected = [];
function setSelected(list) {
  selected = list.filter((x) => !/(^|\/)(\._|\.DS_Store$|__MACOSX(\/|$))/.test(x.path));
  const skipped = list.length - selected.length;
  const size = selected.reduce((s, x) => s + x.file.size, 0);
  const osgb = selected.filter((x) => /\.osgb$/i.test(x.path)).length;
  $("picked").textContent = selected.length
    ? `已选择 ${selected.length} 个文件 (${(size / 1048576).toFixed(1)} MB)` + (osgb ? `, 其中 ${osgb} 个 OSGB` : "") + (skipped ? `, 忽略 ${skipped} 个附属文件` : "")
    : "尚未选择数据";
  $("picked").classList.toggle("muted", !selected.length);
  $("upload").disabled = !selected.length;
}
async function readEntry(entry, prefix = "") {
  if (entry.isFile) return [{ file: await new Promise((res, rej) => entry.file(res, rej)), path: prefix + entry.name }];
  const reader = entry.createReader();
  const out = [];
  for (;;) {
    const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    for (const e of batch) out.push(...(await readEntry(e, prefix + entry.name + "/")));
  }
  return out;
}
const drop = $("drop");
drop.addEventListener("click", () => $("pickDir").click());
drop.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && $("pickDir").click());
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  const items = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (items.length) setSelected((await Promise.all(items.map((it) => readEntry(it)))).flat());
  else setSelected([...e.dataTransfer.files].map((f) => ({ file: f, path: f.name })));
});
$("pickDir").addEventListener("change", (e) => setSelected([...e.target.files].map((f) => ({ file: f, path: f.webkitRelativePath || f.name }))));
$("pickFiles").addEventListener("change", (e) => setSelected([...e.target.files].map((f) => ({ file: f, path: f.name }))));

function limits() {
  const n = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : null; };
  const out = { max_seconds: n("limSeconds"), max_memory_mb: n("limMemory"), max_decoded_mb: n("limDecoded"), bridge_timeout: n("limBridge") };
  return Object.fromEntries(Object.entries(out).filter(([, v]) => v !== null));
}
function detectOptions() {
  const categories = [...document.querySelectorAll(".cats input:checked")].map((x) => x.value);
  return {
    detection: { categories, min_score: parseFloat($("optMinScore").value) || 0.55 },
    auto_accept_score: parseFloat($("optAuto").value) || 0.85,
    diagnostics: $("optDiag").checked,
    limits: limits(),
  };
}

$("upload").addEventListener("click", () => {
  if (!selected.length) return;
  const fd = new FormData();
  for (const { file, path } of selected) fd.append("files", file, path);
  fd.append("options", JSON.stringify({ limits: limits() }));
  fd.append("name", selected[0].path.split("/")[0]);
  $("upload").disabled = true;
  showProgress("上传中…", 0);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = (e) => e.lengthComputable && showProgress(`上传中 ${(100 * e.loaded / e.total).toFixed(0)}%`, 0);
  xhr.onload = () => {
    $("upload").disabled = false;
    if (xhr.status >= 300) return showProgress("上传失败: " + (JSON.parse(xhr.responseText || "{}").detail || xhr.status), 0);
    track(JSON.parse(xhr.responseText).id);
  };
  xhr.onerror = () => { $("upload").disabled = false; showProgress("上传失败 (网络错误)", 0); };
  xhr.send(fd);
});
$("localImport").addEventListener("click", async () => {
  const path = $("localPath").value.trim();
  if (!path) return;
  try { track((await api("/api/jobs/local", jsonOpts("POST", { path, options: { limits: limits() } }))).id); }
  catch (e) { alert("本机路径导入失败: " + e.message); }
});
$("demo").addEventListener("click", async () => {
  showProgress("生成合成示例数据 (约 30 秒)…", 0);
  try { track((await api("/api/demo", jsonOpts("POST", { limits: limits() }))).id); }
  catch (e) { showProgress("失败: " + e.message, 0); }
});

// ---------------------------------------------------------------------------
// job tracking
// ---------------------------------------------------------------------------
let job = null;
let jobId = null;
let timer = null;
let loaded = { detection: null, scan: null, repair: null, review: null, overviewBefore: false, overviewAfter: false };

function showProgress(msg, frac) {
  $("jobCard").hidden = false;
  $("jobMsg").textContent = msg;
  $("jobPct").textContent = `${Math.round((frac || 0) * 100)}%`;
  $("barFill").style.width = `${(frac || 0) * 100}%`;
}

function track(id) {
  if (jobId !== id) {
    loaded = { detection: null, scan: null, repair: null, review: null, overviewBefore: false, overviewAfter: false };
    clearViewer();
    ["scanCard", "detectCard", "reviewCard", "repairCard", "resultCard", "viewerCard"].forEach((c) => ($(c).hidden = true));
  }
  jobId = id;
  clearInterval(timer);
  poll();
  timer = setInterval(poll, 1200);
}

async function poll() {
  if (!jobId) return;
  let j;
  try { j = await api(`/api/jobs/${jobId}`); } catch { return; }
  job = j;
  render(j);
  if (!["running", "queued"].includes(j.status)) refreshHistory();
}

function stepStatus(j, s) {
  if (s === "review") return j.review_saved ? "done" : (j.steps.detect.status === "done" ? "pending" : "pending");
  return j.steps[s]?.status || "pending";
}

function render(j) {
  $("jobCard").hidden = false;
  $("jobName").textContent = `${j.name || j.id} · ${JOB_LABEL[j.status] || j.status}`;
  const src = j.source || {};
  $("jobSource").textContent = src.type === "local" ? `本机目录 (只读复制): ${src.path}` + (src.unchanged === true ? " · 源目录哈希前后一致" : "") : src.type === "demo" ? "合成示例数据 (每次重新生成)" : `上传 ${src.files ?? ""} 个文件` + (src.ignored ? `, 忽略 ${src.ignored} 个附属文件` : "");
  $("stepper").innerHTML = STEPS.map((s) => {
    const st = stepStatus(j, s);
    return `<li class="st-${st}${j.current === s ? " cur" : ""}"><span>${STEP_LABEL[s]}</span><small>${ST_LABEL[st] || st}</small></li>`;
  }).join("");
  showProgress(j.message || "", j.current ? j.progress : (j.status === "done" ? 1 : j.progress));
  $("cancel").hidden = !["running", "queued"].includes(j.status);
  $("log").textContent = (j.log || []).join("\n");
  const busy = ["running", "queued"].includes(j.status);

  if (j.steps.import.status === "done" && !loaded.scan) loadScan();
  if (j.overview?.before && !loaded.overviewBefore) { loaded.overviewBefore = true; loadModel("before", j.overview.before); }
  if (j.overview?.after && !loaded.overviewAfter) { loaded.overviewAfter = true; loadModel("after", j.overview.after); }

  $("detectCard").hidden = j.steps.import.status !== "done";
  $("detect").disabled = busy;
  if (j.detection) {
    $("detectSummary").innerHTML = `<span class="tag">候选 ${j.detection.total}</span><span class="tag ok">可自动 ${j.detection.auto}</span><span class="tag warn">待审核 ${j.detection.review}</span><span class="muted">颜色候选 ${j.detection.colour_candidates ?? "—"} · ${j.detection.seconds ?? "—"} 秒</span>` + (j.detection.stopped ? `<div class="warnbox">检测提前停止: ${esc(j.detection.stopped.message)} (已保存的结果仍可查看)</div>` : "");
  }
  if (j.steps.detect.status === "done" || j.steps.detect.status === "stopped") {
    if (!loaded.detection) loadDetection();
  }
  $("reviewCard").hidden = !loaded.detection;
  $("saveReview").disabled = busy;
  $("repairCard").hidden = !loaded.detection;
  $("repair").disabled = busy || !j.review_saved || (j.scan && !j.scan.ok_for_repair);
  $("repair").title = j.scan && !j.scan.ok_for_repair ? "导入检查存在阻断性问题, 不能修复" : (!j.review_saved ? "请先保存审核结果" : "");
  if (["done", "stopped", "cancelled", "failed"].includes(j.steps.repair.status) && !loaded.repair) loadRepair();
  $("resultCard").hidden = j.steps.package.status !== "done";
  if (j.steps.package.status === "done") $("download").href = `/api/jobs/${j.id}/download`;
  renderVerdict(j);
}

$("cancel").onclick = async () => { if (jobId) { await api(`/api/jobs/${jobId}/cancel`, { method: "POST" }).catch(() => {}); poll(); } };

async function refreshHistory() {
  const jobs = await api("/api/jobs").catch(() => []);
  const ul = $("history");
  ul.innerHTML = jobs.length ? "" : '<li class="muted">暂无</li>';
  for (const j of jobs) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = j.id === jobId ? "on" : "";
    b.innerHTML = `<span>${esc(j.name || j.id)}<br><small class="muted">${new Date(j.created * 1000).toLocaleString()}</small></span><span class="st-${j.status}">${JOB_LABEL[j.status] || j.status}</span>`;
    b.onclick = () => track(j.id);
    li.appendChild(b);
    ul.appendChild(li);
  }
}

// ---------------------------------------------------------------------------
// 1 import check
// ---------------------------------------------------------------------------
async function loadScan() {
  loaded.scan = await api(`/api/jobs/${jobId}/scan`).catch(() => null);
  const s = loaded.scan;
  if (!s) return;
  $("scanCard").hidden = false;
  const list = (arr, f, max = 20) => (arr && arr.length ? `<ul class="plain">${arr.slice(0, max).map((x) => `<li>${f(x)}</li>`).join("")}${arr.length > max ? `<li class="muted">… 共 ${arr.length} 项</li>` : ""}</ul>` : '<span class="muted">无</span>');
  const levels = s.levels ? Object.entries(s.levels).map(([d, n]) => `L${d}: ${n}`).join(" · ") : "—";
  $("scanBody").innerHTML = `
    ${(s.blocking_errors || []).length ? `<div class="errbox"><b>阻断性问题 (修复将被拒绝)</b>${list(s.blocking_errors, esc)}</div>` : '<div class="okbox">结构检查通过, 未发现阻断性问题</div>'}
    <table class="kv">
      <tr><td>格式</td><td>${esc(s.format)}</td><td>可读文件</td><td>${s.files_ok} / ${s.files}</td></tr>
      <tr><td>瓦片</td><td>${s.tiles}</td><td>最精细层级文件</td><td>${s.leaf_files}</td></tr>
      <tr><td>三角形</td><td>${(s.triangles || 0).toLocaleString()}</td><td>贴图解码估算</td><td>${s.decoded_bytes_total ? (s.decoded_bytes_total / 1048576).toFixed(0) + " MB" : "—"}</td></tr>
      <tr><td>LOD 层级</td><td colspan="3">${levels}</td></tr>
      <tr><td>根文件</td><td>${(s.roots || []).length}</td><td>多父节点文件</td><td>${(s.multi_parent || []).length}</td></tr>
      <tr><td>忽略的附属文件</td><td>${(s.ignored || []).length}</td><td>输入哈希</td><td>${s.input_manifest ? `${s.input_manifest.files} 个文件已记录` : "—"}</td></tr>
    </table>
    <details><summary>读取失败的文件 (${(s.failed || []).length})</summary>${list(s.failed, (x) => `${esc(x.rel)}: ${esc(x.error)}`)}</details>
    <details><summary>缺失的子文件引用 (${(s.missing_refs || []).length})</summary>${list(s.missing_refs, (x) => `${esc(x.parent)} → ${esc(x.ref)}`)}</details>
    <details><summary>引用环 (${(s.cycles || []).length}) / 无法从根到达 (${(s.unreachable || []).length})</summary>${list(s.cycles, (c) => esc([].concat(c).join(" → ")))}${list(s.unreachable, esc)}</details>
    <details><summary>贴图问题 (${(s.texture_issues || []).length})</summary>${list(s.texture_issues, (x) => `${esc(x.rel)}: ${esc([].concat(x.issues).join("; "))}`)}</details>
    <details><summary>警告 (${(s.warnings || []).length})</summary>${list(s.warnings, esc)}</details>`;
}

// ---------------------------------------------------------------------------
// 3 detection
// ---------------------------------------------------------------------------
$("detect").onclick = async () => {
  try { await api(`/api/jobs/${jobId}/detect`, jsonOpts("POST", { options: detectOptions() })); loaded.detection = null; poll(); }
  catch (e) { alert(e.message); }
};

let decisions = {};
let protectedBoxes = [];
async function loadDetection() {
  loaded.detection = await api(`/api/jobs/${jobId}/detection`).catch(() => null);
  if (!loaded.detection) return;
  const review = await api(`/api/jobs/${jobId}/review`).catch(() => null);
  loaded.review = review;
  decisions = {};
  for (const c of loaded.detection.candidates) {
    const d = review?.decisions?.find((x) => x.id === c.id);
    decisions[c.id] = d ? { accept: !!d.accept, operation: d.operation || c.suggested.operation, margin: d.margin || 0 }
      : { accept: c.status === "auto" && c.suggested.accept, operation: c.suggested.operation === "none" ? "texture" : c.suggested.operation, margin: 0 };
  }
  protectedBoxes = (review?.protected || []).map((p) => ({ name: p.name || "", min: p.min, max: p.max }));
  renderCandidates();
  renderProtected();
  setMarkers(loaded.detection.candidates);
}

function evidenceImg(p, alt) { return p ? `<img loading="lazy" src="/api/jobs/${jobId}/files/${p}" alt="${alt}">` : '<div class="noimg">无图</div>'; }

function renderCandidates() {
  const det = loaded.detection;
  $("reviewCount").textContent = `(${det.candidates.length} 个候选: ${det.counts.auto} 可自动 / ${det.counts.review} 待审核)`;
  const box = $("candidates");
  box.innerHTML = det.candidates.length ? "" : '<p class="muted">未检测到候选。若确认存在敏感标志, 请勾选“保存被拒颜色候选”重新检测以排查原因。</p>';
  for (const c of det.candidates) {
    const d = decisions[c.id];
    const geoAllowed = c.mapping.ok && ["pole", "wall", "free"].includes(c.mount.type);
    const p = c.provenance || {};
    const el = document.createElement("div");
    el.className = `region cand ${c.status}`;
    el.dataset.id = c.id;
    el.innerHTML = `
      <div class="pics">
        <figure>${evidenceImg(c.evidence?.overlay, "检测特写")}<figcaption>检测特写 (红框为掩膜)</figcaption></figure>
        <figure>${evidenceImg(c.evidence?.context, "周边")}<figcaption>周边视角</figcaption></figure>
      </div>
      <div class="meta">
        <b>#${c.id} ${esc(c.category_label)}${c.label ? " · " + esc(c.label) : ""} <span class="badge ${c.status}">${c.status === "auto" ? "可自动" : "待审核"}</span></b>
        ${c.review_reasons.length ? `<span class="reasons">${c.review_reasons.map(esc).join("; ")}</span>` : ""}
        <span>置信度 (非准确率)</span><span>${fmt(c.score)}</span>
        <span>二维→三维映射</span><span class="${c.mapping.ok ? "" : "bad"}">${c.mapping.ok ? "成功" : "失败"}${c.mapping.reason ? " · " + esc(c.mapping.reason) : ""} (${c.mapping.mapped_pixels ?? "—"}/${c.mapping.mask_pixels ?? "—"} 像素)</span>
        <span>安装方式</span><span>${MOUNT[c.mount.type] || c.mount.type} · 置信${c.mount.confidence === "high" ? "高" : "低"}</span>
        <span>尺寸 / 位置</span><span>${fmt(c.region.width)}×${fmt(c.region.height)} m · ${c.region.center.map((v) => v.toFixed(2)).join(", ")}</span>
        <span>来源</span><span>${esc((p.tiles || []).join(", "))} · ${(p.files || []).length} 个文件 · 最精细 LOD${p.detector ? " · " + esc(p.detector) : ""}</span>
      </div>
      <details class="prov"><summary>来源文件与相机</summary><pre>${esc(JSON.stringify({ files: p.files, lod: p.lod, bbox2d: p.bbox2d, camera: p.camera }, null, 1))}</pre></details>
      <div class="decide">
        <label><input type="radio" name="acc${c.id}" value="1" ${d.accept ? "checked" : ""}> 去除</label>
        <label><input type="radio" name="acc${c.id}" value="0" ${!d.accept ? "checked" : ""}> 保留 (拒绝)</label>
        <select class="op" ${c.mapping.ok ? "" : "disabled"}>
          <option value="texture" ${d.operation === "texture" ? "selected" : ""}>仅修复纹理 (几何与 UV 不变)</option>
          <option value="geometry" ${d.operation === "geometry" ? "selected" : ""} ${geoAllowed ? "" : "disabled"}>删除牌面/立杆几何 + 修复纹理</option>
        </select>
        <label class="small">外扩 (m) <input class="margin" type="number" step="0.01" min="-0.2" max="1" value="${d.margin || 0}"></label>
        <button class="btn ghost small locate" type="button">定位</button>
        <button class="btn ghost small protect" type="button" title="例如误检: 把这个位置设为保护区域">设为保护区</button>
      </div>`;
    el.querySelectorAll(`input[name=acc${c.id}]`).forEach((r) => (r.onchange = () => { decisions[c.id].accept = r.value === "1" && r.checked; }));
    el.querySelector(".op").onchange = (e) => { decisions[c.id].operation = e.target.value; };
    el.querySelector(".margin").onchange = (e) => { decisions[c.id].margin = parseFloat(e.target.value) || 0; };
    el.querySelector(".locate").onclick = () => focusRegion(c);
    el.querySelector(".protect").onclick = () => {
      const r = Math.max(c.region.width, c.region.height) / 2 + 0.3;
      protectedBoxes.push({ name: `候选 #${c.id} 周边`, min: c.region.center.map((v) => +(v - r).toFixed(3)), max: c.region.center.map((v) => +(v + r).toFixed(3)) });
      decisions[c.id].accept = false;
      renderCandidates();
      renderProtected();
    };
    if (!c.mapping.ok) { d.accept = false; el.querySelector(`input[name=acc${c.id}][value="1"]`).disabled = true; }
    box.appendChild(el);
  }
}
$("acceptAuto").onclick = () => { for (const c of loaded.detection.candidates) if (c.status === "auto" && c.mapping.ok) decisions[c.id].accept = true; renderCandidates(); };
$("rejectAll").onclick = () => { for (const id in decisions) decisions[id].accept = false; renderCandidates(); };

function renderProtected() {
  const box = $("protected");
  box.innerHTML = protectedBoxes.length ? "" : '<p class="muted small">未设置保护区域</p>';
  protectedBoxes.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "prow";
    const xyz = (k) => [0, 1, 2].map((a) => `<input type="number" step="0.01" data-k="${k}" data-a="${a}" value="${p[k][a]}">`).join("");
    row.innerHTML = `<input class="pname" type="text" value="${esc(p.name)}" placeholder="名称"> <span class="small">min</span>${xyz("min")} <span class="small">max</span>${xyz("max")} <button class="btn ghost small" type="button">删除</button>`;
    row.querySelector(".pname").onchange = (e) => { p.name = e.target.value; };
    row.querySelectorAll("input[data-k]").forEach((inp) => (inp.onchange = () => { p[inp.dataset.k][+inp.dataset.a] = parseFloat(inp.value); }));
    row.querySelector("button").onclick = () => { protectedBoxes.splice(i, 1); renderProtected(); };
    box.appendChild(row);
  });
}
$("addProtected").onclick = () => {
  const c = controls.target;
  const w = [c.x + origin[0], -c.z + origin[1], c.y + origin[2]];
  protectedBoxes.push({ name: "", min: w.map((v) => +(v - 1).toFixed(2)), max: w.map((v) => +(v + 1).toFixed(2)) });
  renderProtected();
};
$("saveReview").onclick = async () => {
  const body = {
    decisions: Object.entries(decisions).map(([id, d]) => ({ id: +id, accept: d.accept, operation: d.operation, margin: d.margin })),
    protected: protectedBoxes,
  };
  try {
    const r = await api(`/api/jobs/${jobId}/review`, jsonOpts("PUT", body));
    $("reviewMsg").innerHTML = `<div class="okbox">审核已保存: ${r.targets.length} 个目标将被处理 (${r.targets.filter((t) => t.operation === "geometry").length} 个删除几何), ${r.skipped.length} 个不处理</div>`;
    poll();
  } catch (e) {
    $("reviewMsg").innerHTML = `<div class="errbox">无法保存: ${esc(e.message)}</div>`;
  }
};

// ---------------------------------------------------------------------------
// 5 repair + 6 result
// ---------------------------------------------------------------------------
$("repair").onclick = async () => {
  try { await api(`/api/jobs/${jobId}/repair`, jsonOpts("POST", { options: { limits: limits() } })); loaded.repair = null; poll(); }
  catch (e) { alert(e.message); }
};

async function loadRepair() {
  loaded.repair = await api(`/api/jobs/${jobId}/repair`).catch(() => null);
  const r = loaded.repair;
  if (!r) return;
  const rb = (r.files || []).filter((f) => f.readback && !f.readback.ok);
  const atlas = (r.atlas_residual || []).filter((a) => a.sign_colour_fraction > 0.02);
  const prot = r.protected_check || [];
  $("repairBody").innerHTML = `
    ${r.stopped ? `<div class="errbox">修复提前停止: ${esc(r.stopped.message)}。输出不完整, 不能使用。</div>` : ""}
    ${(r.warnings || []).length ? `<div class="warnbox">${r.warnings.map(esc).join("<br>")}</div>` : ""}
    <table class="kv">
      <tr><td>处理目标</td><td>${(r.targets || []).length}</td><td>未处理候选</td><td>${(r.skipped || []).length}</td></tr>
      <tr><td>修改文件</td><td>${r.files_modified ?? 0}</td><td>回读失败</td><td class="${rb.length ? "bad" : ""}">${rb.length}</td></tr>
      <tr><td>删除面片</td><td>${r.faces_removed ?? 0}</td><td>补洞面片</td><td>${r.faces_added ?? 0}</td></tr>
      <tr><td>重绘纹理像素</td><td>${(r.texels_painted ?? 0).toLocaleString()}</td><td>图集残留超标</td><td class="${atlas.length ? "bad" : ""}">${atlas.length}</td></tr>
      <tr><td>保护区域</td><td>${prot.length ? (prot.every((p) => p.ok) ? "全部未变化" : '<span class="bad">有变化</span>') : "未设置"}</td><td>输入哈希核对</td><td class="${r.input_check?.ok === false ? "bad" : ""}">${r.input_check?.ok === true ? "一致" : r.input_check?.ok === false ? "不一致!" : "未记录"}</td></tr>
      <tr><td>复检 (辅助)</td><td colspan="3">${(r.residual_detections || []).length} 处疑似标志 · 复检为 0 不代表绝对没有残留, 请看前后对比图</td></tr>
    </table>
    <details><summary>逐文件改动 (${(r.files || []).length})</summary><pre>${esc(JSON.stringify(r.files, null, 1))}</pre></details>`;
  const box = $("regions");
  box.innerHTML = "";
  for (const g of r.regions || []) {
    const el = document.createElement("div");
    el.className = "region";
    el.innerHTML = `<div class="pics"><figure>${evidenceImg(g.preview_before, "修复前")}<figcaption>修复前</figcaption></figure><figure>${evidenceImg(g.preview_after, "修复后")}<figcaption>修复后 (同一相机)</figcaption></figure></div>
      <div class="meta"><b>#${g.id} ${esc(g.category_label)} ${esc(g.label || "")}</b><span>操作</span><span>${g.operation === "geometry" ? "删除几何 + 纹理" : "仅纹理"} (${g.decided_by === "review" ? "人工确认" : "自动"})</span><span>安装方式</span><span>${MOUNT[g.mount] || g.mount}</span></div>`;
    el.onclick = () => focusRegion({ region: g, id: g.id });
    box.appendChild(el);
  }
}

function renderVerdict(j) {
  const s = j.scan || {};
  const r = j.repair || {};
  const items = [
    ["解析成功", s.files ? (s.files_ok === s.files && !(s.blocking_errors || []).length ? "是" : `否 (${s.files_ok}/${s.files}, 阻断 ${(s.blocking_errors || []).length})`) : "—"],
    ["流程运行成功", j.steps.package.status === "done" ? (r.readback_failures ? `有 ${r.readback_failures} 个文件回读失败` : "是 (输出全部回读通过)") : "—"],
    ["检测候选", j.detection ? `${j.detection.total} 个 (可自动 ${j.detection.auto} / 待审核 ${j.detection.review})` : "—"],
    ["视觉验收", "需人工对照前后图确认 (系统不自动判定通过)"],
    ["真实数据验收", j.source?.type === "demo" ? "否: 这是合成数据" : "需按 docs/真实数据验证.md 逐项完成"],
  ];
  $("verdict").innerHTML = items.map(([k, v]) => `<div><span>${k}</span><b>${esc(v)}</b></div>`).join("");
}

// ---------------------------------------------------------------------------
// 3D viewer
// ---------------------------------------------------------------------------
const wrap = $("canvasWrap");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe6ee);
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 5000);
camera.position.set(30, 30, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 1));
const groups = { before: new THREE.Group(), after: new THREE.Group(), markers: new THREE.Group() };
Object.values(groups).forEach((g) => scene.add(g));
let showing = "before";
let origin = [0, 0, 0];
let labels = [];

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap);
resize();
const toView = (p) => new THREE.Vector3(p[0] - origin[0], p[2] - origin[2], -(p[1] - origin[1]));
const dirToView = (d) => new THREE.Vector3(d[0], d[2], -d[1]);

function clearGroup(g) {
  for (const c of [...g.children]) {
    g.remove(c);
    c.traverse((o) => { o.geometry?.dispose(); [].concat(o.material || []).forEach((m) => { m.map?.dispose(); m.dispose(); }); });
  }
}
function clearViewer() {
  clearGroup(groups.before); clearGroup(groups.after); clearGroup(groups.markers);
  labels.forEach((l) => l.el.remove());
  labels = [];
  $("showAfter").disabled = true;
  setShowing("before");
}
function asUnlit(root) {
  root.traverse((o) => {
    if (o.isMesh) {
      const map = o.material.map || null;
      o.material = new THREE.MeshBasicMaterial({ map, color: map ? 0xffffff : 0xbbbbbb, side: THREE.DoubleSide });
    }
  });
  return root;
}
async function loadModel(which, ov) {
  $("viewerCard").hidden = false;
  $("viewerEmpty").hidden = true;
  if (which === "before") origin = ov.origin;
  $("viewerInfo").textContent = "加载模型…";
  const g = await new Promise((res, rej) => new GLTFLoader().load(`/api/jobs/${jobId}/files/${ov.path}`, (x) => res(x.scene), undefined, rej)).catch((e) => { $("viewerInfo").textContent = "模型加载失败: " + e.message; return null; });
  if (!g) return;
  clearGroup(groups[which]);
  groups[which].add(asUnlit(g));
  $("viewerInfo").textContent = `${ov.files} 个文件 · ${(ov.triangles / 1000).toFixed(0)}k 三角形` + (ov.level !== undefined ? ` · LOD 第 ${ov.level} 级` : "") + (ov.note ? " · " + ov.note : "");
  if (which === "after") { $("showAfter").disabled = false; setShowing("after"); }
  else resetView();
}
function setMarkers(cands) {
  clearGroup(groups.markers);
  labels.forEach((l) => l.el.remove());
  labels = [];
  for (const c of cands) {
    const r = c.region;
    const geo = new THREE.BoxGeometry(r.width + 0.1, r.height + 0.1, 0.15);
    const color = c.status === "auto" ? 0xff3030 : 0xff9800;
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color, depthTest: false }));
    box.renderOrder = 10;
    box.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(dirToView(r.axis_u || [1, 0, 0]), dirToView(r.axis_v || [0, 0, 1]), dirToView(r.normal)));
    box.position.copy(toView(r.center));
    groups.markers.add(box);
    const el = document.createElement("div");
    el.className = "marker-label " + c.status;
    el.textContent = `#${c.id} ${c.label || c.category_label}`;
    wrap.appendChild(el);
    labels.push({ el, pos: toView(r.center).add(new THREE.Vector3(0, r.height / 2 + 0.3, 0)) });
  }
}
function setShowing(which) {
  showing = which;
  groups.before.visible = which === "before";
  groups.after.visible = which === "after";
  $("showBefore").classList.toggle("on", which === "before");
  $("showAfter").classList.toggle("on", which === "after");
}
$("showBefore").onclick = () => setShowing("before");
$("showAfter").onclick = () => setShowing("after");
$("showMarkers").onchange = (e) => { groups.markers.visible = e.target.checked; };
function resetView() {
  const box = new THREE.Box3().setFromObject(groups.before.children.length ? groups.before : groups.after);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const s = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.45, 0.55, 0.7).multiplyScalar(s * 0.8));
  camera.near = s / 2000; camera.far = s * 20; camera.updateProjectionMatrix();
  controls.update();
}
$("resetView").onclick = resetView;
function focusRegion(c) {
  const r = c.region;
  document.querySelectorAll(".cand").forEach((el) => el.classList.toggle("on", Number(el.dataset.id) === c.id));
  const t = toView(r.center);
  const dist = Math.max(4, (r.width || 1) * 6);
  controls.target.copy(t);
  camera.position.copy(t).add(dirToView(r.normal).multiplyScalar(dist)).add(new THREE.Vector3(0, dist * 0.35, 0));
  camera.near = 0.02; camera.far = 2000; camera.updateProjectionMatrix();
  controls.update();
  $("viewerCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
const tmp = new THREE.Vector3();
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  const w = wrap.clientWidth, h = wrap.clientHeight;
  for (const l of labels) {
    tmp.copy(l.pos).project(camera);
    const vis = groups.markers.visible && tmp.z < 1 && Math.abs(tmp.x) < 1.1 && Math.abs(tmp.y) < 1.1;
    l.el.style.display = vis ? "" : "none";
    if (vis) { l.el.style.left = `${(tmp.x * 0.5 + 0.5) * w}px`; l.el.style.top = `${(-tmp.y * 0.5 + 0.5) * h}px`; }
  }
})();

// ---------------------------------------------------------------------------
// start-up
// ---------------------------------------------------------------------------
api("/api/health").then((h) => {
  const st = h.bridge || {};
  const ok = h.osgb_bridge;
  $("health").textContent = ok ? `服务正常 · OSGB 组件自检通过 · v${h.version}` : `服务正常 · OSGB 组件不可用${st.error ? ": " + st.error : ""}${st.hint ? " (" + st.hint + ")" : ""}`;
  $("health").className = "pill " + (ok ? "ok" : "bad");
  $("health").title = JSON.stringify(st.selftest || st, null, 1);
  if ((h.local_import || []).length) {
    $("localBox").hidden = false;
    $("localRoots").textContent = "允许的目录: " + h.local_import.join(", ");
  }
}).catch(() => { $("health").textContent = "无法连接服务"; $("health").className = "pill bad"; });
refreshHistory().then(async () => {
  const jobs = await api("/api/jobs").catch(() => []);
  if (jobs[0]) track(jobs[0].id);
});
