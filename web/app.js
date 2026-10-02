import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});

const STATUS = { queued: "排队中", running: "处理中", done: "完成", failed: "失败" };
const MOUNT = { pole: "立杆", wall: "墙面", free: "悬挂/其他", none: "仅纹理", unknown: "未知" };

// ---------------------------------------------------------------------------
// file selection (drag & drop folders, folder picker, zip)
// ---------------------------------------------------------------------------
let selected = []; // [{file, path}]

function setSelected(list) {
  selected = list;
  const n = list.length;
  const size = list.reduce((s, x) => s + x.file.size, 0);
  const osgb = list.filter((x) => /\.osgb$/i.test(x.path)).length;
  $("picked").textContent = n
    ? `已选择 ${n} 个文件 (${(size / 1048576).toFixed(1)} MB)` + (osgb ? `, 其中 ${osgb} 个 OSGB` : "") + ` · ${list[0].path.split("/")[0]}`
    : "尚未选择数据";
  $("picked").classList.toggle("muted", !n);
  $("start").disabled = !n;
}

async function readEntry(entry, prefix = "") {
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej));
    return [{ file, path: prefix + entry.name }];
  }
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
  if (items.length) {
    const lists = await Promise.all(items.map((it) => readEntry(it)));
    setSelected(lists.flat());
  } else {
    setSelected([...e.dataTransfer.files].map((f) => ({ file: f, path: f.name })));
  }
});
$("pickDir").addEventListener("change", (e) => setSelected([...e.target.files].map((f) => ({ file: f, path: f.webkitRelativePath || f.name }))));
$("pickFiles").addEventListener("change", (e) => setSelected([...e.target.files].map((f) => ({ file: f, path: f.name }))));

function options() {
  const categories = [...document.querySelectorAll(".cats input:checked")].map((x) => x.value);
  if (categories.length) categories.push("other_sign"); // generic signs reported by a learned detector
  return {
    detection: { categories },
    remove_geometry: $("optGeom").checked,
    texture: { method: $("optInpaint").value },
  };
}

// ---------------------------------------------------------------------------
// jobs
// ---------------------------------------------------------------------------
let currentJob = null;
let pollTimer = null;

$("start").addEventListener("click", () => {
  if (!selected.length) return;
  const fd = new FormData();
  for (const { file, path } of selected) fd.append("files", file, path);
  fd.append("options", JSON.stringify(options()));
  fd.append("name", selected[0].path.split("/")[0]);
  $("start").disabled = true;
  showJobCard("上传中…", 0);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = (e) => e.lengthComputable && showJobCard(`上传中 ${(100 * e.loaded / e.total).toFixed(0)}%`, 0);
  xhr.onload = () => {
    $("start").disabled = false;
    if (xhr.status >= 300) return showJobCard("上传失败: " + xhr.responseText, 0);
    track(JSON.parse(xhr.responseText).id);
  };
  xhr.onerror = () => { $("start").disabled = false; showJobCard("上传失败 (网络错误)", 0); };
  xhr.send(fd);
});

$("demo").addEventListener("click", async () => {
  showJobCard("准备示例数据 (首次约需 30 秒)…", 0);
  try {
    const job = await api("/api/demo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(options()) });
    track(job.id);
  } catch (e) { showJobCard("失败: " + e.message, 0); }
});

function showJobCard(msg, frac, name) {
  $("jobCard").hidden = false;
  if (name !== undefined) $("jobName").textContent = name;
  $("jobMsg").textContent = msg;
  $("jobPct").textContent = `${Math.round(frac * 100)}%`;
  $("barFill").style.width = `${frac * 100}%`;
}

function track(id) {
  currentJob = id;
  $("resultBox").hidden = true;
  clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch { return; }
    if (currentJob !== id) return;
    showJobCard(job.error ? `${job.message}: ${job.error}` : job.message, job.progress, `${job.name || job.id} · ${STATUS[job.status]}`);
    $("log").textContent = (job.log || []).join("\n");
    if (job.status === "done" || job.status === "failed") {
      clearInterval(pollTimer);
      refreshHistory();
      if (job.status === "done") showResult(job);
    }
  };
  tick();
  pollTimer = setInterval(tick, 1000);
  refreshHistory();
}

async function refreshHistory() {
  const jobs = await api("/api/jobs").catch(() => []);
  const ul = $("history");
  ul.innerHTML = "";
  if (!jobs.length) { ul.innerHTML = '<li class="muted">暂无</li>'; return; }
  for (const j of jobs) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = j.id === currentJob ? "on" : "";
    const t = new Date(j.created * 1000).toLocaleString();
    b.innerHTML = `<span>${escapeHtml(j.name || j.id)}<br><small class="muted">${t}</small></span><span class="st-${j.status}">${STATUS[j.status]}</span>`;
    b.onclick = () => track(j.id);
    li.appendChild(b);
    ul.appendChild(li);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------
function showResult(job) {
  const rep = job.report || {};
  const regions = rep.regions || [];
  $("resultBox").hidden = false;
  $("download").href = `/api/jobs/${job.id}/download`;
  const tags = Object.entries(rep.summary || {}).map(([k, v]) => `<span class="tag">${escapeHtml(k)} × ${v}</span>`);
  const facts = [`修改文件 ${rep.files_modified ?? 0} / ${rep.files ?? 0}`, `耗时 ${rep.seconds ?? "-"} 秒`];
  const warns = (rep.warnings || []).map((w) => `<div class="warn">${escapeHtml(w)}</div>`);
  $("summary").innerHTML = (tags.length ? tags.join("") : '<span class="tag">未发现敏感标志</span>') +
    facts.map((f) => `<span class="muted">${f}</span>`).join(" · ") + warns.join("");
  renderRegions(job.id, regions);
  loadModels(job.id, rep);
}

function renderRegions(jobId, regions) {
  $("regionCount").textContent = regions.length ? `(${regions.length})` : "";
  const box = $("regions");
  box.innerHTML = regions.length ? "" : '<p class="muted">未发现需要去除的敏感标志</p>';
  for (const r of regions) {
    const el = document.createElement("div");
    el.className = "region";
    el.dataset.id = r.id;
    const img = (p) => (p ? `/api/jobs/${jobId}/files/${p}` : "");
    el.innerHTML = `
      <div class="pics">
        <figure><img loading="lazy" src="${img(r.preview_before)}" alt="处理前"><figcaption>处理前</figcaption></figure>
        <figure><img loading="lazy" src="${img(r.preview_after)}" alt="处理后"><figcaption>处理后</figcaption></figure>
      </div>
      <div class="meta">
        <b>#${r.id} ${escapeHtml(r.category_label)}${r.label ? " · " + escapeHtml(r.label) : ""}</b>
        <span>置信度</span><span>${(r.score * 100).toFixed(0)}%</span>
        <span>安装方式</span><span>${MOUNT[r.mount] || r.mount}</span>
        <span>尺寸</span><span>${r.width.toFixed(2)} × ${r.height.toFixed(2)} m</span>
        <span>位置</span><span>${r.center.map((v) => v.toFixed(2)).join(", ")}</span>
      </div>`;
    el.onclick = () => focusRegion(r.id);
    box.appendChild(el);
  }
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
let showing = "after";
let regionData = [];
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

// working frame (Z up) -> viewer (Y up), relative to the overview origin
const toView = (p) => new THREE.Vector3(p[0] - origin[0], p[2] - origin[2], -(p[1] - origin[1]));
const dirToView = (d) => new THREE.Vector3(d[0], d[2], -d[1]);

function clearGroup(g) {
  for (const c of [...g.children]) {
    g.remove(c);
    c.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) [].concat(o.material).forEach((m) => { m.map && m.map.dispose(); m.dispose(); });
    });
  }
}

function asUnlit(root) {
  // textures of reality meshes already contain the lighting
  root.traverse((o) => {
    if (o.isMesh) {
      const map = o.material.map || null;
      o.material = new THREE.MeshBasicMaterial({ map, color: map ? 0xffffff : 0xbbbbbb, side: THREE.DoubleSide });
    }
  });
  return root;
}

async function loadModels(jobId, rep) {
  const ov = rep.overview;
  clearGroup(groups.before); clearGroup(groups.after); clearGroup(groups.markers);
  labels.forEach((l) => l.el.remove());
  labels = [];
  if (!ov) { $("viewerInfo").textContent = "无三维预览"; return; }
  origin = ov.origin;
  regionData = rep.regions || [];
  $("viewerEmpty").hidden = true;
  $("viewerInfo").textContent = "加载模型…";
  const loader = new GLTFLoader();
  const load = (p) => new Promise((res, rej) => loader.load(`/api/jobs/${jobId}/files/${p}`, (g) => res(g.scene), undefined, rej));
  try {
    const [b, a] = await Promise.all([load(ov.before), load(ov.after)]);
    groups.before.add(asUnlit(b));
    groups.after.add(asUnlit(a));
  } catch (e) {
    $("viewerInfo").textContent = "模型加载失败: " + e.message;
    return;
  }
  for (const r of regionData) addMarker(r);
  $("viewerInfo").textContent = `${ov.files} 个文件 · ${(ov.triangles / 1000).toFixed(0)}k 三角形` + (ov.level ? ` · LOD 第 ${ov.level} 级` : "");
  setShowing(showing);
  resetView();
}

function addMarker(r) {
  const n = dirToView(r.normal), u = dirToView(r.axis_u), v = dirToView(r.axis_v);
  const geo = new THREE.BoxGeometry(r.width + 0.1, r.height + 0.1, 0.15);
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0xff3030, depthTest: false }));
  box.renderOrder = 10;
  const m = new THREE.Matrix4().makeBasis(u, v, n);
  box.quaternion.setFromRotationMatrix(m);
  box.position.copy(toView(r.center));
  box.userData.id = r.id;
  groups.markers.add(box);
  const el = document.createElement("div");
  el.className = "marker-label";
  el.textContent = `#${r.id} ${r.label || r.category_label}`;
  wrap.appendChild(el);
  labels.push({ el, pos: toView(r.center).add(new THREE.Vector3(0, r.height / 2 + 0.3, 0)) });
}

function setShowing(which) {
  showing = which;
  groups.before.visible = which === "before";
  groups.after.visible = which === "after";
  $("showBefore").classList.toggle("on", which === "before");
  $("showAfter").classList.toggle("on", which === "after");
  labels.forEach((l) => l.el.classList.toggle("after", which === "after"));
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

function focusRegion(id) {
  const r = regionData.find((x) => x.id === id);
  if (!r) return;
  document.querySelectorAll(".region").forEach((el) => el.classList.toggle("on", Number(el.dataset.id) === id));
  const c = toView(r.center);
  const n = dirToView(r.normal);
  const dist = Math.max(4, r.width * 6);
  controls.target.copy(c);
  camera.position.copy(c).add(n.multiplyScalar(dist)).add(new THREE.Vector3(0, dist * 0.35, 0));
  camera.near = 0.02; camera.far = 2000; camera.updateProjectionMatrix();
  controls.update();
  wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

const tmp = new THREE.Vector3();
function animate() {
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
}
animate();

// ---------------------------------------------------------------------------
// start-up
// ---------------------------------------------------------------------------
api("/api/health").then((h) => {
  $("health").textContent = h.osgb_bridge ? `服务正常 · OSGB 组件已就绪 · v${h.version}` : "服务正常 · 未找到 OSGB 组件 (仅支持 OBJ/GLB)";
  $("health").className = "pill " + (h.osgb_bridge ? "ok" : "bad");
}).catch(() => { $("health").textContent = "无法连接服务"; $("health").className = "pill bad"; });
refreshHistory().then(async () => {
  const jobs = await api("/api/jobs").catch(() => []);
  const last = jobs.find((j) => j.status === "done") || jobs[0];
  if (last) track(last.id);
});
