'use strict';
/* ReliaGraph interactive graph engine.
 * Live force simulation on <canvas>: draggable nodes, cluster halos, hover focus (neighbours light up,
 * the rest fades), animated camera (zoom-to-node / zoom-to-cluster), answer-path highlighting.
 * Loaded BEFORE app.js; it only touches `state`, `$`, `renderSelectedNode`, `currentFilteredGraph`
 * at call time, so load order is safe. */

const GV = {
  nodes: [], edges: [], byId: new Map(), adj: new Map(), groups: new Map(),
  pos: new Map(),                  // id -> {x,y} persists across rebuilds, so the graph grows smoothly
  sig: '', alpha: 0, raf: 0, dirty: true, w: 0, h: 0,
  cam: { x: 0, y: 0, k: 1 }, tween: null, autoFit: true,
  hover: null, drag: null, pan: null, focusGroup: null, t: 0,
};
const GV_PALETTE = ['#7c8cff', '#4fd1c5', '#f6ad55', '#f56f9a', '#68d391', '#63b3ed', '#d69e2e', '#b794f4', '#fc8181', '#76e4f7'];

function gvHash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function gvEdgeKey(a, b) { return a < b ? a + '|' + b : b + '|' + a; }

/* ---------- build model from the filtered graph ---------- */
function gvBuild(g) {
  const ids = new Set(g.nodes.map(n => n.node_id));
  const old = GV.byId;
  GV.nodes = g.nodes.map(n => {
    const prev = GV.pos.get(n.node_id) || old.get(n.node_id);
    return { id: n.node_id, raw: n, label: (n.label || n.node_id).trim(), type: n.node_type || 'entity',
             x: prev ? prev.x : NaN, y: prev ? prev.y : NaN, vx: 0, vy: 0, deg: 0, group: null, pinned: false };
  });
  GV.byId = new Map(GV.nodes.map(n => [n.id, n]));
  GV.edges = g.edges.filter(e => ids.has(e.source_node_id) && ids.has(e.target_node_id) && e.source_node_id !== e.target_node_id)
    .map(e => ({ a: GV.byId.get(e.source_node_id), b: GV.byId.get(e.target_node_id), raw: e,
                 derived: e.status === 'derived' || e.relation_type === 'contains_concept',
                 conflict: e.status === 'conflict', bend: ((gvHash(e.source_node_id + e.target_node_id) % 100) / 100 - .5) * .35 }));
  GV.adj = new Map(GV.nodes.map(n => [n.id, new Set()]));
  GV.edges.forEach(e => { GV.adj.get(e.a.id).add(e.b.id); GV.adj.get(e.b.id).add(e.a.id); e.a.deg++; e.b.deg++; });

  // cluster = topic. Topic nodes own their id; entities join the topic they are linked to.
  const topicIds = new Set(GV.nodes.filter(n => n.type === 'topic').map(n => n.id));
  GV.nodes.forEach(n => { if (topicIds.has(n.id)) n.group = n.id; });
  GV.edges.forEach(e => {
    if (topicIds.has(e.a.id) && !e.b.group) e.b.group = e.a.id;
    else if (topicIds.has(e.b.id) && !e.a.group) e.a.group = e.b.id;
  });
  GV.nodes.forEach(n => { if (!n.group) n.group = (n.raw.topic_ids || [])[0] || 'none'; });
  GV.groups = new Map();
  GV.nodes.forEach(n => { if (!GV.groups.has(n.group)) GV.groups.set(n.group, { id: n.group, nodes: [], color: '', label: '' }); GV.groups.get(n.group).nodes.push(n); });
  [...GV.groups.values()].sort((a, b) => b.nodes.length - a.nodes.length).forEach((gr, i) => {
    gr.color = gr.id === 'none' ? '#7891b3' : GV_PALETTE[i % GV_PALETTE.length];
    const t = GV.byId.get(gr.id);
    gr.label = t ? t.label : (gr.id === 'none' ? 'Unclustered' : String(gr.id));
    gr.nodes.forEach(n => { n.color = gr.color; });
  });
  // place newcomers near an already-positioned neighbour (graph "grows"), else on a loose ring
  const cx = GV.w / 2, cy = GV.h / 2;
  GV.nodes.forEach(n => {
    if (Number.isFinite(n.x)) return;
    let anchor = null;
    for (const id of GV.adj.get(n.id)) { const m = GV.byId.get(id); if (m && Number.isFinite(m.x)) { anchor = m; break; } }
    const a = (gvHash(n.id) % 6283) / 1000;
    if (anchor) { n.x = anchor.x + Math.cos(a) * 40; n.y = anchor.y + Math.sin(a) * 40; }
    else { const r = 60 + (gvHash(n.id + 'r') % 260); n.x = cx + Math.cos(a) * r; n.y = cy + Math.sin(a) * r; }
  });
  GV.nodes.forEach(n => { n.r = n.type === 'topic' ? 11 + Math.min(8, Math.sqrt(n.deg)) : 5 + Math.min(9, Math.sqrt(n.deg) * 1.8); });
}

/* ---------- physics ---------- */
function gvTick() {
  const N = GV.nodes, a = GV.alpha;
  if (!N.length) return;
  for (let i = 0; i < N.length; i++) {            // repulsion (n is capped near 140, so O(n^2) is fine)
    const p = N[i];
    for (let j = i + 1; j < N.length; j++) {
      const q = N[j]; let dx = q.x - p.x, dy = q.y - p.y, d2 = dx * dx + dy * dy;
      if (d2 > 160000) continue;
      if (d2 < 1) { dx = Math.random() - .5; dy = Math.random() - .5; d2 = 1; }
      const d = Math.sqrt(d2), f = (900 + (p.r + q.r) * 40) / d2 * a, fx = dx / d * f, fy = dy / d * f;
      p.vx -= fx; p.vy -= fy; q.vx += fx; q.vy += fy;
      const min = p.r + q.r + 6;                  // hard collision so circles never overlap
      if (d < min) { const push = (min - d) * .5 * .35, ux = dx / d * push, uy = dy / d * push; p.x -= ux; p.y -= uy; q.x += ux; q.y += uy; }
    }
  }
  for (const e of GV.edges) {                     // springs
    const dx = e.b.x - e.a.x, dy = e.b.y - e.a.y, d = Math.hypot(dx, dy) || .01;
    const ideal = e.derived ? 95 : 70, f = (d - ideal) * (e.derived ? .035 : .06) * a;
    const fx = dx / d * f, fy = dy / d * f;
    e.a.vx += fx; e.a.vy += fy; e.b.vx -= fx; e.b.vy -= fy;
  }
  const cx = GV.w / 2, cy = GV.h / 2;
  const cen = new Map();                          // cluster cohesion: topics visibly gather into islands
  GV.groups.forEach(gr => { let sx = 0, sy = 0; gr.nodes.forEach(n => { sx += n.x; sy += n.y; }); cen.set(gr.id, { x: sx / gr.nodes.length, y: sy / gr.nodes.length }); });
  for (const n of N) {
    const c = cen.get(n.group);
    if (c && GV.groups.get(n.group).nodes.length > 1) { n.vx += (c.x - n.x) * .045 * a; n.vy += (c.y - n.y) * .045 * a; }
    n.vx += (cx - n.x) * .012 * a; n.vy += (cy - n.y) * .012 * a;
    if (n.pinned) { n.vx = 0; n.vy = 0; continue; }
    n.vx *= .82; n.vy *= .82; n.x += n.vx; n.y += n.vy;
  }
  GV.alpha *= .985;
  if (GV.alpha < .004) { GV.alpha = 0; GV.nodes.forEach(n => GV.pos.set(n.id, { x: n.x, y: n.y })); }
}

/* ---------- camera ---------- */
function gvBounds(list) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  list.forEach(n => { x0 = Math.min(x0, n.x - n.r); y0 = Math.min(y0, n.y - n.r); x1 = Math.max(x1, n.x + n.r); y1 = Math.max(y1, n.y + n.r); });
  return { x0, y0, x1, y1 };
}
function gvFitTo(list, pad = 70, maxK = 2.2) {
  if (!list.length) return null;
  const b = gvBounds(list), w = Math.max(80, b.x1 - b.x0), h = Math.max(80, b.y1 - b.y0);
  const k = Math.min(maxK, (GV.w - pad * 2) / w, (GV.h - pad * 2) / h);
  return { k, x: GV.w / 2 - (b.x0 + b.x1) / 2 * k, y: GV.h / 2 - (b.y0 + b.y1) / 2 * k };
}
function gvGo(target, ms = 650) {
  if (!target) return;
  GV.tween = { from: { ...GV.cam }, to: target, t0: performance.now(), ms };
  GV.dirty = true; gvLoop();
}
function gvScreen(p) { return { x: p.x * GV.cam.k + GV.cam.x, y: p.y * GV.cam.k + GV.cam.y }; }
function gvWorld(sx, sy) { return { x: (sx - GV.cam.x) / GV.cam.k, y: (sy - GV.cam.y) / GV.cam.k }; }
function gvPick(sx, sy) {
  const w = gvWorld(sx, sy); let best = null, bd = 1e9;
  for (const n of GV.nodes) { const d = Math.hypot(n.x - w.x, n.y - w.y); if (d < n.r + 5 / GV.cam.k && d < bd) { bd = d; best = n; } }
  return best;
}

/* ---------- public actions ---------- */
function gvFitAll() { GV.autoFit = true; GV.focusGroup = null; gvGo(gvFitTo(GV.nodes)); }
function gvZoomBy(f) { GV.autoFit = false; const k = Math.max(.2, Math.min(4, GV.cam.k * f)), cx = GV.w / 2, cy = GV.h / 2; gvGo({ k, x: cx - (cx - GV.cam.x) / GV.cam.k * k, y: cy - (cy - GV.cam.y) / GV.cam.k * k }, 220); }
function gvFocusNode(id) {
  const n = GV.byId.get(id); if (!n) return;
  GV.autoFit = false;
  const hood = [n, ...[...GV.adj.get(id)].map(i => GV.byId.get(i))];
  gvGo(gvFitTo(hood, 120, 2.0));
}
function gvFocusGroup(gid) {
  const gr = GV.groups.get(gid); if (!gr) return;
  GV.autoFit = false; GV.focusGroup = gid; gvGo(gvFitTo(gr.nodes, 90, 1.8)); GV.dirty = true;
}
function gvHighlightAnswer(h) {                    // h = {nodes:Set(ids), edges:Set(keys)} or null
  state.graphHighlight = h; GV.sig = ''; drawMainGraph();
  if (h) setTimeout(() => { const list = GV.nodes.filter(n => h.nodes.has(n.id)); GV.autoFit = false; gvGo(gvFitTo(list, 110, 1.9)); }, 450);
  const bar = $('graph-banner');
  if (bar) { bar.classList.toggle('hidden', !h); if (h) $('graph-banner-text').textContent = `Evidence for your last answer: ${h.nodes.size} node(s), ${h.edges.size} relationship(s)`; }
}

/* ---------- rendering ---------- */
function gvDraw() {
  const c = GV.canvas, ctx = GV.ctx; if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, GV.w, GV.h);
  ctx.save(); ctx.translate(GV.cam.x, GV.cam.y); ctx.scale(GV.cam.k, GV.cam.k);
  const k = GV.cam.k, hl = state.graphHighlight, sel = state.selectedNodeId;
  const focusId = GV.hover ? GV.hover.id : sel;                    // hover wins, otherwise selection
  const hood = focusId ? new Set([focusId, ...(GV.adj.get(focusId) || [])]) : null;
  const fg = GV.focusGroup;
  const lit = n => (hl ? hl.nodes.has(n.id) : hood ? hood.has(n.id) : fg ? n.group === fg : true);

  // cluster halos
  GV.groups.forEach(gr => {
    if (gr.nodes.length < 3) return;
    let sx = 0, sy = 0; gr.nodes.forEach(n => { sx += n.x; sy += n.y; }); sx /= gr.nodes.length; sy /= gr.nodes.length;
    let r = 0; gr.nodes.forEach(n => { r = Math.max(r, Math.hypot(n.x - sx, n.y - sy) + n.r); }); r += 34;
    const g = ctx.createRadialGradient(sx, sy, r * .15, sx, sy, r);
    g.addColorStop(0, gr.color + '26'); g.addColorStop(1, gr.color + '00');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(sx, sy, r, 0, 7); ctx.fill();
    gr._c = { x: sx, y: sy, r };
  });

  // edges (curved)
  for (const e of GV.edges) {
    const on = hl ? hl.edges.has(gvEdgeKey(e.a.id, e.b.id)) : hood ? (hood.has(e.a.id) && hood.has(e.b.id) && (e.a.id === focusId || e.b.id === focusId)) : false;
    const dim = (hl || hood) && !on;
    const mx = (e.a.x + e.b.x) / 2, my = (e.a.y + e.b.y) / 2, dx = e.b.x - e.a.x, dy = e.b.y - e.a.y;
    ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.quadraticCurveTo(mx - dy * e.bend, my + dx * e.bend, e.b.x, e.b.y);
    ctx.strokeStyle = on ? (hl ? '#ffd166' : '#9fd0ff') : e.conflict ? '#ff778a' : e.derived ? '#5c6aa0' : '#4a6a93';
    ctx.globalAlpha = dim ? .05 : on ? .95 : (e.derived ? .16 : .3);
    ctx.lineWidth = (on ? 2.2 : e.derived ? .7 : 1) / Math.sqrt(k);
    ctx.stroke();
    if (on && k > .9) {                            // relation label on active edges
      ctx.globalAlpha = 1; ctx.font = `${10 / Math.sqrt(k)}px Inter, system-ui`; ctx.fillStyle = '#d9e8fb'; ctx.textAlign = 'center';
      ctx.fillText((e.raw.relation_type || '').replace(/_/g, ' '), mx - dy * e.bend / 2, my + dx * e.bend / 2 - 4 / k); ctx.textAlign = 'left';
    }
  }
  ctx.globalAlpha = 1;

  // nodes
  const labelBoxes = [], pulse = (Math.sin(GV.t / 260) + 1) / 2;
  const order = GV.nodes.slice().sort((a, b) => b.deg - a.deg);
  const drawNode = n => {
    const isLit = lit(n), isSel = n.id === sel, isHov = GV.hover && GV.hover.id === n.id;
    ctx.globalAlpha = isLit ? 1 : .13;
    if (isLit && (isSel || isHov || (hl && hl.nodes.has(n.id)))) { ctx.shadowColor = n.color; ctx.shadowBlur = 22; }
    const g = ctx.createRadialGradient(n.x - n.r * .3, n.y - n.r * .3, n.r * .1, n.x, n.y, n.r);
    g.addColorStop(0, '#ffffffcc'); g.addColorStop(.25, n.color); g.addColorStop(1, n.color + 'cc');
    ctx.fillStyle = g; ctx.beginPath();
    if (n.type === 'topic') { for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i + Math.PI / 6; ctx[i ? 'lineTo' : 'moveTo'](n.x + Math.cos(a) * n.r * 1.1, n.y + Math.sin(a) * n.r * 1.1); } ctx.closePath(); }
    else ctx.arc(n.x, n.y, n.r, 0, 7);
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.lineWidth = (isSel ? 2.4 : 1) / Math.sqrt(k); ctx.strokeStyle = isSel ? '#fff' : '#ffffff55'; ctx.stroke();
    if (hl && hl.nodes.has(n.id)) { ctx.globalAlpha = .75 * (1 - pulse); ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 2 / Math.sqrt(k); ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 5 + pulse * 9, 0, 7); ctx.stroke(); }
    ctx.globalAlpha = 1;
  };
  order.slice().reverse().forEach(drawNode);

  // labels with level-of-detail + simple collision avoidance
  ctx.textBaseline = 'middle';
  const showAllAt = k >= 1.15, topN = k >= .75 ? 26 : 10;
  order.forEach((n, i) => {
    const must = n.id === sel || (GV.hover && GV.hover.id === n.id) || (hl && hl.nodes.has(n.id)) || (hood && focusId && hood.has(n.id) && k > .6);
    const want = must || n.type === 'topic' && k > .5 || showAllAt || i < topN;
    if (!want || (!lit(n) && !must)) return;
    const fs = (n.type === 'topic' ? 12 : 10.5) / Math.sqrt(Math.max(k, .8));
    ctx.font = `${n.type === 'topic' || must ? 600 : 500} ${fs}px Inter, system-ui`;
    let t = n.label; const max = must ? 40 : 24; if (t.length > max) t = t.slice(0, max - 1) + '…';
    const tw = ctx.measureText(t).width, bx = n.x + n.r + 5 / k, by = n.y, bh = fs + 7 / k;
    const box = { x0: bx - 4 / k, y0: by - bh / 2, x1: bx + tw + 4 / k, y1: by + bh / 2 };
    if (!must && labelBoxes.some(o => box.x0 < o.x1 && box.x1 > o.x0 && box.y0 < o.y1 && box.y1 > o.y0)) return;
    labelBoxes.push(box);
    ctx.fillStyle = 'rgba(8,16,28,.82)'; ctx.beginPath(); ctx.roundRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0, 5 / k); ctx.fill();
    ctx.fillStyle = must ? '#fff' : '#cfe0f5'; ctx.fillText(t, bx, by + .5 / k);
  });

  // cluster titles when zoomed out
  if (k < .8) GV.groups.forEach(gr => {
    if (!gr._c || gr.nodes.length < 3) return;
    ctx.globalAlpha = .75; ctx.font = `700 ${15 / k}px Inter, system-ui`; ctx.fillStyle = gr.color; ctx.textAlign = 'center';
    ctx.fillText(gr.label.slice(0, 28), gr._c.x, gr._c.y - gr._c.r + 4 / k); ctx.textAlign = 'left'; ctx.globalAlpha = 1;
  });
  ctx.restore();

  // tooltip
  if (GV.hover && !GV.drag) {
    const n = GV.hover, s = gvScreen(n), lines = [n.label, `${n.type} · ${n.deg} connection${n.deg === 1 ? '' : 's'}`];
    ctx.font = '600 12px Inter, system-ui'; const w = Math.max(...lines.map(l => ctx.measureText(l).width)) + 20;
    const x = Math.min(GV.w - w - 8, s.x + n.r * k + 12), y = Math.max(8, s.y - 28);
    ctx.fillStyle = 'rgba(12,24,40,.96)'; ctx.strokeStyle = n.color; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(x, y, w, 46, 8); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.fillText(lines[0].slice(0, 42), x + 10, y + 16);
    ctx.font = '11px Inter, system-ui'; ctx.fillStyle = '#9db4d2'; ctx.fillText(lines[1], x + 10, y + 34);
  }
}

function gvLoop() {
  if (GV.raf) return;
  const step = now => {
    GV.raf = 0; GV.t = now;
    if (!GV.canvas || $('page-graph').classList.contains('hidden')) return;
    if (GV.alpha > 0) {
      gvTick(); gvTick(); GV.dirty = true;
      if (GV.autoFit && GV.alpha > .15 && !GV.tween) { const t = gvFitTo(GV.nodes); if (t) { GV.cam.k += (t.k - GV.cam.k) * .08; GV.cam.x += (t.x - GV.cam.x) * .08; GV.cam.y += (t.y - GV.cam.y) * .08; } }
    }
    if (GV.tween) {
      const T = GV.tween, p = Math.min(1, (now - T.t0) / T.ms), e = 1 - Math.pow(1 - p, 3);
      GV.cam.k = T.from.k + (T.to.k - T.from.k) * e; GV.cam.x = T.from.x + (T.to.x - T.from.x) * e; GV.cam.y = T.from.y + (T.to.y - T.from.y) * e;
      if (p >= 1) GV.tween = null; GV.dirty = true;
    }
    const animating = GV.alpha > 0 || GV.tween || state.graphHighlight;   // highlight rings pulse continuously
    if (GV.dirty || animating) { gvDraw(); GV.dirty = false; }
    if (animating || GV.dirty) GV.raf = requestAnimationFrame(step);
  };
  GV.raf = requestAnimationFrame(step);
}

/* ---------- entry point used by app.js (filters, navigation, data refresh) ---------- */
function drawMainGraph() {
  const canvas = $('main-graph'); if (!canvas || $('page-graph').classList.contains('hidden')) return;
  const rect = canvas.getBoundingClientRect(); if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  if (GV.canvas !== canvas) { GV.canvas = canvas; GV.ctx = canvas.getContext('2d'); gvBind(canvas); }
  if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) { canvas.width = Math.floor(rect.width * dpr); canvas.height = Math.floor(rect.height * dpr); }
  GV.w = rect.width; GV.h = rect.height;
  const g = currentFilteredGraph();
  $('graph-result-count').textContent = `${g.nodes.length} nodes · ${g.edges.length} edges`;
  const hlSig = state.graphHighlight ? state.graphHighlight.nodes.size : 0;
  const sig = g.nodes.map(n => n.node_id).join(',') + '#' + g.edges.length + '#' + hlSig;
  if (sig !== GV.sig) {
    const first = !GV.nodes.length;
    GV.sig = sig; gvBuild(g); GV.alpha = first ? 1 : .6;
    if (first || ($('graph-search').value || '').trim()) GV.autoFit = true;
    gvClusterLegend();
  }
  if (!g.nodes.length) { $('graph-status').textContent = 'No nodes match the current filters'; GV.dirty = true; gvDraw(); return; }
  $('graph-status').textContent = `Showing ${g.nodes.length} of ${(state.graph?.nodes || []).length} nodes · ${GV.groups.size} cluster(s) · drag nodes, scroll to zoom, double-click to focus`;
  GV.dirty = true; gvLoop();
}

function gvClusterLegend() {
  const box = $('graph-clusters'); if (!box) return;
  const rows = [...GV.groups.values()].sort((a, b) => b.nodes.length - a.nodes.length).slice(0, 7);
  box.innerHTML = rows.length > 1 ? '<div class="gc-title">CLUSTERS</div>' + rows.map(r =>
    `<button class="gc-row" data-gid="${escapeHtml(String(r.id))}"><i style="background:${r.color}"></i><span>${escapeHtml(r.label.slice(0, 26))}</span><em>${r.nodes.length}</em></button>`).join('') : '';
  box.querySelectorAll('.gc-row').forEach(b => b.addEventListener('click', () => gvFocusGroup(b.dataset.gid)));
}

/* ---------- pointer interaction ---------- */
function gvBind(canvas) {
  const pt = e => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  canvas.addEventListener('pointerdown', e => {
    const p = pt(e), n = gvPick(p.x, p.y); canvas.setPointerCapture(e.pointerId); GV.tween = null;
    if (n) GV.drag = { n, moved: false, p0: p }; else GV.pan = { p0: p, cam0: { ...GV.cam }, moved: false };
  });
  canvas.addEventListener('pointermove', e => {
    const p = pt(e);
    if (GV.drag) {
      const d = GV.drag; if (Math.hypot(p.x - d.p0.x, p.y - d.p0.y) > 4) d.moved = true;
      if (d.moved) { const w = gvWorld(p.x, p.y); d.n.x = w.x; d.n.y = w.y; d.n.vx = d.n.vy = 0; d.n.pinned = true; GV.alpha = Math.max(GV.alpha, .35); GV.autoFit = false; gvLoop(); }
    } else if (GV.pan) {
      const d = GV.pan; if (Math.hypot(p.x - d.p0.x, p.y - d.p0.y) > 4) d.moved = true;
      if (d.moved) { GV.autoFit = false; GV.cam.x = d.cam0.x + p.x - d.p0.x; GV.cam.y = d.cam0.y + p.y - d.p0.y; GV.dirty = true; gvLoop(); }
    } else {
      const n = gvPick(p.x, p.y);
      if (n !== GV.hover) { GV.hover = n; canvas.style.cursor = n ? 'pointer' : 'grab'; GV.dirty = true; gvLoop(); }
    }
  });
  canvas.addEventListener('pointerup', () => {
    const d = GV.drag, pn = GV.pan; GV.drag = null; GV.pan = null;
    if (d) {
      d.n.pinned = false;
      if (!d.moved) { state.selectedNodeId = d.n.id; renderSelectedNode(); gvFocusNode(d.n.id); }
    } else if (pn && !pn.moved) { state.selectedNodeId = null; GV.focusGroup = null; renderSelectedNode(); }
    GV.dirty = true; gvLoop();
  });
  canvas.addEventListener('pointerleave', () => { if (GV.hover) { GV.hover = null; GV.dirty = true; gvLoop(); } });
  canvas.addEventListener('dblclick', e => { const p = pt(e), n = gvPick(p.x, p.y); if (n) { state.selectedNodeId = n.id; renderSelectedNode(); gvFocusNode(n.id); } else gvFitAll(); });
  canvas.addEventListener('wheel', e => {
    e.preventDefault(); GV.autoFit = false; GV.tween = null; const p = pt(e);
    const f = e.deltaY < 0 ? 1.13 : 1 / 1.13, k = Math.max(.2, Math.min(4, GV.cam.k * f)), w = gvWorld(p.x, p.y);
    GV.cam.k = k; GV.cam.x = p.x - w.x * k; GV.cam.y = p.y - w.y * k; GV.dirty = true; gvLoop();
  }, { passive: false });
}
window.addEventListener('resize', () => { if (GV.canvas) { GV.sig = ''; drawMainGraph(); } });

/* ---------- "Show on graph" for an answer ---------- */
function gvAnswerHighlight(result) {
  const g = state.graph || { nodes: [], edges: [] }, ids = new Set(), edges = new Set();
  const byKey = new Map();
  g.nodes.forEach(n => { byKey.set(String(n.node_id).toLowerCase(), n.node_id); byKey.set(String(n.label || '').toLowerCase(), n.node_id); });
  const find = s => byKey.get(String(s || '').toLowerCase());
  (result.graph_paths || []).forEach(p => {
    const steps = p.steps || [];
    steps.forEach(s => { const a = find(s.source), b = find(s.target); if (a) ids.add(a); if (b) ids.add(b); if (a && b) edges.add(gvEdgeKey(a, b)); });
    if (Array.isArray(p)) p.forEach(n => { const a = find(typeof n === 'string' ? n : (n.node_id || n.label)); if (a) ids.add(a); });
  });
  const chunks = new Set((result.hits || []).map(h => h.chunk_id).filter(Boolean));
  if (chunks.size) g.nodes.forEach(n => { if ((n.evidence_chunk_ids || []).some(c => chunks.has(c))) ids.add(n.node_id); });
  const type = new Map(g.nodes.map(n => [n.node_id, n.node_type]));
  g.edges.forEach(e => {                             // pull in the topic cluster each highlighted node belongs to
    const a = e.source_node_id, b = e.target_node_id;
    if (ids.has(a) && type.get(b) === 'topic') { ids.add(b); edges.add(gvEdgeKey(a, b)); }
    else if (ids.has(b) && type.get(a) === 'topic') { ids.add(a); edges.add(gvEdgeKey(a, b)); }
  });
  return ids.size ? { nodes: ids, edges } : null;
}
