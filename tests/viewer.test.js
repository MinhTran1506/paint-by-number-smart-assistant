// Logic tests for components/viewer/index.html, run with:  node --test tests/
// The page script runs against a minimal stub DOM, so this checks zoom/pan/click behaviour,
// not browser rendering.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "..", "components", "viewer", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

class El {
  constructor() {
    this.handlers = {}; this.style = {}; this.textContent = ""; this.classes = new Set();
    this.clientWidth = 1000; this.clientHeight = 800;
    this.classList = {
      toggle: (c, on) => (on ? this.classes.add(c) : this.classes.delete(c)),
      add: (c) => this.classes.add(c), remove: (c) => this.classes.delete(c),
    };
  }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  setPointerCapture() {}
  fire(type, ev = {}) {
    const e = { button: 0, pointerId: 1, preventDefault() {}, ...ev };
    (this.handlers[type] || []).forEach((fn) => fn(e));
  }
}

function boot() {
  const els = Object.fromEntries(["stage", "img", "zoom", "zin", "zout", "fit"].map((k) => [k, new El()]));
  const sent = [];
  const winHandlers = {};
  const window = {
    parent: { postMessage: (m) => sent.push(m) },
    addEventListener: (t, fn) => (winHandlers[t] ||= []).push(fn),
  };
  vm.runInNewContext(script, { document: { getElementById: (id) => els[id] }, window });
  const render = (args) => winHandlers.message.forEach((fn) =>
    fn({ data: { type: "streamlit:render", args } }));
  const state = () => {
    const m = els.img.style.transform.match(/translate\(([-\d.e]+)px,([-\d.e]+)px\) scale\(([-\d.e]+)\)/);
    return { tx: +m[1], ty: +m[2], z: +m[3], label: els.zoom.textContent };
  };
  const clicks = () => sent.filter((m) => m.type === "streamlit:setComponentValue");
  return { els, sent, render, state, clicks, resize: () => winHandlers.resize.forEach((f) => f()) };
}

// 1600 x 2000 image in a 1000 x 800 stage: fit scale = 0.4, centred horizontally.
const ARGS = { src: "data:a", img_w: 1600, img_h: 2000, view_id: "v1", height: 800 };

test("announces itself and sizes its frame", () => {
  const v = boot();
  assert.equal(v.sent[0].type, "streamlit:componentReady");
  v.render(ARGS);
  assert.ok(v.sent.some((m) => m.type === "streamlit:setFrameHeight" && m.height === 800));
  assert.equal(v.els.img.src, "data:a");
});

test("starts fitted to the view and centred", () => {
  const v = boot(); v.render(ARGS);
  const s = v.state();
  assert.ok(Math.abs(s.z - 0.4) < 1e-9);
  assert.equal(s.label, "100%");
  assert.ok(Math.abs(s.tx - 180) < 1e-9 && Math.abs(s.ty) < 1e-9);
});

test("wheel zooms in toward the cursor, keeping the point under it fixed", () => {
  const v = boot(); v.render(ARGS);
  const cx = 600, cy = 300;
  const before = v.state();
  const imgX = (cx - before.tx) / before.z, imgY = (cy - before.ty) / before.z;
  v.els.stage.fire("wheel", { clientX: cx, clientY: cy, deltaY: -400, deltaMode: 0 });
  const after = v.state();
  assert.ok(after.z > before.z * 1.5);
  assert.ok(Math.abs((cx - after.tx) / after.z - imgX) < 1e-6);
  assert.ok(Math.abs((cy - after.ty) / after.z - imgY) < 1e-6);
  assert.notEqual(after.label, "100%");
});

test("wheel zooms back out and is bounded", () => {
  const v = boot(); v.render(ARGS);
  for (let i = 0; i < 40; i++) v.els.stage.fire("wheel", { clientX: 500, clientY: 400, deltaY: -500, deltaMode: 0 });
  assert.equal(v.state().label, "1600%");
  for (let i = 0; i < 80; i++) v.els.stage.fire("wheel", { clientX: 500, clientY: 400, deltaY: 500, deltaMode: 0 });
  assert.equal(v.state().label, "50%");
});

test("+, - and fit buttons work", () => {
  const v = boot(); v.render(ARGS);
  v.els.zin.fire("click"); assert.equal(v.state().label, "150%");
  v.els.zout.fire("click"); assert.equal(v.state().label, "100%");
  v.els.zin.fire("click"); v.els.zin.fire("click");
  v.els.fit.fire("click"); assert.equal(v.state().label, "100%");
});

test("dragging pans the picture and does not select anything", () => {
  const v = boot(); v.render(ARGS);
  v.els.zin.fire("click"); v.els.zin.fire("click");
  const before = v.state();
  v.els.stage.fire("pointerdown", { clientX: 500, clientY: 400 });
  v.els.stage.fire("pointermove", { clientX: 450, clientY: 370 });
  v.els.stage.fire("pointerup", { clientX: 450, clientY: 370 });
  const after = v.state();
  assert.ok(Math.abs(after.tx - (before.tx - 50)) < 1e-6);  // follows the pointer 1:1
  assert.ok(Math.abs(after.ty - (before.ty - 30)) < 1e-6);
  assert.equal(after.z, before.z);
  assert.equal(v.clicks().length, 0);
});

test("a click reports the image pixel under the cursor", () => {
  const v = boot(); v.render(ARGS);
  const s = v.state();  // z = 0.4, tx = 180, ty = 0
  v.els.stage.fire("pointerdown", { clientX: 580, clientY: 200 });
  v.els.stage.fire("pointerup", { clientX: 580, clientY: 200 });
  const [c] = v.clicks();
  assert.equal(c.value.x, Math.floor((580 - s.tx) / s.z));
  assert.equal(c.value.y, Math.floor((200 - s.ty) / s.z));
  assert.equal(c.value.view, "v1");
  assert.equal(c.dataType, "json");
});

test("a click while zoomed and panned maps through the transform", () => {
  const v = boot(); v.render(ARGS);
  v.els.stage.fire("wheel", { clientX: 600, clientY: 300, deltaY: -600, deltaMode: 0 });
  v.els.stage.fire("pointerdown", { clientX: 300, clientY: 300 });
  v.els.stage.fire("pointermove", { clientX: 340, clientY: 330 });
  v.els.stage.fire("pointerup", { clientX: 340, clientY: 330 });
  const s = v.state();
  v.els.stage.fire("pointerdown", { clientX: 420, clientY: 360 });
  v.els.stage.fire("pointerup", { clientX: 420, clientY: 360 });
  const [c] = v.clicks();
  assert.equal(c.value.x, Math.floor((420 - s.tx) / s.z));
  assert.equal(c.value.y, Math.floor((360 - s.ty) / s.z));
});

test("tiny pointer jitter still counts as a click; repeated clicks get new ids", () => {
  const v = boot(); v.render(ARGS);
  for (let i = 0; i < 2; i++) {
    v.els.stage.fire("pointerdown", { clientX: 500, clientY: 300 });
    v.els.stage.fire("pointermove", { clientX: 502, clientY: 301 });
    v.els.stage.fire("pointerup", { clientX: 502, clientY: 301 });
  }
  const cs = v.clicks();
  assert.equal(cs.length, 2);
  assert.notEqual(cs[0].value.n, cs[1].value.n);
});

test("clicking the empty margin around the picture selects nothing", () => {
  const v = boot(); v.render(ARGS);  // picture spans x 180..820
  v.els.stage.fire("pointerdown", { clientX: 50, clientY: 300 });
  v.els.stage.fire("pointerup", { clientX: 50, clientY: 300 });
  assert.equal(v.clicks().length, 0);
});

test("a new frame for the same image keeps the user's zoom and pan", () => {
  const v = boot(); v.render(ARGS);
  v.els.stage.fire("wheel", { clientX: 600, clientY: 300, deltaY: -600, deltaMode: 0 });
  const before = v.state();
  v.render({ ...ARGS, src: "data:b" });
  assert.equal(v.els.img.src, "data:b");
  assert.deepEqual(v.state(), before);
});

test("a different image (new view id) re-fits the view", () => {
  const v = boot(); v.render(ARGS);
  v.els.zin.fire("click");
  v.render({ ...ARGS, src: "data:c", view_id: "v2" });
  assert.equal(v.state().label, "100%");
});

test("double-click returns to the fitted view", () => {
  const v = boot(); v.render(ARGS);
  v.els.zin.fire("click"); v.els.zin.fire("click");
  v.els.stage.fire("dblclick");
  assert.equal(v.state().label, "100%");
});

test("resizing while fitted refits to the new size", () => {
  const v = boot(); v.render(ARGS);
  v.els.stage.clientWidth = 500;
  v.resize();
  const s = v.state();
  assert.ok(Math.abs(s.z - 0.3125) < 1e-9);  // min(500/1600, 800/2000)
  assert.equal(s.label, "100%");
});

test("resizing while zoomed keeps the zoom relative to fit", () => {
  const v = boot(); v.render(ARGS);
  v.els.zin.fire("click"); v.els.zin.fire("click");
  assert.equal(v.state().label, "225%");
  v.els.stage.clientWidth = 500;
  v.resize();
  assert.equal(v.state().label, "225%");
});

// ---- focus: jump to a region (used by "next unpainted region") -------------------------
const focusOf = (x, y, w, h, n) => ({ ...ARGS, focus: { x, y, w, h, n } });
const screenOf = (s, x, y) => ({ x: s.tx + x * s.z, y: s.ty + y * s.z });

test("focus centres a small region and zooms in on it", () => {
  const v = boot(); v.render(ARGS);
  v.render(focusOf(1200, 600, 10, 10, 1));
  const s = v.state(), p = screenOf(s, 1200, 600);
  assert.ok(Math.abs(s.z - 0.4 * 16) < 1e-9);  // capped at 16x the fitted view
  assert.ok(Math.abs(p.x - 500) < 1e-6 && Math.abs(p.y - 400) < 1e-6);
});

test("focus on a large region zooms only as far as needed to show all of it", () => {
  const v = boot(); v.render(ARGS);
  for (let i = 0; i < 4; i++) v.els.zin.fire("click");   // start zoomed in, so it must zoom out
  v.render(focusOf(800, 1000, 1200, 1500, 1));
  const s = v.state();
  assert.ok(s.z >= 0.4 - 1e-9);
  assert.ok(1200 * s.z <= 1000 + 1e-6 && 1500 * s.z <= 800 + 1e-6);
});

test("re-rendering with the same focus does not snap the view back", () => {
  const v = boot(); v.render(focusOf(800, 1000, 10, 10, 7));
  v.els.stage.fire("wheel", { clientX: 300, clientY: 300, deltaY: 300, deltaMode: 0 });
  v.els.stage.fire("pointerdown", { clientX: 500, clientY: 400 });
  v.els.stage.fire("pointermove", { clientX: 450, clientY: 380 });
  v.els.stage.fire("pointerup", { clientX: 450, clientY: 380 });
  const before = v.state();
  v.render({ ...focusOf(800, 1000, 10, 10, 7), src: "data:next" });
  assert.deepEqual(v.state(), before);
});

test("a new focus request moves the view again", () => {
  const v = boot(); v.render(focusOf(800, 1000, 10, 10, 1));
  v.render(focusOf(200, 300, 10, 10, 2));
  const p = screenOf(v.state(), 200, 300);
  assert.ok(Math.abs(p.x - 500) < 1e-6 && Math.abs(p.y - 400) < 1e-6);
});

test("focus keeps the current zoom when the region is already comfortably visible", () => {
  const v = boot(); v.render(ARGS);
  for (let i = 0; i < 4; i++) v.els.zin.fire("click");   // 506% of fit
  const z = v.state().z;
  v.render(focusOf(1200, 600, 10, 10, 1));
  assert.equal(v.state().z, z);
  const p = screenOf(v.state(), 1200, 600);
  assert.ok(Math.abs(p.x - 500) < 1e-6 && Math.abs(p.y - 400) < 1e-6);
});

test("focus near an image corner still brings the region into view", () => {
  const v = boot(); v.render(ARGS);
  for (let i = 0; i < 4; i++) v.els.zin.fire("click");
  v.render(focusOf(5, 5, 8, 8, 1));
  const p = screenOf(v.state(), 5, 5);
  assert.ok(p.x >= 0 && p.x <= 1000 && p.y >= 0 && p.y <= 800);
});

test("clicking right after a focus jump maps through the new view", () => {
  const v = boot(); v.render(focusOf(1200, 600, 10, 10, 1));
  v.els.stage.fire("pointerdown", { clientX: 500, clientY: 400 });  // the centre of the view
  v.els.stage.fire("pointerup", { clientX: 500, clientY: 400 });
  const [c] = v.clicks();
  assert.ok(Math.abs(c.value.x - 1200) <= 1 && Math.abs(c.value.y - 600) <= 1);
});
