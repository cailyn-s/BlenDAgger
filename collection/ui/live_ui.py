"""Zero-dependency real-time web UI for shared-control blending diagnostics.

Runs a stdlib HTTP server in a background thread and streams the latest
per-DoF-group blend state to a browser via Server-Sent Events (SSE). The
control loop calls :meth:`LiveUIServer.publish` once per step — a non-blocking
dict swap that never touches the socket — so the UI cannot slow the loop.

Layout driven by ``style``:
  * ``"blend"``  (combined_blend + ablations): one tall vertical alpha bar per
    DoF group (full autonomy at top / full human at bottom, coloured with the
    matplotlib RdYlGn ramp), with three thin blue horizontal bars beneath each
    for similarity / confidence / magnitude.
  * ``"simple"`` (hg_dagger): a single vertical alpha bar, binary 0/1, same
    colouring, no sub-bars.

Group names come from the arbitrator via :func:`dof_group_names` so the layout
auto-adapts (e.g. merged ``xyz`` when ``separate_z_dof=False``).

Standalone demo (synthetic data, no robot) to eyeball the visuals::

    python -m collection.ui.live_ui --demo blend       # combined_blend layout
    python -m collection.ui.live_ui --demo hg_dagger    # simplified single-bar layout
"""
import http.server
import json
import math
import threading
import time

# Canonical column index of each DoF-group name inside the (T,3) [xy, z, ori]
# blend-term arrays CombinedBlend always reports. A merged "xyz" group shares
# the "xy" column (CombinedBlend duplicates the value there).
_COL = {"xy": 0, "xyz": 0, "z": 1, "ori": 2}

# Pretty labels for the group headers.
_LABEL = {"xy": "XY", "xyz": "XYZ", "z": "Z", "ori": "Orientation", "control": "Control"}


def dof_group_names(arbitrator):
    """Best-effort list of DoF-group names for the given arbitrator.

    Uses ``arbitrator._dof_groups()`` when available (CombinedBlend family) so
    the UI matches the actual grouping; falls back to the canonical layout.
    """
    fn = getattr(arbitrator, "_dof_groups", None)
    if callable(fn):
        try:
            return list(fn().keys())
        except Exception:
            pass
    return ["xy", "z", "ori"]


def build_state(style, group_names, arb):
    """Turn one ``arbitrate()`` result into a JSON-safe UI state dict.

    NaN confidence (no uncertainty estimator) is emitted as ``None`` so the
    browser can render it as "n/a" — raw NaN would break ``JSON.parse``.
    """
    mode = arb.get("control_mode", "robot")

    if style == "simple":
        # hg_dagger: use the logged alpha directly (0/1, or an in-between value
        # during the EMA onset window). Fall back to control_mode if absent.
        alphas = arb.get("alphas")
        if alphas is not None:
            alpha = float(alphas[0])
        else:
            alpha = 1.0 if mode == "robot" else 0.0
        return {"mode": mode, "groups": [{"name": "control", "alpha": alpha}]}

    # Similarity/confidence are only meaningful while the human is driving; show
    # n/a whenever the robot is in full control. Magnitude stays (reads 0.0).
    robot_mode = (mode == "robot")

    bt = arb.get("blend_terms")
    groups = []
    for name in group_names:
        col = _COL.get(name, 0)
        if bt is not None:
            conf = float(bt["confidence"][col])
            groups.append({
                "name": name,
                "alpha": float(bt["alpha"][col]),
                "similarity": None if robot_mode else float(bt["similarity"][col]),
                "confidence": None if (robot_mode or math.isnan(conf)) else conf,
                "magnitude": float(bt["magnitude"][col]),
            })
        else:
            # Robot-only early return (human inactive): full autonomy, no input.
            groups.append({
                "name": name, "alpha": 1.0,
                "similarity": None, "confidence": None, "magnitude": 0.0,
            })
    return {"mode": mode, "groups": groups}


class LiveUIServer:
    def __init__(self, style="blend", group_names=None, host="127.0.0.1", port=8000):
        self.style = style
        self.group_names = list(group_names) if group_names else ["xy", "z", "ori"]
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._state = {"mode": "robot", "groups": []}
        self._httpd = None
        self._thread = None

    # ---- control-loop side (non-blocking) --------------------------------
    def publish(self, state):
        with self._lock:
            self._state = state

    def publish_arb(self, arb):
        """Convenience: build + publish from a raw arbitrate() result."""
        self.publish(build_state(self.style, self.group_names, arb))

    def _snapshot(self):
        with self._lock:
            return dict(self._state)

    # ---- server side -----------------------------------------------------
    def start(self):
        meta = {"style": self.style,
                "groups": [{"name": n, "label": _LABEL.get(n, n.upper())}
                           for n in self.group_names]}
        html = _PAGE
        ui = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):  # silence per-request stderr spam
                pass

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    body = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/meta":
                    body = json.dumps(meta).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    try:
                        while True:
                            payload = json.dumps(ui._snapshot(), allow_nan=False)
                            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(0.04)  # ~25 Hz
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                else:
                    self.send_error(404)

        self._httpd = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://{self.host}:{self.port}"
        line = f"  LIVE UI:  {url}  "
        bar = "=" * len(line)
        print(f"\n{bar}\n{line}\n{bar}"
              f"\n  (style={self.style}, groups={self.group_names})\n")

    def stop(self):
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


# --------------------------------------------------------------------------
# Single-file frontend. Fetches /meta once to build the layout, then consumes
# the /stream SSE feed. RdYlGn stops (ColorBrewer 11-class == matplotlib RdYlGn)
# are interpolated client-side: alpha 0 -> red, 0.5 -> yellow, 1 -> green.
# --------------------------------------------------------------------------
_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BlenDAgger Live</title>
<style>
  :root { color-scheme: light;
          /* Bars scale with the window; clamp keeps them sane on tiny/huge screens. */
          --bar-h: clamp(180px, 30vh, 480px);
          --bar-w: clamp(44px, 4.3vw, 120px); }
  body { margin: 0; min-height: 100vh; background: #ffffff; color: #1a1d21;
         display: flex; flex-direction: column;
         font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  header { padding: 14px 20px; display: flex; align-items: center; gap: 16px;
           border-bottom: 1px solid #e2e4e8; }
  #titlebox { display: flex; flex-direction: column; gap: 3px; }
  #algo { font-size: 20px; font-weight: 700; letter-spacing: .2px; }
  #mode { font-size: 14px; color: #5a606a; }
  #groups { flex: 1; display: flex; gap: 72px; padding: 26px 30px;
            justify-content: center; align-items: center; flex-wrap: wrap; }
  .group { display: grid; grid-template-columns: 140px var(--bar-w);
           column-gap: 10px; align-items: center; }
  .gname { grid-column: 2; grid-row: 1; justify-self: center; white-space: nowrap;
           font-size: 16px; font-weight: 600; margin-bottom: 10px; }
  .axis { grid-column: 1; grid-row: 2; white-space: nowrap;
          display: flex; flex-direction: column; justify-content: space-between;
          font-size: 13px; font-weight: 600; color: #4a4f57; text-align: right;
          height: var(--bar-h); padding: 2px 0; }
  .bar { grid-column: 2; grid-row: 2; position: relative; width: var(--bar-w);
         height: var(--bar-h); background: #eef0f3;
         border: 1px solid #d5d8de; border-radius: 6px; overflow: hidden; }
  .fill { position: absolute; left: 0; right: 0; bottom: 0;
          transition: height .05s linear, background-color .05s linear; }
  .aval { position: absolute; left: 50%; top: 6px; transform: translateX(-50%);
          font-size: 14px; font-weight: 700; color: #1a1d21;
          background: rgba(255,255,255,.85); padding: 1px 6px; border-radius: 8px; }
  .subs { grid-column: 2; grid-row: 3; justify-self: center; margin-top: 14px;
          width: max(var(--bar-w), 150px);
          display: flex; flex-direction: column; gap: 8px; }
  .sub { font-size: 12px; color: #5a606a; }
  .sub .lab { display: flex; justify-content: space-between; margin-bottom: 2px; }
  .track { height: 10px; background: #eef0f3; border: 1px solid #d5d8de;
           border-radius: 5px; overflow: hidden; }
  .sfill { height: 100%; width: 0%; background: #3a86ff;
           transition: width .05s linear; }
  .na .sfill { background: #c4c9d0; }
  #conn { font-size: 12px; color: #6b7078; margin-left: auto; }
</style>
</head>
<body>
<header>
  <div id="titlebox">
    <div id="algo">—</div>
    <div id="mode">mode: —</div>
  </div>
  <span id="conn">connecting…</span>
</header>
<div id="groups"></div>
<script>
// RdYlGn (ColorBrewer 11-class, == matplotlib RdYlGn), pos 0..1.
const CMAP = [
  [165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,139],
  [255,255,191],[217,239,139],[166,217,106],[102,189,99],[26,152,80],[0,104,55]
];
function ramp(t){
  t = Math.max(0, Math.min(1, t));
  const x = t*(CMAP.length-1), i = Math.floor(x), f = x-i;
  const a = CMAP[i], b = CMAP[Math.min(i+1, CMAP.length-1)];
  const r = Math.round(a[0]+(b[0]-a[0])*f);
  const g = Math.round(a[1]+(b[1]-a[1])*f);
  const bl = Math.round(a[2]+(b[2]-a[2])*f);
  return `rgb(${r},${g},${bl})`;
}
const SUBS = [["similarity","similarity"],["confidence","confidence"],["magnitude","magnitude"]];
let STYLE = "blend";
const els = {};  // group name -> {fill, aval, subs:{key:{track,fill,val}}}

function buildLayout(meta){
  STYLE = meta.style;
  document.getElementById("algo").textContent =
    (STYLE === "simple") ? "HG-DAgger" : "Blended Shared Control";
  const root = document.getElementById("groups");
  root.innerHTML = "";
  for (const g of meta.groups){
    const col = document.createElement("div"); col.className = "group";
    const name = document.createElement("div"); name.className = "gname";
    name.textContent = g.label; col.appendChild(name);

    const axis = document.createElement("div"); axis.className = "axis";
    axis.innerHTML = "<span>Full Autonomy</span><span>Full Human Control</span>";
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("div"); fill.className = "fill";
    const aval = document.createElement("div"); aval.className = "aval";
    bar.appendChild(fill); bar.appendChild(aval);
    col.appendChild(axis); col.appendChild(bar);

    const rec = {fill, aval, subs:{}};
    if (STYLE === "blend"){
      const subs = document.createElement("div"); subs.className = "subs";
      for (const [key,label] of SUBS){
        const s = document.createElement("div"); s.className = "sub";
        const lab = document.createElement("div"); lab.className = "lab";
        const l = document.createElement("span"); l.textContent = label;
        const v = document.createElement("span"); v.textContent = "—";
        lab.appendChild(l); lab.appendChild(v);
        const track = document.createElement("div"); track.className = "track";
        const sfill = document.createElement("div"); sfill.className = "sfill";
        track.appendChild(sfill); s.appendChild(lab); s.appendChild(track);
        subs.appendChild(s);
        rec.subs[key] = {row:s, fill:sfill, val:v};
      }
      col.appendChild(subs);
    }
    root.appendChild(col);
    els[g.name] = rec;
  }
}

function update(state){
  document.getElementById("mode").textContent = "mode: " + (state.mode || "—");
  for (const g of state.groups){
    const rec = els[g.name]; if (!rec) continue;
    const a = g.alpha;
    rec.fill.style.height = (Math.max(a, 0.04)*100) + "%";  // keep a red sliver at 0
    rec.fill.style.backgroundColor = ramp(a);
    rec.aval.textContent = a.toFixed(2);
    if (STYLE !== "blend") continue;
    for (const [key] of SUBS){
      const sr = rec.subs[key]; if (!sr) continue;
      const v = g[key];
      if (v === null || v === undefined){
        sr.row.classList.add("na"); sr.fill.style.width = "100%"; sr.val.textContent = "n/a";
      } else {
        sr.row.classList.remove("na");
        sr.fill.style.width = (Math.max(0, Math.min(1, v))*100) + "%";
        sr.val.textContent = v.toFixed(2);
      }
    }
  }
}

fetch("/meta").then(r => r.json()).then(meta => {
  buildLayout(meta);
  const es = new EventSource("/stream");
  const conn = document.getElementById("conn");
  es.onopen = () => conn.textContent = "● live";
  es.onerror = () => conn.textContent = "○ reconnecting…";
  es.onmessage = e => { try { update(JSON.parse(e.data)); } catch(_){} };
});
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Standalone demo: feed synthetic arbitrate() results so the visuals can be
# checked in a browser without a robot. Run with `python -m collection.ui.live_ui`.
# --------------------------------------------------------------------------
def _demo(style="blend", port=8000, hz=20.0):
    import numpy as np

    if style in ("hg_dagger", "simple"):
        srv = LiveUIServer(style="simple", group_names=["control"], port=port)
    else:
        srv = LiveUIServer(style="blend", group_names=["xy", "z", "ori"], port=port)
    srv.start()
    print(f"[live-ui] demo running (style={style}) — Ctrl+C to stop.")

    dt = 1.0 / hz
    t = 0.0
    ema = 1.0  # hg_dagger demo: smoothed alpha, EMA-eased toward the target
    try:
        while True:
            if style in ("hg_dagger", "simple"):
                # Toggle full autonomy / full human every ~3 s; ease the logged
                # alpha toward the target so the EMA onset window is visible.
                mode = "human" if (int(t) // 3) % 2 == 1 else "robot"
                target = 1.0 if mode == "robot" else 0.0
                ema = 0.85 * ema + 0.15 * target
                arb = {"control_mode": mode,
                       "alphas": np.full(6, ema, np.float32)}
            else:
                # Each group's terms drift on out-of-phase sinusoids in [0, 1];
                # alpha is a smooth product so it plausibly tracks sim*conf.
                def s(phase, speed=0.5):
                    return 0.5 + 0.5 * math.sin(speed * t + phase)
                sim = np.array([s(0.0), s(2.0), s(4.0)], np.float32)
                conf = np.array([s(1.0, 0.7), s(3.0, 0.7), s(5.0, 0.7)], np.float32)
                mag = np.array([s(0.5, 0.9), s(2.5, 0.9), s(4.5, 0.9)], np.float32)
                alpha = (sim * conf).astype(np.float32)
                arb = {"control_mode": "blend",
                       "blend_terms": {"alpha": alpha, "similarity": sim,
                                       "confidence": conf, "magnitude": mag}}
            srv.publish_arb(arb)
            time.sleep(dt)
            t += dt
    except KeyboardInterrupt:
        print("\n[live-ui] demo stopped.")
    finally:
        srv.stop()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Live-UI synthetic demo (no robot).")
    p.add_argument("--demo", choices=["blend", "hg_dagger"], default="blend",
                   help="Which layout to preview (default: blend).")
    p.add_argument("--port", type=int, default=8000, help="Server port (default: 8000).")
    p.add_argument("--hz", type=float, default=20.0, help="Update rate (default: 20).")
    a = p.parse_args()
    _demo(style=a.demo, port=a.port, hz=a.hz)
