"""
main.py — Orchestrator + Flask alert dashboard.

Two modes:
  python main.py --mode inference          # Start live camera pipeline
  python main.py --mode dashboard          # Start Flask alert dashboard only
  python main.py --mode both               # Start both (default)
"""

import argparse
import threading

from flask import Flask, jsonify, render_template_string

import config
import utils
from inference import run_inference
from logger import get_logger

log = get_logger(__name__)

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Flask dashboard — minimal HTML served from a single file (no templates dir)
# ─────────────────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paralysis Monitor — Alerts</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
    header { background: #1e293b; padding: 1rem 2rem;
             display: flex; align-items: center; gap: 1rem; }
    header h1 { font-size: 1.2rem; font-weight: 600; }
    .badge { background: #ef4444; color: #fff; border-radius: 999px;
             font-size: .75rem; padding: .15rem .55rem; }
    main { padding: 2rem; }
    table { width: 100%; border-collapse: collapse; font-size: .875rem; }
    th { text-align: left; padding: .5rem .75rem; background: #1e293b;
         color: #94a3b8; font-weight: 500; }
    td { padding: .5rem .75rem; border-bottom: 1px solid #1e293b; }
    tr:hover td { background: #1e293b44; }
    .tag { border-radius: 4px; font-size: .75rem; padding: .1rem .4rem; }
    .inactivity         { background: #7c3aed22; color: #c4b5fd; }
    .distress_expression{ background: #dc262622; color: #fca5a5; }
    .sustained_lip_activity { background: #0284c722; color: #7dd3fc; }
    #refresh-note { color: #64748b; font-size: .75rem; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <header>
    <h1>Paralysis Monitor</h1>
    <span class="badge" id="alert-count">–</span>
    <span style="color:#94a3b8;font-size:.8rem">Auto-refreshes every 5 s</span>
  </header>
  <main>
    <p id="refresh-note">Last fetched: –</p>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Timestamp (UTC)</th><th>Camera</th>
          <th>Track ID</th><th>Event</th><th>Confidence</th>
        </tr>
      </thead>
      <tbody id="events-body"></tbody>
    </table>
  </main>
  <script>
    async function load() {
      const res  = await fetch('/api/events');
      const data = await res.json();
      document.getElementById('alert-count').textContent = data.length;
      document.getElementById('refresh-note').textContent =
        'Last fetched: ' + new Date().toLocaleTimeString();
      const tbody = document.getElementById('events-body');
      tbody.innerHTML = data.map((e, i) => `
        <tr>
          <td style="color:#64748b">${i + 1}</td>
          <td>${e.timestamp}</td>
          <td>${e.camera_id}</td>
          <td>${e.track_id ?? '–'}</td>
          <td><span class="tag ${e.event_type}">${e.event_type}</span></td>
          <td>${e.confidence != null ? (e.confidence * 100).toFixed(1) + '%' : '–'}</td>
        </tr>
      `).join('');
    }
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(_DASHBOARD_HTML)


@app.route("/api/events")
def api_events():
    """Return the 200 most recent events as JSON."""
    events = utils.fetch_events(limit=200)
    return jsonify(events)


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _start_dashboard() -> None:
    log.info("Dashboard starting at http://%s:%d", config.FLASK_HOST, config.FLASK_PORT)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False,   # Must be False when run from a thread
    )


def _start_inference() -> None:
    run_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paralysis monitor orchestrator.")
    parser.add_argument(
        "--mode",
        choices=["inference", "dashboard", "both"],
        default="both",
        help="What to start (default: both).",
    )
    args = parser.parse_args()

    utils.init_db()

    if args.mode == "dashboard":
        _start_dashboard()

    elif args.mode == "inference":
        _start_inference()

    else:  # both
        # Dashboard runs in a daemon thread; inference runs in the main thread
        # so Ctrl-C cleanly shuts everything down.
        t = threading.Thread(target=_start_dashboard, daemon=True)
        t.start()
        log.info("Dashboard thread started. Starting inference …")
        _start_inference()
