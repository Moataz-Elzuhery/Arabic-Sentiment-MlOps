"""Tiny webhook receiver: prints every alert Alertmanager sends (see `docker compose logs alert_sink`).

Replace it with a Slack/e-mail receiver in monitoring/alertmanager.yml for real notifications.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        for alert in payload.get("alerts", []):
            labels, notes = alert.get("labels", {}), alert.get("annotations", {})
            print(f"[{alert.get('status', '?').upper()}] {labels.get('alertname')} "
                  f"feature={labels.get('feature', '-')} severity={labels.get('severity', '-')} "
                  f"- {notes.get('description') or notes.get('summary', '')}", flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("alert sink listening on :9000", flush=True)
    HTTPServer(("0.0.0.0", 9000), Handler).serve_forever()
