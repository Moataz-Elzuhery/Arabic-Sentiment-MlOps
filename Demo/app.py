"""Gradio demo for Arabic-Sentiment-MlOps.
 
A thin client for the running FastAPI service (default http://localhost:8000).
It does not load the model itself, so every prediction goes through the same
API that Prometheus and Grafana observe.
 
Run:  pip install -r requirements-demo.txt
      python app.py        then open http://127.0.0.1:7860
"""
from __future__ import annotations
 
import csv
import html
import os
import random
import re
import tempfile
import time
from collections import Counter
 
import gradio as gr
import requests
 
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CANARY_URL = os.getenv("CANARY_URL", "http://localhost:8080").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3001")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
REPO_URL = "https://github.com/Moataz-Elzuhery/Arabic-Sentiment-MlOps"
 
TIMEOUT = 30           # seconds per request
MAX_CHARS = 2000       # matches PredictRequest.max_length in src/api.py
MAX_BATCH = 50         # reviews per batch run
HISTORY_LEN = 8        # rows kept in the "recent predictions" table
LOW_CONFIDENCE = 0.60  # below this, show a hint under the result
NO_RATING = -1         # value of the "No rating" option in the star selector
 
LABELS = {
    "positive": {"name": "Positive", "ar": "إيجابي", "css": "pos"},
    "negative": {"name": "Negative", "ar": "سلبي", "css": "neg"},
    "neutral": {"name": "Neutral", "ar": "محايد", "css": "neu"},
}
SCORE_ORDER = ["positive", "neutral", "negative"]
 
 
def stars_to_label(stars: int) -> str:
    """The rule the training labels were built with: 0-2 negative, 3 neutral, 4-5 positive."""
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"
 
 
def stars_text(stars: int) -> str:
    return "★" * stars + "☆" * (5 - stars)
 
 
# A batch line may start with a star rating: "4 | review text" (a tab works instead of the bar).
_RATED_LINE = re.compile(r"^\s*([0-5])\s*[|\t]\s*(.+)$")
 
 
def parse_batch_line(line: str) -> tuple[int | None, str]:
    m = _RATED_LINE.match(line)
    return (int(m.group(1)), m.group(2).strip()) if m else (None, line.strip())
 
 
EXAMPLES = [
    "الفندق ممتاز جدا والموظفين محترمين والنظافة رائعة",
    "الغرفة قذرة والخدمة سيئة جدا ولن أعود مرة أخرى",
    "الموقع جيد لكن الغرف صغيرة والفطور عادي",
    "تجربة رائعة، أنصح الجميع بالإقامة هنا",
    "سعر مرتفع مقارنة بالخدمة المقدمة",
    "المكان نظيف بس التكييف كان بايظ",
]
# Sample pool for the traffic simulator.
POOL = EXAMPLES + [
    "الاستقبال كان سريع والغرفة واسعة ومريحة جدا وإطلالة الغرفة خيالية",
    "لم تعجبني الإقامة، الضوضاء كانت مزعجة طوال الليل والموظفون غير متعاونين",
    "الفندق عادي لا يوجد فيه شيء مميز يستحق الذكر ولكن لا بأس به",
    "من أفضل الفنادق التي أقمت فيها، الإفطار متنوع والموقع قريب من كل شيء",
]
# Batch example: the first five lines carry a star rating, the last one does not.
BATCH_EXAMPLE = """5 | الفندق ممتاز جدا والموظفين محترمين والنظافة رائعة
1 | الغرفة قذرة والخدمة سيئة جدا ولن أعود مرة أخرى
3 | الموقع جيد لكن الغرف صغيرة والفطور عادي
4 | تجربة رائعة، أنصح الجميع بالإقامة هنا
2 | سعر مرتفع مقارنة بالخدمة المقدمة
المكان نظيف بس التكييف كان بايظ"""
 
 
# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """An error with a message that is safe to show to the person using the demo."""
 
 
_session = requests.Session()
 
 
def call_predict(text: str, base_url: str = API_URL) -> dict:
    """POST one review to /predict. Returns the API payload plus latency_ms."""
    started = time.perf_counter()
    try:
        r = _session.post(f"{base_url}/predict", json={"text": text}, timeout=TIMEOUT)
    except requests.ConnectionError:
        raise ApiError(
            f"Cannot reach the API at {base_url}. "
            "Start it with: docker compose up -d --build api"
        ) from None
    except requests.Timeout:
        raise ApiError("The API took too long to answer. Try again in a moment.") from None
    latency_ms = (time.perf_counter() - started) * 1000
 
    if r.status_code == 503:
        raise ApiError("The API is running but the model is not loaded. Check that models/v1 exists.")
    if r.status_code == 422:
        raise ApiError(f"The API rejected this text. Reviews must be 1 to {MAX_CHARS} characters.")
    if not r.ok:
        raise ApiError(f"The API answered with HTTP {r.status_code}.")
    data = r.json()
    data["latency_ms"] = latency_ms
    return data
 
 
def api_status() -> str:
    """Small status pill for the header."""
    try:
        r = _session.get(f"{API_URL}/health", timeout=4)
    except requests.RequestException:
        return _pill("down", "API unreachable")
    if r.status_code == 200:
        return _pill("ok", "API online")
    if r.status_code == 503:
        return _pill("warn", "API up, model not loaded")
    return _pill("warn", f"API returned HTTP {r.status_code}")
 
 
def _pill(kind: str, text: str) -> str:
    return f'<span class="status status-{kind}"><span class="dot"></span>{html.escape(text)}</span>'
 
 
# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
EMPTY_CARD = (
    '<div class="card card-empty"><p>Enter a review and press <b>Analyze sentiment</b>. '
    "Results show the predicted class, the model's confidence, and the latency of the API call.</p></div>"
)
 
 
def error_card(message: str) -> str:
    return f'<div class="card card-error"><b>Could not analyze this review.</b><br>{html.escape(message)}</div>'
 
 
def result_card(d: dict) -> str:
    meta = LABELS.get(d["label"], {"name": d["label"], "ar": "", "css": "neu"})
    conf = float(d["confidence"])
 
    breakdown = ""
    scores = d.get("scores")  # optional: only present if the API returns all class probabilities
    if isinstance(scores, dict) and all(k in scores for k in SCORE_ORDER):
        rows = "".join(
            f'<div class="bar-row"><span class="bar-name">{LABELS[k]["name"]}</span>'
            f'<div class="bar"><div class="fill {LABELS[k]["css"]}" style="width:{float(scores[k]) * 100:.1f}%"></div></div>'
            f'<span class="bar-val">{float(scores[k]):.1%}</span></div>'
            for k in SCORE_ORDER
        )
        breakdown = f'<div class="breakdown">{rows}</div>'
    else:
        breakdown = (
            f'<div class="bar single"><div class="fill {meta["css"]}" style="width:{conf * 100:.1f}%"></div></div>'
        )
 
    hint = ""
    if conf < LOW_CONFIDENCE:
        hint = (
            '<p class="hint">Low confidence. Reviews that mix praise and criticism are the hardest '
            "cases for this model, and the neutral class has the lowest F1 (0.76).</p>"
        )
 
    rating = ""
    stars = d.get("stars")  # set by the demo when a star rating was entered
    if stars is not None:
        expected = stars_to_label(stars)
        ok = expected == d["label"]
        word = "star" if stars == 1 else "stars"
        verdict = "The model agrees." if ok else f'The model predicted {meta["name"].lower()}.'
        rating = (
            f'<div class="rating"><span class="stars" title="{stars} of 5">{stars_text(stars)}</span>'
            f'<span>{stars} {word} maps to {LABELS[expected]["name"].lower()}.</span>'
            f'<span class="{"rating-ok" if ok else "rating-miss"}">{verdict}</span></div>'
        )
 
    return f"""
<div class="card {meta['css']}">
  <div class="card-top">
    <div>
      <div class="verdict">{meta['name']}</div>
      <div class="verdict-ar" lang="ar" dir="rtl">{meta['ar']}</div>
    </div>
    <div class="conf"><span class="conf-num">{conf:.1%}</span><span class="conf-cap">confidence</span></div>
  </div>
  {breakdown}
  <dl class="facts">
    <div><dt>Model version</dt><dd>{html.escape(str(d.get('model_version', 'unknown')))}</dd></div>
    <div><dt>API latency</dt><dd>{d.get('latency_ms', 0):.0f} ms</dd></div>
  </dl>
  {rating}
  {hint}
</div>"""
 
 
def _short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"
 
 
# --------------------------------------------------------------------------- #
# Tab 1: analyze one review
# --------------------------------------------------------------------------- #
def analyze(text: str, stars: int | None, history: list):
    text = (text or "").strip()
    if not text:
        gr.Warning("Enter a review first.")
        return EMPTY_CARD, history, history
    try:
        d = call_predict(text[:MAX_CHARS])
    except ApiError as e:
        return error_card(str(e)), history, history
    if stars is None or int(stars) == NO_RATING:
        stars_cell, matches = "-", "-"
    else:
        d["stars"] = int(stars)
        stars_cell = f"{int(stars)} ★"
        matches = "Yes" if stars_to_label(int(stars)) == d["label"] else "No"
    row = [_short(text), stars_cell, LABELS[d["label"]]["name"], f'{float(d["confidence"]):.1%}',
           f'{d["latency_ms"]:.0f}', matches]
    history = ([row] + history)[:HISTORY_LEN]
    return result_card(d), history, history
 
 
def clear_analyze():
    return "", NO_RATING, EMPTY_CARD
 
 
# --------------------------------------------------------------------------- #
# Tab 2: batch
# --------------------------------------------------------------------------- #
def analyze_batch(raw: str, progress=gr.Progress()):  # noqa: B008
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        gr.Warning("Paste at least one review, one per line.")
        return [], "", None
    if len(lines) > MAX_BATCH:
        gr.Warning(f"Only the first {MAX_BATCH} reviews were analyzed.")
        lines = lines[:MAX_BATCH]
 
    rows, counts, rated, agreed = [], Counter(), 0, 0
    for line in progress.tqdm(lines, desc="Analyzing"):
        stars, text = parse_batch_line(line)
        try:
            d = call_predict(text[:MAX_CHARS])
        except ApiError as e:
            raise gr.Error(str(e)) from None
        counts[d["label"]] += 1
        if stars is None:
            stars_cell, matches = "-", "-"
        else:
            rated += 1
            hit = stars_to_label(stars) == d["label"]
            agreed += hit
            stars_cell, matches = f"{stars} ★", "Yes" if hit else "No"
        rows.append([text, stars_cell, LABELS[d["label"]]["name"], f'{float(d["confidence"]):.1%}',
                     round(d["latency_ms"]), matches])
 
    parts = [f'{counts[k]} {LABELS[k]["name"].lower()}' for k in SCORE_ORDER if counts[k]]
    summary = f"**{len(rows)} reviews analyzed:** " + ", ".join(parts) + "."
    if rated:
        summary += f"\n\n**Agreement with star ratings:** {agreed} of {rated} ({agreed / rated:.1%})."
 
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, newline="", encoding="utf-8-sig"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(["review", "stars", "sentiment", "confidence", "latency_ms", "matches_stars"])
        writer.writerows(rows)
    return rows, summary, out.name
 
 
# --------------------------------------------------------------------------- #
# Tab 3: traffic simulator (feeds Prometheus / Grafana)
# --------------------------------------------------------------------------- #
def _pct(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]
 
 
def send_traffic(base_url: str, n: int, mode: str, rate: float):
    """Send n requests. mode 'short' keeps only the first 1-3 words (synthetic drift)."""
    base_url = (base_url or API_URL).rstrip("/")
    n, rate = int(n), float(rate)
    versions, labels, latencies, failures = Counter(), Counter(), [], 0
    started = time.perf_counter()
 
    for i in range(n):
        text = random.choice(POOL)
        if mode == "short":
            text = " ".join(text.split()[: random.randint(1, 3)])
        try:
            d = call_predict(text, base_url)
            latencies.append(d["latency_ms"])
            versions[d.get("model_version", "unknown")] += 1
            labels[d["label"]] += 1
        except ApiError:
            failures += 1
        if rate > 0:
            time.sleep(1 / rate)
        if (i + 1) % 10 == 0 or i + 1 == n:
            yield f"Sent {i + 1} of {n} requests..."
 
    ok = n - failures
    elapsed = time.perf_counter() - started
    lines = [f"**Sent {n} requests to `{base_url}` in {elapsed:.0f} s.** {ok} succeeded, {failures} failed.", ""]
    if latencies:
        lat = sorted(latencies)
        lines += [
            "| Latency | ms |",
            "|---|---|",
            f"| mean | {sum(lat) / len(lat):.0f} |",
            f"| p50 | {_pct(lat, 0.50):.0f} |",
            f"| p95 | {_pct(lat, 0.95):.0f} |",
            "",
            "| Model version | Requests | Share |",
            "|---|---|---|",
            *[f"| `{v}` | {c} | {c / ok:.1%} |" for v, c in versions.most_common()],
            "",
            "| Predicted class | Requests | Share |",
            "|---|---|---|",
            *[f"| {LABELS[k]['name']} | {c} | {c / ok:.1%} |" for k, c in labels.most_common()],
        ]
    yield "\n".join(lines)
 
 
# --------------------------------------------------------------------------- #
# About tab content (numbers come from the repo README)
# --------------------------------------------------------------------------- #
ABOUT = f"""
### What this demo shows
 
The model is one part of the system. Every request on this page goes through the FastAPI service,
which exposes Prometheus metrics, so the same traffic appears on the Grafana dashboard.
 
| | |
|---|---|
| **Task** | 3-class sentiment (positive, neutral, negative) for Arabic hotel reviews |
| **Model** | `aubmindlab/bert-base-arabertv02`, fine-tuned (max length 128) |
| **Data** | HARD, Booking.com reviews in Modern Standard Arabic and dialects. 320,064 train, 40,008 validation, 40,009 test after removing duplicates |
| **Labels** | Star ratings mapped to classes: 1 to 2 stars negative, 3 neutral, 4 to 5 positive |
| **Training** | 60,000-review subset of the training set, Colab T4, selected on validation F1 |
| **Test results** | accuracy 0.898, macro F1 0.847 (negative 0.834, neutral 0.758, positive 0.950) |
| **Serving** | FastAPI (`/predict`, `/health`, `/metrics`), BentoML with adaptive batching, nginx canary router |
| **Operations** | MLflow tracking and registry, DVC pipeline, GitHub Actions quality gate, PSI drift alerts |
 
### Limits to keep in mind
 
- The model was trained on hotel reviews. Accuracy on other domains, such as product reviews, was not measured.
- Neutral is the weakest class. Three-star reviews often mix praise and criticism.
- Each configuration was trained with a single seed, so small differences between runs are not conclusive.
- Agreement with star ratings is a fair check only on reviews the model has not seen. Reviews from the HARD training split will look better than they should.
- The drift shown by the simulator is synthetic. It demonstrates detection and alerting, not a real change in users.
- The API records only text length, confidence and predicted label for monitoring. It never stores review text.
 
Source code, results and the full write-up: [{REPO_URL}]({REPO_URL})
"""
 
 
# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #
THEME = gr.themes.Base(
    primary_hue="teal",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("IBM Plex Sans Arabic"), "Tahoma", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)
 
CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto; }
.app-header { align-items: center; margin-bottom: 4px; }
.app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; }
.app-header p { margin: 2px 0 0; opacity: .75; }
.status-wrap { display: flex; justify-content: flex-end; }
.status { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
          font-size: .85rem; border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary); }
.status .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.status-ok .dot { color: #2e9e6a; } .status-warn .dot { color: #c79a2b; } .status-down .dot { color: #d9534f; }
 
.card { border: 1px solid var(--border-color-primary); border-left: 5px solid var(--tone, var(--border-color-primary));
        background: var(--background-fill-secondary); border-radius: 10px; padding: 18px 20px; min-height: 150px; }
.card.pos { --tone: #2e9e6a; } .card.neg { --tone: #d9534f; } .card.neu { --tone: #c79a2b; }
.card-empty, .card-error { border-left-color: var(--border-color-primary); opacity: .9; display: flex; align-items: center; }
.card-empty p { margin: 0; line-height: 1.6; }
.card-error { --tone: #d9534f; border-left-color: #d9534f; display: block; }
.card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.verdict { font-size: 2rem; font-weight: 600; line-height: 1.1; color: var(--tone); }
.verdict-ar { font-size: 1.15rem; opacity: .8; margin-top: 4px; text-align: left; }
.conf { text-align: right; }
.conf-num { display: block; font-size: 1.6rem; font-weight: 600; }
.conf-cap { font-size: .8rem; opacity: .7; }
 
.bar { height: 8px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; margin: 14px 0 6px; }
.fill { height: 100%; border-radius: 4px; background: #888; transition: width .35s ease; }
.fill.pos { background: #2e9e6a; } .fill.neg { background: #d9534f; } .fill.neu { background: #c79a2b; }
.breakdown { margin: 14px 0 6px; display: grid; gap: 8px; }
.bar-row { display: grid; grid-template-columns: 78px 1fr 56px; align-items: center; gap: 10px; font-size: .9rem; }
.bar-row .bar { margin: 0; } .bar-val { text-align: right; font-variant-numeric: tabular-nums; }
 
.rating { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-color-primary);
          display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; font-size: .9rem; }
.stars { color: #e0a800; letter-spacing: 2px; font-size: 1.05rem; }
.rating-ok { color: #2e9e6a; font-weight: 500; } .rating-miss { color: #d9534f; font-weight: 500; }
.facts { display: flex; gap: 28px; margin: 12px 0 0; }
.facts dt { font-size: .78rem; opacity: .65; } .facts dd { margin: 0; font-weight: 500; }
.hint { margin: 12px 0 0; font-size: .88rem; opacity: .85; }
 
.site-footer { opacity: .7; font-size: .85rem; margin-top: 6px; }
.rtl-table td, .rtl-table th { unicode-bidi: plaintext; }
"""
 
 
# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Arabic Sentiment Analysis") as demo:
    with gr.Row(elem_classes="app-header"):
        gr.Markdown(
            "# Arabic Sentiment Analysis\n"
            "AraBERT fine-tuned on hotel reviews, served through a monitored FastAPI service."
        )
        status = gr.HTML(api_status(), elem_classes="status-wrap")
 
    with gr.Tabs():
        with gr.Tab("Analyze"):
            history = gr.State([])
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    review = gr.Textbox(
                        label="Review (Arabic)",
                        placeholder="اكتب تقييم الفندق هنا",
                        lines=5,
                        max_lines=10,
                        max_length=MAX_CHARS,
                        rtl=True,
                    )
                    stars_in = gr.Radio(
                        [("No rating", NO_RATING)] + [(f"{i} ★", i) for i in range(6)],
                        value=NO_RATING,
                        label="Star rating (optional)",
                        info="0 to 2 is negative, 3 is neutral, 4 to 5 is positive. The result shows whether the model agrees.",
                    )
                    with gr.Row():
                        go = gr.Button("Analyze sentiment", variant="primary")
                        clear = gr.Button("Clear")
                    gr.Examples(EXAMPLES, inputs=review, label="Try an example")
                with gr.Column(scale=1):
                    card = gr.HTML(EMPTY_CARD)
            gr.Markdown("**Recent predictions**")
            recent = gr.Dataframe(
                headers=["Review", "Stars", "Sentiment", "Confidence", "ms", "Matches stars"],
                value=[],
                interactive=False,
                wrap=True,
                elem_classes="rtl-table",
            )
            go.click(analyze, [review, stars_in, history], [card, history, recent])
            review.submit(analyze, [review, stars_in, history], [card, history, recent])
            clear.click(clear_analyze, None, [review, stars_in, card])
 
        with gr.Tab("Batch"):
            gr.Markdown(
                f"Paste up to {MAX_BATCH} reviews, one per line. To compare the model with a star rating, "
                "start the line with the stars and a bar, for example `4 | الفندق ممتاز`. "
                "Results can be downloaded as CSV."
            )
            batch_in = gr.Textbox(label="Reviews", lines=8, max_lines=14, rtl=True,
                                  placeholder="كل تقييم في سطر منفصل")
            gr.Examples([[BATCH_EXAMPLE]], inputs=batch_in, label="Try an example with star ratings")
            batch_go = gr.Button("Analyze all", variant="primary")
            batch_summary = gr.Markdown()
            batch_out = gr.Dataframe(
                headers=["Review", "Stars", "Sentiment", "Confidence", "ms", "Matches stars"],
                interactive=False,
                wrap=True,
                elem_classes="rtl-table",
            )
            batch_file = gr.File(label="Download CSV", interactive=False)
            batch_go.click(analyze_batch, batch_in, [batch_out, batch_summary, batch_file])
 
        with gr.Tab("Monitoring"):
            gr.Markdown(
                "Send test traffic and watch it appear on the dashboards.\n\n"
                "- **Drift:** choose *Short reviews* and send about 500 requests. The PSI panel in Grafana "
                "should rise above 0.25 and fire the `DataDriftHigh` alert. The drift is synthetic.\n"
                "- **Canary:** choose *Canary router* (start it with `docker compose --profile canary up -d --build`) "
                "and check the version split below. With `CANARY_PCT=5`, about 5% of requests go to `v2`."
            )
            with gr.Row():
                target = gr.Radio(
                    [(f"Stable API ({API_URL})", API_URL), (f"Canary router ({CANARY_URL})", CANARY_URL)],
                    value=API_URL,
                    label="Send traffic to",
                )
                mode = gr.Radio(
                    [("Full reviews", "normal"), ("Short reviews (1 to 3 words, drift)", "short")],
                    value="normal",
                    label="Traffic type",
                )
            with gr.Row():
                n = gr.Slider(10, 600, value=100, step=10, label="Requests")
                rate = gr.Slider(1, 20, value=5, step=1, label="Requests per second")
            with gr.Row():
                send = gr.Button("Send traffic", variant="primary")
                stop = gr.Button("Stop")
                gr.Button("Open Grafana", link=GRAFANA_URL)
                gr.Button("Open Prometheus", link=PROMETHEUS_URL)
            sim_out = gr.Markdown()
            run = send.click(send_traffic, [target, n, mode, rate], sim_out)
            stop.click(None, None, None, cancels=[run])
 
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
 
    gr.Markdown(f"[Source code on GitHub]({REPO_URL})", elem_classes="site-footer")
 
    # Refresh the status pill on page load and every 30 seconds.
    demo.load(api_status, None, status)
    gr.Timer(30).tick(api_status, None, status)
 
 
if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        theme=THEME,
        css=CSS,
        footer_links=["gradio", "settings"],  
    )