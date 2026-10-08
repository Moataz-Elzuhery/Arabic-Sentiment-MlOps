"""Load test for POST /predict.

Run (headless, saves CSVs):
  locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
         -u 10 -r 2 -t 60s --csv benchmarks/raw/baseline_fastapi

-u = concurrent users, -r = users spawned per second, -t = test duration.
Keep the same -u/-r/-t for every model you compare, or the numbers are not comparable.

API_STYLE=fastapi (default): POST /predict {"text": "..."}      -> dict
API_STYLE=bento            : POST /predict {"texts": ["..."]}   -> [dict]   (BentoML batchable API)
--reset-stats drops the ramp-up requests so cold-start does not pollute p95/p99.
TAG_VERSION=1: report stats per model version ("/predict [v1]", "/predict [v2]") - used for canary.
"""
import os
import random

from locust import HttpUser, between, task

# A mix of short/long reviews in MSA and dialect, positive / negative / neutral.
REVIEWS = [
    "المنتج ممتاز جدًا والخامة كويسة",
    "الخدمة سيئة والمنتج وصل مكسور",
    "المنتج وصل في معاده",
    "جودة رائعة وسعر مناسب أنصح به بشدة",
    "تجربة سيئة للغاية ولن أكررها أبدًا",
    "الغرفة نظيفة والموظفين متعاونين لكن الموقع بعيد شوية عن وسط البلد",
    "عادي مفيش حاجة مميزة",
    "تحفة والله، أحسن حاجة اشتريتها السنة دي",
    "وحش جدًا، الصورة غير الحقيقة والتوصيل اتأخر أسبوعين",
    "الفطار كان كويس بس الإنترنت ضعيف جدًا في الغرفة وده مضايق في الشغل",
    "مش بطال، يؤدي الغرض",
    ("الفندق جميل والإطلالة خيالية والاستقبال محترم وسريع في إنهاء الإجراءات "
     "ولكن الضوضاء في الليل كانت مزعجة قليلًا"),
]

API_STYLE = os.getenv("API_STYLE", "fastapi")
TAG_VERSION = os.getenv("TAG_VERSION") == "1"
WAIT_MIN = float(os.getenv("WAIT_MIN", "0.1"))
WAIT_MAX = float(os.getenv("WAIT_MAX", "0.5"))


class ReviewUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)

    @task
    def predict(self):
        text = random.choice(REVIEWS)
        payload = {"texts": [text]} if API_STYLE == "bento" else {"text": text}
        with self.client.post("/predict", json=payload, name="/predict",
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body = r.json()
            item = body[0] if isinstance(body, list) and body else body
            if not isinstance(item, dict) or item.get("label") not in {
                    "positive", "negative", "neutral"}:
                r.failure("unexpected response body")
            elif TAG_VERSION:
                r.request_meta["name"] = f"/predict [{item.get('model_version', '?')}]"
