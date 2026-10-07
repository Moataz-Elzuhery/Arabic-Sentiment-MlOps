from scripts.locust_summary import summarize

CSV = (
    "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
    "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
    "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%\n"
    "POST,/predict,300,3,120,140.5,50,900,60,20.5,0.2,120,130,150,170,210,250,300,340,800,900,900\n"
    ",Aggregated,300,3,120,140.5,50,900,60,20.5,0.2,120,130,150,170,210,250,300,340,800,900,900\n"
)


def test_summarize_reads_percentiles(tmp_path):
    f = tmp_path / "x_stats.csv"
    f.write_text(CSV)
    s = summarize(str(f), "demo", users=10, duration=60)
    assert s["p95_ms"] == 250 and s["p50_ms"] == 120 and s["p99_ms"] == 340
    assert s["requests"] == 300 and s["failures"] == 3 and s["rps"] == 20.5


def test_summarize_all_splits_versions(tmp_path):
    header = CSV.split("\n")[0]
    rows = [
        "POST,/predict [v1],285,0,100,110,50,400,60,19.0,0,100,110,120,130,150,180,200,250,380,400,400",
        "POST,/predict [v2],15,1,140,150,60,500,60,1.0,0.1,140,150,160,170,200,300,400,450,500,500,500",
        ",Aggregated,300,1,100,112,50,500,60,20.0,0.1,100,110,120,130,150,190,210,260,400,500,500",
    ]
    f = tmp_path / "c_stats.csv"
    f.write_text(header + "\n" + "\n".join(rows) + "\n")
    from scripts.locust_summary import summarize_all
    out = summarize_all(str(f), "canary_5pct")
    assert [r["name"] for r in out] == ["/predict [v1]", "/predict [v2]"]
    assert out[1]["failures"] == 1 and out[1]["p95_ms"] == 300
