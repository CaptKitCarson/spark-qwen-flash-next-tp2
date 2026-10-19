"""Fixed-length streaming decode comparison at concurrency 1 and 4."""
import concurrent.futures
import os
import json
import pathlib
import statistics
import sys
import time
import urllib.request

label = sys.argv[1]
def request(index, limit=1024):
    body = {
        "model": os.environ.get("MODEL", "qwen3.8-flash-next"), "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": limit,
        "ignore_eos": True, "temperature": 0, "seed": 123 + index,
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content":
            f"Write a complete Python implementation of an LRU cache, with type hints, "
            f"thread safety, expiration, statistics, and extensive unit tests. Variant {index}. "
            "Explain the design and provide the full code."}],
    }
    req = urllib.request.Request(os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/") + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    last = time.monotonic()
                    first = first or last
    assert usage and first and last and usage["completion_tokens"] == limit, usage
    return {"tokens": limit, "ttft": first - start, "elapsed": time.monotonic() - start,
            "decode_seconds": last - first, "decode_tok_s": (limit - 1) / (last - first)}

request(0, 128)
rows = []
for concurrency in (1, 4):
    for repeat in range(3):
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            runs = list(pool.map(request, range(concurrency)))
        row = {"concurrency": concurrency, "repeat": repeat, "runs": runs,
               "aggregate_e2e_tok_s": sum(x["tokens"] for x in runs) / (time.monotonic() - start),
               "mean_decode_tok_s": statistics.mean(x["decode_tok_s"] for x in runs)}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "runs"}), flush=True)
result = {"label": label, "tokens_per_request": 1024, "temperature": 0,
          "reasoning_effort": "medium", "ignore_eos": True, "rows": rows,
          "summary": {str(c): {"mean_decode_tok_s": statistics.mean(
              r["mean_decode_tok_s"] for r in rows if r["concurrency"] == c),
              "aggregate_e2e_tok_s": statistics.mean(
              r["aggregate_e2e_tok_s"] for r in rows if r["concurrency"] == c)} for c in (1, 4)}}
pathlib.Path(f"tune-{label}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result["summary"]), flush=True)
