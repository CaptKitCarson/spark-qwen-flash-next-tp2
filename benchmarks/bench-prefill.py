"""Uncached, exact-token prefill/TTFT probe for transport comparisons."""
import os
import json
import pathlib
import statistics
import sys
import time
import urllib.request
import uuid

MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
def post(path, payload):
    return urllib.request.urlopen(urllib.request.Request(os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/") + path,
        json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=300)
def run(length):
    text = uuid.uuid4().hex + "\n" + ("A harbor station records wind speed, tide height, and vessel arrivals every hour.\n" * 2600)
    with post("/tokenize", {"model": MODEL, "prompt": text}) as response:
        tokens = json.load(response)["tokens"][:length]
    assert len(tokens) == length
    start=time.monotonic(); first=None; usage=None
    with post("/v1/completions", {"model": MODEL, "prompt": tokens, "max_tokens": 1,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True}}) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
            event=json.loads(line[6:])
            if event.get("usage"): usage=event["usage"]
            for choice in event.get("choices", []):
                if choice.get("text") or choice.get("finish_reason"):
                    first=first or time.monotonic()
    assert first and usage and usage["prompt_tokens"] == length, usage
    cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert cached == 0, usage
    return {"tokens": length, "ttft_s": first-start, "effective_prefill_tok_s": length/(first-start), "cached_tokens": cached}
run(2048)
rows=[]
for length in (8192, 32768):
    samples=[run(length) for _ in range(3)]
    row={"tokens": length, "samples": samples, "ttft_s": statistics.median(x["ttft_s"] for x in samples),
         "effective_prefill_tok_s": statistics.median(x["effective_prefill_tok_s"] for x in samples)}
    rows.append(row); print(json.dumps(row), flush=True)
pathlib.Path(f"prefill-{sys.argv[1]}.json").write_text(json.dumps({"label":sys.argv[1], "rows":rows},indent=2))
