"""Load-test an OpenAI-compatible LLM endpoint (vLLM / SGLang).

Measures the three numbers that matter in production:
  - TTFT  : time to first token
  - ITL   : inter-token latency (time per token after the first)
  - TPS   : throughput in output tokens per second

Works against any OpenAI-compatible server (vLLM's default is
http://localhost:8000/v1 with api_key "EMPTY").

Usage:
    python loadtest.py --model unsloth/Llama-3.2-1B-Instruct --n 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PROMPT = "Explain what a transformer neural network is, in about 120 words."


def one_request(client, model, prompt, max_tokens):
    """Stream one completion and time it. Returns a metrics dict."""
    start = time.perf_counter()
    first_token_at = None
    n_tokens = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            n_tokens += 1
    end = time.perf_counter()

    if first_token_at is None or n_tokens == 0:
        return None

    ttft = first_token_at - start
    gen_time = end - first_token_at
    itl = gen_time / max(1, n_tokens - 1)
    tps = n_tokens / (end - start)
    return {"ttft": ttft, "itl": itl, "tps": tps, "n_tokens": n_tokens}


def _pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))
    return values[k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20, help="total requests")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Sending {args.n} requests at concurrency {args.concurrency} to {args.model} ...")
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, client, args.model, PROMPT, args.max_tokens)
            for _ in range(args.n)
        ]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - wall_start

    results = [r for r in results if r]
    if not results:
        raise SystemExit("No successful requests — is the server running?")

    ttfts = [r["ttft"] for r in results]
    itls = [r["itl"] for r in results]
    total_tokens = sum(r["n_tokens"] for r in results)

    print("\n" + "─" * 60)
    print(f"successful requests : {len(results)}/{args.n}")
    print(f"TTFT  p50 / p95     : {_pct(ttfts, 50) * 1000:.0f} ms / {_pct(ttfts, 95) * 1000:.0f} ms")
    print(f"ITL   p50 / p95     : {_pct(itls, 50) * 1000:.1f} ms / {_pct(itls, 95) * 1000:.1f} ms")
    print(f"per-request TPS avg : {statistics.mean(r['tps'] for r in results):.1f} tok/s")
    print(f"aggregate TPS       : {total_tokens / wall:.1f} tok/s  (over {wall:.1f}s wall)")


if __name__ == "__main__":
    main()
