#!/usr/bin/env python3
"""Run a prompt suite through a reference implementation, greedy, and write
answers in the same JSONL format as `swiftlet generate --prompts-file --answers`,
so scripts/suite_score.py can put the two side by side. See docs/CACHE_SUITE.md.

  # Same weights, held fully in memory (mlx-lm). Leonickson/Qwen3.6-35B-A3B-qpack
  # is byte-identical to mlx-community/Qwen3.6-35B-A3B-4bit, so this isolates
  # what streaming does from what quantization does. Needs ~20 GB free for the
  # weights alone; on a 24 GB Mac run it elsewhere.
  pip install mlx-lm
  scripts/suite_reference.py mlx --model mlx-community/Qwen3.6-35B-A3B-4bit \
      --suite scripts/suites/core.jsonl --out ref-mlx-4bit.jsonl

  # Any OpenAI-compatible endpoint: a lab GPU running vLLM with the full
  # precision model, a hosted provider, LM Studio. Measures what 4-bit costs.
  scripts/suite_reference.py openai --base-url http://gpu-box:8000/v1 \
      --model Qwen/Qwen3.6-35B-A3B --suite scripts/suites/core.jsonl --out ref-bf16.jsonl \
      [--api-key-env OPENROUTER_API_KEY] [--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}']

Both backends render the chat template with reasoning disabled and decode
greedily (temperature 0) up to each prompt's max_new, as Swiftlet does.
mlx answers carry token ids, so agreement with Swiftlet is checked token by
token; API answers carry text only.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def load_suite(path, cap):
    rows = [json.loads(l) for l in open(path) if l.strip() and not l.lstrip().startswith(("#", "//"))]
    for r in rows:
        r["max_new"] = int(r.get("max_new", 256))
        if cap:
            r["max_new"] = min(r["max_new"], cap)
    return rows


def messages_for(p):
    m = []
    if p.get("system"):
        m.append({"role": "system", "content": p["system"]})
    m.append({"role": "user", "content": p["prompt"]})
    return m


def run_mlx(args, suite):
    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.exit("mlx-lm is not installed: pip install mlx-lm")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)
    for i, p in enumerate(suite):
        try:
            ids = tokenizer.apply_chat_template(messages_for(p), add_generation_prompt=True,
                                                enable_thinking=False, tokenize=True)
        except TypeError:
            ids = tokenizer.apply_chat_template(messages_for(p), add_generation_prompt=True, tokenize=True)
        tokens, text, finish, t_first, start = [], "", None, None, time.time()
        for r in stream_generate(model, tokenizer, prompt=ids, max_tokens=p["max_new"], sampler=sampler):
            if t_first is None:
                t_first = time.time()
            if r.finish_reason == "stop":
                finish = "stop"
                break
            tokens.append(int(r.token))
            text += r.text
            finish = r.finish_reason
        end = time.time()
        yield i, p, ids, tokens, tokenizer.decode(tokens), finish == "stop", (t_first or end) - start, end - (t_first or end)


def run_openai(args, suite):
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    extra = json.loads(args.extra_body) if args.extra_body else {}
    for i, p in enumerate(suite):
        body = {"model": args.model, "messages": messages_for(p), "temperature": 0,
                "max_tokens": p["max_new"], "stream": False, **extra}
        req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {key}"} if key else {})})
        start = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        end = time.time()
        choice = out["choices"][0]
        text = choice["message"].get("content") or ""
        n = (out.get("usage") or {}).get("completion_tokens", 0)
        yield i, p, [None] * (out.get("usage") or {}).get("prompt_tokens", 0), [None] * n, text, \
            choice.get("finish_reason") == "stop", 0.0, end - start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["mlx", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default=os.path.join(os.path.dirname(__file__), "suites", "core.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-cap", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--extra-body", default="")
    args = ap.parse_args()
    suite = load_suite(args.suite, args.max_new_cap)
    gen = run_mlx(args, suite) if args.backend == "mlx" else run_openai(args, suite)
    with open(args.out, "w") as f:
        for i, p, ids, tokens, text, eos, prefill_s, decode_s in gen:
            rec = {"index": i, "id": p["id"], "category": p.get("category", "uncategorized"),
                   "prompt_tokens": len(ids), "generated": len(tokens), "hit_eos": eos,
                   "prefill_s": prefill_s, "decode_s": decode_s,
                   "decode_tok_s": len(tokens) / decode_s if decode_s else 0.0,
                   "text": text, "tokens": tokens if tokens and tokens[0] is not None else None,
                   "backend": args.backend, "model": args.model}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(suite)}] {p['id']}: {len(tokens)} tokens{'' if eos else ' (hit max_new)'}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
