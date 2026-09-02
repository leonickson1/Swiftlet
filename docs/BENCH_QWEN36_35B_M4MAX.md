# Benchmark: Qwen3.6-35B-A3B (qpack Q4) on Apple M4 Max

First real-model benchmark of the full Swiftlet stack (S1a + S1b layer-major
prefill + token-batched GEMVs + S2 Metal attention/GPU KV + S3a/S3b
instrumentation) on the production 35B qpack. Run 2026-08-31 on macbook4.

Headline (warm, greedy, 64 new tokens):

| Metric | Value |
|---|---|
| Decode, short context (16-tok prompt) | **25.0 tok/s** (25.00–25.08 across 3 runs) |
| Decode, long context (503-tok prompt, default 8 GB cache) | 13.8–13.9 tok/s |
| Decode, long context, 20 GB expert cache | 15.4–15.5 tok/s |
| TTFT, 503-tok prompt, layer-major chunk 32 (default) | **11.65–11.70 s** (43.0–43.2 tok/s prefill) |
| TTFT, 503-tok prompt, layer-major chunk 128 | 10.92–10.93 s (46.0–46.1 tok/s prefill) |
| TTFT, 503-tok prompt, token-major (`--prefill-chunk 0`) | 23.69–23.81 s (21.1–21.2 tok/s prefill) |
| Layer-major (32) vs token-major prefill | **2.04× faster** |
| Cold-start decode (page cache purged) | 13.92 tok/s (1.80× slower than warm) |
| Peak RSS | 9.7 GiB (short ctx) / 10.9 GiB (long ctx) / 14.5 GiB (20 GB cache) |
| Greedy parity across all three prefill schedules | identical outputs, bit for bit |

S4 verdict: **NO-GO (defer async expert loading)** — traces attribute the
warm-path cost to command-buffer waits and GPU execution, not storage. Details
in "S4 go/no-go" below.

Low-end sibling: the same protocol on a base M1 Mac mini (8-core GPU, 16 GB),
upstream `aaa910a` vs PR #24 `5d8b2ab` with a `--cache-gb` 2/4/6 sweep, is in
[`BENCH_QWEN36_35B_M1_MINI.md`](BENCH_QWEN36_35B_M1_MINI.md) — the
RAM-constrained trace the S4 note below asks for.

## Environment

Per the Benchmark contract in the port roadmap (`ROADMAP.md`, origin/main of
the `switftbri` planning repo).

- **Commit**: `69820f7c6abd872c899ae9067eef224a5271cd81`
  (`switftbri/s1b-token-batched-gemv`, clean tree; "Chunk the DeltaNet
  recurrence into one T-step scan per layer"). Worktree rsync'd to
  `~/build/s1c-swiftlet` on the Mac.
- **Build**: `swift build -c release --scratch-path ~/build/s1c-swiftlet/.swiftcache`;
  binary invoked directly (`.swiftcache/release/swiftlet`). No extra flags; the
  CLT Testing-framework flags are needed only for `swift test`. Build step was
  fully separated from all timed runs.
- **Machine**: macbook4 — Apple M4 Max, 40-core GPU, 64 GB unified memory
  (68719476736 B), internal APFS SSD (`/dev/disk3s5`, 926 GiB). macOS 26.6.2
  (build 25G83). Swift 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101),
  swift-driver 1.148.6, Command Line Tools only (no Xcode).
- **Exclusive box**: `ps aux` idle-check before every batch; only resident
  daemons (node_exporter, tailscaled) present, no other swift/clang/make jobs.
- **Model**: `~/models/qwen3.6-35b.qpack` — QPACK v1 container of
  Qwen3.6-35B-A3B (repack source `mlx-community/Qwen3.6-35B-A3B-4bit`), MLX
  affine Q4 group-64. 40 layers (30 DeltaNet + 10 GQA), 256 routed experts per
  layer (top-8) + 1 shared expert, hidden 2048, vocab 248320. Expert stride
  1 769 472 B; 40 × 432 MiB `packed_experts/layer_NN.bin`; dense
  `model.safetensors` 1.39 GB; container total 18 GB. Installed 2026-08-31 via
  the C6d hash-pinned streaming transfer (dual-vantage pin, `hashes.json` in
  container), hash-verified.
- **Runtime config**: `--gpu` Metal runtime; expert cache budget default
  `--cache-gb 8` (= 4854 slots of 1.69 MiB) except run set F (`--cache-gb 20`
  = 10240 slots, the full expert pool). S3b counter probe on this device:
  `timestampSet=yes stage=yes(sample resolved) dispatch=no` — the per-phase
  GPU split honestly reports **unsupported** (M4 Max samples timestamps only
  at encoder boundaries; the fast path encodes several phases per encoder), so
  GPU/wait times below are per command-buffer label, encode-side phase times
  are exact.

## Protocol

- **Sampler**: greedy argmax (the CLI `generate` path), no temperature, EOS
  ids [248046, 248044] from the container's `config.json`.
- **Output length**: `--max-new 64`; all A–F runs generated the full 64 tokens
  (no early EOS).
- **Short prompt** (run set A), exactly 16 tokens:
  `The fieldfare is a large thrush that breeds across northern Europe and winters in`
- **Long prompt** (run sets B/C/D/F), exactly 503 tokens: a fixed 2787-byte
  English paragraph about MoE inference scheduling, stored verbatim at
  `~/bench-logs/plong.txt` on macbook4, sha256
  `d5ff9b0a8d0a52935b04d71ad39e874e96176f9ea20f6a105ac5d5ff506fbc19`.
  Tokenized by the container's own tokenizer assets (hash-pinned, so the
  encoding is reproducible).
- **Parity prompt** (run set E): 200 deterministic raw ids bypassing the
  tokenizer: `ids[i] = 1000 + 211*i` for `i = 0..199` (max 42 989 < vocab),
  passed via `--ids`.
- **Runs**: per configuration 1 warmup + 3 measured, every measured run
  reported (no best-of). Each run is a fresh process (in-process expert cache
  always starts empty). "Warm" = OS page cache warm from prior runs; on this
  64 GB machine the 16.88 GiB expert pool fits in page cache entirely. The
  labeled **cold** run was taken immediately after `sudo purge` with no
  intervening file access.
- **Metric definitions**: TTFT = the S3a prefill step wall time (model +
  tokenizer load excluded; process totals reported separately). Decode tok/s
  as printed by the harness over the 64-step loop. Peak RSS from
  `/usr/bin/time -l`. Bytes read = expert-cache misses × stride (each miss is
  exactly one pread of 1 769 472 B). Wait = the sum of blocking
  `waitUntilCompleted` calls; GPU = sum of per-command-buffer GPU durations
  (all 100% of buffers timed in every run below, `err=0` throughout).

Repro (on the Mac, release binary `$BIN`, container `$MODEL`):

```sh
$BIN generate $MODEL --gpu --prompt "<short prompt>" --max-new 64                       # A
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --prefill-chunk 0   # B
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --prefill-chunk 32  # C
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --prefill-chunk 128 # D
$BIN generate $MODEL --gpu --ids "1000,1211,...,42989" --max-new 32 --prefill-chunk N   # E
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --prefill-chunk 32 --cache-gb 20  # F
```

## Results

### A — decode-focused: 16-token prompt, 64 new tokens, chunk 32 (default), 8 GB cache

| Run | TTFT (prefill wall) | Decode tok/s | Decode wait / GPU (64 steps) | Cache hits/misses (hit rate) | Bytes read | Peak RSS | Process real |
|---|---|---|---|---|---|---|---|
| cold-1 (purged) | 1.853 s | **13.92** | 2.900 s / 2.226 s | 18429 / 4160 (82%) | 6.86 GiB (SSD) | 9.68 GiB | 8.79 s |
| warm-0 (warmup) | 0.593 s | 25.07 | 1.908 s / 1.377 s | 18429 / 4160 (82%) | 6.86 GiB (page cache) | 9.67 GiB | 3.95 s |
| warm-1 | 0.595 s | **25.02** | 1.910 s / 1.379 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 3.94 s |
| warm-2 | 0.596 s | **25.00** | 1.911 s / 1.377 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 3.96 s |
| warm-3 | 0.594 s | **25.08** | 1.905 s / 1.378 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 3.94 s |

Spread of measured warm decode: 25.00–25.08 tok/s (0.3%). Per-step schedule is
fixed: 41 command buffers and 2082 dispatches per decode token (2624 cb /
133 248 dispatches per 64 steps). 4160 unique (layer, expert) blobs touched —
below the 4854-slot budget, so **zero evictions** in this configuration; the
identical hit/miss counts across cold and warm confirm identical routing.
Greedy continuation (identical in all five runs) is coherent and factually
reasonable: " southern Europe. It is a common winter visitor to the UK, …".

### B — prefill token-major (`--prefill-chunk 0`): 503-token prompt, 64 new

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | cb / dispatches | Decode tok/s | Cache hits/misses (rate) | Bytes read | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 23.689 s | 21.23 | 19.604 / 15.261 s | 20623 / 1 046 242 | 13.17 | 171983 / 9457 (95%) | 15.58 GiB | 10.93 GiB | 29.37 s |
| warm-1 | **23.791 s** | 21.14 | 19.686 / 15.299 s | 20623 / 1 046 242 | 13.82 | 171983 / 9457 (95%) | 15.58 GiB | 10.92 GiB | 29.29 s |
| warm-2 | **23.806 s** | 21.13 | 19.671 / 15.285 s | 20623 / 1 046 242 | 13.78 | 171983 / 9457 (95%) | 15.58 GiB | 10.91 GiB | 29.33 s |
| warm-3 | **23.800 s** | 21.13 | 19.635 / 15.275 s | 20623 / 1 046 242 | 13.70 | 171983 / 9457 (95%) | 15.58 GiB | 10.94 GiB | 29.34 s |

41 cb per prompt token (503 × 41 = 20 623). Miss count 9457 exceeds the
4854-slot budget → LFU eviction churn; re-reads served from page cache.

### C — prefill layer-major chunk 32 (default): 503-token prompt, 64 new

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | cb / dispatches | Decode tok/s | Cache hits/misses (rate) | Bytes read | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 11.647 s | 43.19 | 8.377 / 8.004 s | 656 / 226 729 | 13.89 | 54768 / 9181 (86%) | 15.13 GiB | 10.92 GiB | 17.13 s |
| warm-1 | **11.670 s** | 43.10 | 8.360 / 7.982 s | 656 / 226 729 | 13.89 | 54768 / 9181 (86%) | 15.13 GiB | 10.89 GiB | 17.15 s |
| warm-2 | **11.679 s** | 43.07 | 8.373 / 8.000 s | 656 / 226 729 | 13.86 | 54768 / 9181 (86%) | 15.13 GiB | 10.90 GiB | 17.18 s |
| warm-3 | **11.702 s** | 42.98 | 8.389 / 8.014 s | 656 / 226 729 | 13.79 | 54768 / 9181 (86%) | 15.13 GiB | 10.89 GiB | 17.22 s |

16 chunks × 41 cb = 656 cb; **2.04× faster TTFT than token-major** with 4.6×
fewer dispatches and 31× fewer command buffers.

### D — prefill layer-major chunk 128: 503-token prompt, 64 new

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | cb / dispatches | Decode tok/s | Cache hits/misses (rate) | Bytes read | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 10.921 s | 46.06 | 7.618 / 7.324 s | 164 / 136 267 | 14.12 | 28796 / 8959 (76%) | 14.76 GiB | 10.91 GiB | 16.32 s |
| warm-1 | **10.915 s** | 46.08 | 7.618 / 7.322 s | 164 / 136 267 | 14.11 | 28796 / 8959 (76%) | 14.76 GiB | 10.92 GiB | 16.33 s |
| warm-2 | **10.927 s** | 46.03 | 7.622 / 7.320 s | 164 / 136 267 | 14.10 | 28796 / 8959 (76%) | 14.76 GiB | 10.91 GiB | 16.35 s |
| warm-3 | **10.932 s** | 46.01 | 7.622 / 7.324 s | 164 / 136 267 | 14.08 | 28796 / 8959 (76%) | 14.76 GiB | 10.92 GiB | 16.35 s |

4 chunks × 41 cb = 164 cb — the chunk-128 scratch allocation succeeded (no
token-major fallback; the cb count proves the schedule). 6.9% faster than
chunk 32; the curve is flattening as prefill becomes GPU-execution-bound
(wait 7.62 s of which GPU is 7.32 s).

### F — cache-size knob: chunk 32, 503-token prompt, `--cache-gb 20` (full pool cacheable)

| Run | TTFT | Prefill tok/s | Decode tok/s | Cache hits/misses (rate) | Bytes read | Peak RSS | Real |
|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 11.347 s | 44.33 | 15.38 | 56931 / 7018 (89%) | 11.57 GiB | 13.82 GiB | 16.40 s |
| warm-1 | **11.251 s** | 44.71 | **15.41** | 56931 / 7018 (89%) | 11.57 GiB | 14.46 GiB | 17.05 s |
| warm-2 | **11.293 s** | 44.54 | **15.49** | 56931 / 7018 (89%) | 11.57 GiB | 14.46 GiB | 16.34 s |
| warm-3 | **11.325 s** | 44.42 | **15.40** | 56931 / 7018 (89%) | 11.57 GiB | 14.48 GiB | 16.38 s |

With 10240 slots every miss is a first touch (7018 unique blobs, zero
evictions). Versus C (same schedule, 8 GB): long-context decode improves
13.86 → 15.43 tok/s (**+11%**), prefill ~3% faster, and 2163 eviction
re-reads (3.6 GiB of page-cache preads) disappear. Output identical to C.

## S3b breakdowns

Encode-side phase split is exact; wait/GPU can only be attributed per
command-buffer label because one buffer spans several phases (per-phase GPU
split is unsupported on M4 Max — encoder-boundary timestamps only).

Decode, A warm-1 (64 steps, totals):

| | attention | delta | moe | router | lmHead |
|---|---|---|---|---|---|
| dispatches | 5760 | 24960 | 97280 | 5120 | 128 |
| encode s | 0.002 | 0.007 | 0.029 | 0.002 | 0.000 |

| buffer label | cb | wait s | GPU s |
|---|---|---|---|
| attention+moe+router | 640 | 0.554 | 0.425 |
| delta+moe+router | 1856 | 1.157 | 0.814 |
| delta+router | 64 | 0.073 | 0.025 |
| moe+lmHead | 64 | 0.127 | 0.114 |

Decode step decomposition (A warm-1, 40.0 ms/step wall): GPU execution
21.5 ms + command-buffer scheduling latency (wait − GPU, 41 waits/step ≈
0.20 ms each) 8.3 ms + encode 0.6 ms + **unattributed CPU gap 9.5 ms** (expert
preads/memcpy, per-layer router readback + CPU top-k, argmax over the 248 320
vocab, Swift loop).

Same decomposition elsewhere:

| Configuration | wall | wait | GPU | encode | CPU gap (share) |
|---|---|---|---|---|---|
| A decode warm (per step) | 40.0 ms | 29.8 ms | 21.5 ms | 0.6 ms | 9.5 ms (24%) |
| A decode **cold** (per step) | 71.9 ms | 45.3 ms | 34.8 ms | 0.9 ms | 25.7 ms (36%) |
| C decode warm, 8 GB (per step) | 72.0 ms | 55.8 ms | 46.4 ms | 0.6 ms | 15.5 ms (22%) |
| F decode warm, 20 GB (per step) | 64.9 ms | 55.6 ms | 46.3 ms | 0.6 ms | 8.7 ms (13%) |
| B prefill (whole, warm-1) | 23.79 s | 19.69 s | 15.30 s | 0.37 s | 3.73 s (16%) |
| C prefill (whole, warm-1) | 11.67 s | 8.36 s | 7.98 s | 0.08 s | 3.23 s (28%) |
| D prefill (whole, warm-1) | 10.92 s | 7.62 s | 7.32 s | 0.05 s | 3.25 s (30%) |

Token-major prefill's extra 8 s over chunk 32 is almost entirely
command-buffer round-trips: wait−GPU = 4.39 s across 20 623 cb (0.21 ms each)
vs 0.38 s across 656 cb, plus lower GPU efficiency on single-token GEMVs
(15.3 s vs 8.0 s GPU for identical math).

## Output parity across schedules (production scale)

Mechanism reported honestly: **greedy token/text equality**, not a logit-level
comparison (the CLI surfaces no logits at 35B and an in-process dual-model
harness was out of scope for a docs-only run; numerical logit parity remains
pinned by the tiny-model gates on the S1b/S2 branches).

- 503-token natural prompt, 64 greedy tokens: stdout is **bitwise identical**
  across token-major, chunk 32, chunk 128 (`diff` of B/C/D `warm-1.out`), and
  also identical for F (chunk 32, 20 GB cache). The continuation is a coherent
  markdown table about MoE routing overhead.
- 200-id deterministic prompt (`--ids`, formula above), all three schedules:
  identical output — greedy token 220 followed by EOS. (EOS after one token
  limits this arm's sequence length; the 64-token text runs above carry the
  sequence-level claim.)
- A cold vs warm: identical continuation and identical hit/miss counts —
  routing is unaffected by cache temperature.

## S4 go/no-go: are storage/expert-fetch stalls a material cost?

**Verdict: NO-GO — defer S4 (async expert miss loading). The measured warm
bottleneck is command-buffer wait and GPU execution, not storage.**

Evidence from the S3a/S3b traces above:

1. **Warm decode wait dominates.** 41 blocking `waitUntilCompleted` per token
   spanning 74–78% of decode wall (29.8 of 40.0 ms/step short-context; 55.8 of
   72.0 ms/step long-context), of which GPU execution is 21.5 / 46.4 ms.
   Expert fetches do not sit inside these waits — they happen on the CPU
   thread before commit, inside the unattributed gap.
2. **The storage-attributable share is bounded and small when warm.** The
   entire CPU gap — which also contains router readback, CPU top-k, and the
   248 320-wide argmax — is 9.5 ms/step (24%) short-context and 15.5 ms/step
   (22%) long-context. The direct differential F−C (only variable: eviction
   re-reads, 3.6 GiB fewer preads) is **6.8 ms/step ≈ 9.5% of the step** —
   that is the actual marginal fetch cost at the default 8 GB budget, and it
   is removable by cache sizing alone (`--cache-gb 20`), no async loader
   needed on this machine.
3. **Prefill is GPU-bound, not storage-bound.** Layer-major chunk 128: wait
   7.62 s of 10.92 s wall with GPU execution 7.32 s of that wait; wait−GPU is
   only 0.30 s. Faster expert loading cannot shorten it materially.
4. **Storage is material exactly once: cold start.** After `purge`, the first
   pass preads 6.86 GiB from SSD: decode 13.92 vs 25.05 tok/s (1.80×
   penalty), TTFT 1.85 vs 0.59 s, and the CPU gap grows 9.5 → 25.7 ms/step
   with wait inflating alongside (I/O contending with the GPU on unified
   memory). On this 64 GB machine that is a one-time-per-boot window — the
   16.88 GiB pool then lives in page cache.

Phase-4 gate check ("add async miss loading only after timeline data
distinguishes storage wait from command dispatch and GPU compute"): the data
now distinguishes them — dispatch wait ~0.2 ms × 41 cb/token plus GPU
execution dominate; storage contributes ≤ ~10% warm and ~1.8× only when cold.
S4 should be revisited, not resurrected, when a trace from a RAM-constrained
target (iPhone-class device, or `--cache-gb 2` server default on a small-RAM
Mac where the pool cannot live in page cache) shows the cold-decode profile as
steady state. The higher-leverage next steps on this hardware are dispatch
reduction (fewer than 41 cb/token; batching the per-layer router readback) and
S2 attention tiling.

## Caveats

- Parity is greedy-sequence/text equality at production scale, not a
  35B logit-level diff (see mechanism note above).
- Per-phase GPU split is honestly unsupported on M4 Max (encoder-boundary
  counters only); wait/GPU attribution is per command-buffer label, and the
  CPU-gap decomposition is arithmetic (wall − wait − encode), an upper bound
  that mixes expert fetches with router readback and sampling.
- There is no dedicated storage-fetch timer in `StepMetrics`; storage cost was
  isolated by differentials (purged-cold vs warm; 8 GB vs 20 GB cache), which
  is indirect but consistent across all runs.
- "Warm" bytes read are page-cache preads (the pool fits in 64 GB RAM); on
  smaller-RAM machines the cold profile (1.8× decode penalty) is the realistic
  steady state and the S4 verdict would likely flip there.
- Decode throughput depends on context length (25.0 tok/s at ~16+64 ctx vs
  13.9 at 503+64 ctx with the default cache); single machine, single day;
  resident daemons (node_exporter, tailscaled) idle but present.
- The `--ids` parity arm produced only one token before EOS; sequence-level
  parity rests on the 64-token natural-prompt runs.
- Warmup runs are shown but excluded from claims; B-warm-0's 13.17 tok/s
  decode reflects residual page-cache warming.
- Raw logs (`*.out`/`*.err` per run, prompt file) retained on macbook4 in
  `~/bench-logs/`.

---

# Re-run 2026-08-31: conv scan + attend tiling (`3dfd004`), SG sweep, colibri head-to-head

Second benchmark on the same box, model, prompts, sampler, and protocol as
the baseline above (code `69820f7`, doc commit `0f6a9a8`). Everything in the
Environment and Protocol sections holds unchanged (same binary invocation,
same 8 GB default cache, same 1 warmup + 3 measured discipline, `ps aux`
idle-check logged before every timed run, builds fully separated from
measurement; raw logs in `~/bench-logs-3dfd004/` on macbook4). This section
measures the eight commits that landed since the baseline:

- `ef264e5` + `b4a986c` — token-grid batch twin for the causal attend;
  chunk pass attends batched across tokens.
- `7bc850f` + `848128f` — stride twins for the DeltaNet prep chain; prep
  chain batched across the chunk.
- `1cf3dbf` + `9661248` — cross-token scan twin for the depthwise conv;
  chunk pass conv encoded as one scan dispatch per layer.
- `d15ca75` + `3dfd004` — attention parity gate widened; causal attend
  tiled across KV rows over `ATTN_DECODE_SG` cooperating simdgroups with
  float4-vectorized K/V loads.

Benchmark commit: `3dfd004` (clean tree, rsync'd to `~/build/s1c-swiftlet`),
followed by the sweep-driven `7352871` (below). Swift 6.3.3, macOS 26.6.2,
same M4 Max.

## Headline vs the 69820f7 baseline (warm, greedy, 64 new tokens, 8 GB cache)

| Metric | `69820f7` | `3dfd004` | delta |
|---|---|---|---|
| Decode, short context (16-tok prompt) | 25.00-25.08 tok/s | **26.36-26.45 tok/s** | **+5.4%** |
| Decode, long context (503-tok prompt, chunk 32) | 13.79-13.89 tok/s | **21.57-21.87 tok/s** | **+57%** |
| TTFT, 503-tok prompt, chunk 32 (default) | 11.67-11.70 s | **8.76-8.84 s** | **-24.7%** |
| TTFT, 503-tok prompt, chunk 128 | 10.92-10.93 s | **8.17-8.18 s** | **-25.1%** |
| TTFT, 503-tok prompt, token-major | 23.79-23.81 s | **17.43-17.68 s** | **-26.5%** |
| Prefill dispatches, chunk 32 | 226 729 | **148 809** | -34% |
| Prefill dispatches, chunk 128 | 136 267 | **56 427** | -59% |
| Greedy outputs vs the 69820f7 logs | — | **bitwise identical, every arm** | — |

## Results (same run-set labels as the baseline)

### A — decode-focused: 16-token prompt, 64 new, chunk 32, 8 GB cache

| Run | TTFT | Decode tok/s | Decode wait / GPU (64 steps) | Hits/misses (rate) | Bytes read | Peak RSS | Real |
|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 0.903 s | 22.25 | 1.910 s / 1.369 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 5.01 s |
| warm-1 | 0.580 s | **26.38** | 1.773 s / 1.234 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 3.83 s |
| warm-2 | 0.586 s | **26.36** | 1.775 s / 1.235 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 3.83 s |
| warm-3 | 0.586 s | **26.45** | 1.765 s / 1.234 s | 18429 / 4160 (82%) | 6.86 GiB | 9.67 GiB | 3.83 s |

Decode schedule unchanged: 41 cb / 2082 dispatches per token (2624 cb /
133 248 per 64 steps) — the decode gain is per-dispatch, not fewer
dispatches. The 16-token prefill dropped 9879 → 7479 dispatches (-24%):
attention 240 → 90 (batched attends), delta 2670 → 420 (conv scan + prep
chain), moe/router/lmHead unchanged — the conv scan moves dispatch counts
exactly as designed. Cache hits/misses identical to the baseline.

### B — token-major prefill (`--prefill-chunk 0`): 503-token prompt, 64 new

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | cb / dispatches | Decode tok/s | Peak RSS | Real |
|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 17.684 s | 28.44 | 13.123 / 9.141 s | 20623 / 1 046 242 | 21.09 | 10.91 GiB | 21.58 s |
| warm-1 | **17.431 s** | 28.86 | 13.090 / 9.106 s | 20623 / 1 046 242 | 21.31 | 10.95 GiB | 21.32 s |
| warm-2 | **17.522 s** | 28.71 | 13.241 / 9.167 s | 20623 / 1 046 242 | 21.13 | 10.93 GiB | 21.44 s |
| warm-3 | **17.512 s** | 28.72 | 13.300 / 9.212 s | 20623 / 1 046 242 | 21.04 | 10.92 GiB | 21.44 s |

Same 20 623 cb and 1 046 242 dispatches as the baseline (token-major takes
the per-token path; batching/scan don't apply), so this arm isolates the
attend tiling: prefill GPU execution 15.29 → 9.11 s (-40%) with identical
math, TTFT -6.3 s. Cache counters unchanged (171983 / 9457, 95%).

### C — layer-major chunk 32 (default): 503-token prompt, 64 new

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | cb / dispatches | Decode tok/s | Peak RSS | Real |
|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 8.776 s | 57.32 | 5.479 / 5.163 s | 656 / 148 809 | 22.05 | 10.90 GiB | 12.54 s |
| warm-1 | **8.781 s** | 57.28 | 5.511 / 5.190 s | 656 / 148 809 | 21.87 | 10.91 GiB | 12.58 s |
| warm-2 | **8.763 s** | 57.40 | 5.480 / 5.148 s | 656 / 148 809 | 21.61 | 10.91 GiB | 12.60 s |
| warm-3 | **8.836 s** | 56.93 | 5.538 / 5.190 s | 656 / 148 809 | 21.57 | 10.91 GiB | 12.67 s |

Dispatches 226 729 → 148 809 (-34%; prefill phases now attention 1440,
delta 6720, moe 139 367, router 1280) at the same 656 cb. Layer-major
chunk 32 vs token-major is now 1.99× on TTFT (was 2.04×; both schedules
sped up). Cache counters identical to baseline (54768 / 9181, 86%).

### D — layer-major chunk 128: 503-token prompt, 64 new

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | cb / dispatches | Decode tok/s | Peak RSS | Real |
|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 8.182 s | 61.48 | 4.876 / 4.583 s | 164 / 56 427 | 22.33 | 10.91 GiB | 11.93 s |
| warm-1 | **8.182 s** | 61.48 | 4.880 / 4.588 s | 164 / 56 427 | 22.43 | 10.91 GiB | 11.91 s |
| warm-2 | **8.170 s** | 61.57 | 4.878 / 4.581 s | 164 / 56 427 | 22.45 | 10.91 GiB | 11.89 s |
| warm-3 | **8.184 s** | 61.46 | 4.880 / 4.588 s | 164 / 56 427 | 22.34 | 10.91 GiB | 11.94 s |

Dispatches 136 267 → 56 427 (-59%): the bigger the chunk, the more the
scan/batching amortize. Chunk 128 over chunk 32 is now +7.4% (was +6.9%).

## Decode step decomposition (vs baseline)

| Configuration | wall | wait | GPU | encode | CPU gap (share) |
|---|---|---|---|---|---|
| A decode warm (per step), 69820f7 | 40.0 ms | 29.8 ms | 21.5 ms | 0.6 ms | 9.5 ms (24%) |
| A decode warm (per step), 3dfd004 | 37.9 ms | 27.7 ms | 19.3 ms | 0.6 ms | 9.6 ms (25%) |
| C decode warm (per step), 69820f7 | 72.0 ms | 55.8 ms | 46.4 ms | 0.6 ms | 15.5 ms (22%) |
| C decode warm (per step), 3dfd004 | 45.7 ms | 29.5 ms | 20.8 ms | 0.6 ms | 15.6 ms (34%) |

The attend tiling removed ~26 ms of GPU execution per long-context step —
long-context decode GPU time is now nearly flat versus short context
(20.8 vs 19.3 ms/step; it was 46.4 vs 21.5). The unattributed CPU gap is
untouched (same expert preads, router readback, argmax) and is now the
largest single lever at long context (34% of the step), ahead of dispatch
wait (wait − GPU ≈ 8.7 ms) — reinforcing the baseline's "dispatch
reduction + CPU-side work" priority over any storage work (S4 stays NO-GO;
nothing in this re-run moves that verdict).

## ATTN_DECODE_SG sweep (2 / 4 / 8 simdgroups per (head, token))

Decode case, 1 warmup + 3 measured per point, separate binaries + resource
bundles built per value (both the Swift dispatch width and the kernel's
`#define` move together), builds separated from measurement:

| SG | Decode tok/s, 16-tok prompt | Decode tok/s, 503-tok prompt (chunk 32) | Decode GPU s/64 steps (long) |
|---|---|---|---|
| 2 | 26.10-26.19 | 20.80-21.00 | 1.455-1.456 |
| 4 (3dfd004 default) | 26.36-26.45 | 21.57-21.87 | 1.328-1.332 |
| 8 | **26.42-26.51** | **22.06-22.12** | 1.282-1.283 |

SG=8 wins both: +1.9% decode at 567-row KV, level-to-slightly-ahead at
80-row KV. Greedy outputs bitwise identical across all three values on both
prompts. Committed as **`7352871`** ("Raise the attend tiling to eight
simdgroups per (head, token)", the two matched constant lines only); the
full test suite (96 tests / 18 suites, CLT framework flags) passes at that
commit on this machine. The A/B/C/D tables above are the pre-sweep
`3dfd004` numbers; SG=8 adds its ~+2% on top of the long-context decode
column.

## Output parity vs the 69820f7 run (production scale)

Same mechanism as the baseline (greedy token/text equality, not logit
diff), now also a cross-code-revision check. The attend tiling changed the
attention reduction order (strided per-simdgroup online softmax merged by
exp-rescale, float4 dots), so greedy tie flips were possible — none
occurred:

- A/B/C/D `warm-1..3` stdout: **bitwise identical to the corresponding
  69820f7 logs** (`diff` against `~/bench-logs/*.out`), and identical
  across schedules (B = C = D md5) and across SG = 2/4/8.
- `--ids` parity arm (200 deterministic ids, chunk 0/32/128): identical to
  the baseline logs.
- Cache hit/miss counts identical in every arm — routing untouched.

## Colibri head-to-head (same box, same qpack container, same prompt)

Colibri's C engine (`~/build/c3-colibri`, mirror of branch tip `08c24b2`;
the mirror carries no git metadata, so the commit is taken from the sync
record, not verified against a local clone). Rebuilt for this run:
`make -C c qwen36 METAL=1` — OpenMP engaged (`-Xclang -fopenmp` +
Homebrew `libomp` linked, 8 parallel regions; no single-thread warning)
and the C3 qpack Metal expert path compiled in (`-DCOLI_METAL
-DQWEN36_QPACK`). Run line: `SNAP=<snap> QWEN36_QPACK=<qpack> N_NEW=n
./qwen36 n 4 <prompt file>`, `COLI_TIMERS=1`, greedy, same fieldfare
16-token prompt (colibri tokenizes it to 16 tokens, matching Swiftlet's count; both engines open with the same first two greedy tokens)
and the standard 5-token "capital of France" smoke.

### Decode-focused: 16-token prompt, 64 new tokens

| Run | TTFT | Whole-run tok/s | Decode step time | Peak RSS | Real |
|---|---|---|---|---|---|
| warm-0 (warmup) | 2.00 s | 4.76 | 181.4 ms/tok | 11.31 GiB | 15.33 s |
| warm-1 | **1.92 s** | 4.84 | 179.1 ms/tok | 11.31 GiB | 14.84 s |
| warm-2 | **1.89 s** | 4.84 | 179.5 ms/tok | 11.31 GiB | 14.84 s |
| warm-3 | **1.95 s** | 4.82 | 179.3 ms/tok | 11.31 GiB | 14.89 s |

"Whole-run tok/s" is the harness's `Speed` (64 tokens over the full
generate call including prefill); decode-only from the step timer is
**5.57-5.58 tok/s** (63 steps at 179.1-179.5 ms). Phase split (warm-1,
ms/token): deltanet 65.3 (of which projections 45.4), moe 74.2 (shared
12.6, router 3.1), lm_head 22.0, attention 17.0. CPU threading is real:
94.0 s user / 14.84 s real ≈ 6.3 cores busy on the 12P+4E M4 Max.

qpack counters, identical across all four runs: `metal=75840 cpu=0`
(every routed projection on Metal, zero CPU fallbacks), `blocks=2640
pairs=25280 binds=23266 stalls=0 redirects=0 fallbacks=0`,
**`stage_reuse=2640 stage_grow=0`** (every begin served from a bound
arena, no steady-state growth), `slots=96 fills=22264 evictions=22168`
(the 96-slot bounded pool churns hard at this cache size — the legacy
snapshot expert cache reports 0/0 because qpack mode bypasses it).

### Smoke: 5-token prompt ("The capital of France is"), 16 new

TTFT 0.68 s (0.72 warmup), Speed 4.74-4.75 tok/s, RSS 11.31 GiB;
counters `metal=19200 cpu=0`, `stage_reuse=640 stage_grow=0`, fills=5914 /
evictions=5818. Output stable across runs: " Paris, a city renowned for
its iconic landmarks such as the Eiffel Tower,".

### Head-to-head (identical prompt/params: fieldfare 16-tok, 64 new, greedy)

| | Swiftlet `3dfd004`+SG8 | colibri `08c24b2` | ratio |
|---|---|---|---|
| Decode tok/s (short ctx) | **26.4-26.5** | 5.57-5.58 (step timer) | **4.7×** |
| TTFT (16-tok prompt) | **0.58 s** | 1.89-1.95 s | 3.3× |
| Peak RSS | **9.67 GiB** | 11.31 GiB | 0.86× |
| Routed experts | Metal (bounded 8 GB cache) | Metal via qpack (96 slots) | — |
| Attention / DeltaNet / dense / lm_head | Metal | CPU (OpenMP, int8 dense) | — |

Architectural caveat, stated plainly: this is not a like-for-like GPU
race. Colibri's C3 stage puts only the routed-expert projections on
Metal by design; attention, DeltaNet, the shared expert, router, and the
248 320-wide lm_head all run on CPU threads, and its decode is
correspondingly CPU-bound (deltanet + moe + lm_head ≈ 161 of 179
ms/token). Swiftlet runs the whole forward pass on the GPU. The
comparison pins where the colibri port is on the same container today,
not a verdict on its ceiling.

**Cross-engine output divergence (surfaced, not smoothed):** colibri's
64-token fieldfare continuation is byte-identical across its own runs
but **diverges from Swiftlet's at the 3rd generated token** — both open
" southern Europe", then Swiftlet continues ". It is a common winter
visitor to the UK, …" while colibri continues " and Britain. It is a
migratory bird, …". The two engines compute in different arithmetic
(colibri: CPU f32 with int8-quantized dense matrices; Swiftlet: Metal
kernels), so a greedy tie flipping between them is expected behaviour
and is not a parity claim either engine makes; each engine's own
determinism and its own parity gates (Swiftlet: bitwise vs its 69820f7
logs; colibri: metal=…/cpu=0 with fallbacks=0) all hold. No pinned
cross-engine text exists in either repo's docs; the "byte-identical"
note in the prior colibri session's record matches its internal
qpack-vs-reference parity gate, not a cross-engine pin.

## Caveats (delta from the baseline's list, which still applies)

- The A-D tables above are `3dfd004` with the then-default SG=4; SG=8
  (`7352871`) adds ~+1.9% long-context decode on top and is the new
  default. Short-context A numbers under SG=8: 26.42-26.51 tok/s.
- Colibri's decode tok/s depends on which number you quote: 4.8 whole-run
  vs 5.6 step-timer; both are given above. The earlier session's "~4.4
  tok/s" was measured before this rebuild; today's binary is faster and
  the number to carry forward is 5.57-5.58 step-timer / 4.82-4.84
  whole-run.
- Colibri `cache=64/layer bits=4` per the established run line; its
  96-slot qpack pool evicts heavily (22 168 evictions / 64 tokens), so
  its expert-fetch regime is not comparable to Swiftlet's 8 GB cache.
- Single day, single machine; same resident daemons as the baseline.
