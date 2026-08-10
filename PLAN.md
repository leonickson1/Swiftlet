# Swiftlet: an 80B Qwen on every Apple Silicon Mac

A native Swift + Metal runtime that runs Qwen3-Next-80B-A3B (and later Qwen3.5-397B-A17B)
on ordinary Macs by keeping only the dense core resident and streaming the routed MoE
experts from SSD per token. TurboFieldfare's architecture, colibri's policies, Qwen's models.

Working name "Swiftlet" (a small, extremely fast bird; fits the fieldfare/colibri lineage
and the language pun). Rename freely.

## 1. Why this is buildable in days, not months

TurboFieldfare took 103 logged experiments because its author had no answer key. We have three:

1. **mlx-lm has complete Python implementations** of both target architectures:
   `mlx_lm/models/qwen3_next.py` and `qwen3_5_moe.py`. Every kernel we write gets validated
   against these, layer by layer. The Gated DeltaNet math is no longer research; it is porting.
2. **llama.cpp merged Qwen3-Next support on 2025-11-28** (PR #16095): a second reference,
   in C++, including CPU gated-delta-net kernels we can crib the recurrence from.
3. **TurboFieldfare + colibri are the streaming answer key.** TF's container format, streaming
   installer, pread expert cache, and phase-overlap decode loop are model-agnostic and copyable
   almost verbatim (file list in section 7). colibri documents which cache/IO/quant policies
   survived adversarial benchmarking, including the dead ends (section 8).
4. **Pre-quantized MLX 4-bit checkpoints already exist** for all three target models
   (lmstudio-community / mlx-community). TF's quant format IS MLX affine 4-bit group-64,
   so our installer repacks existing quantized bytes unchanged: no quantization pipeline
   to build, and quality is identical to what MLX users already accept.
5. **Qwen weights are Apache 2.0.** No Gemma-style terms friction.

The wedge vs. the obvious competition: LM Studio / mlx-lm run the 80B only on 48-64 GB+
Macs (the whole 42 GB model must fit in unified memory). Swiftlet targets the 8-24 GB Macs,
which is most Macs sold.

## 2. Target models (one engine, one architecture family)

All three are the same hybrid design: 3 Gated DeltaNet linear-attention layers followed by
1 gated GQA full-attention layer, repeating; every layer has a high-sparsity routed MoE
(top-k of 256-512 experts) plus 1 shared expert. Verified from the HF configs:

| | Qwen3-Next-80B-A3B (primary) | Qwen3.5-397B-A17B (stretch) | Qwen3.6-35B-A3B (later, low-end Macs) |
|---|---|---|---|
| arch / model_type | `qwen3_next` | `qwen3_5_moe` | `qwen3_5_moe`-family |
| layers | 48 (36 linear + 12 full) | 60 (45 + 15) | 40 (30 + 10) |
| hidden | 2048 | 4096 | 2048 |
| experts, top-k | 512, 10 (+1 shared) | 512, 10 (+1 shared) | 256, 8 (+1 shared) |
| moe_intermediate | 512 | 1024 | 512 |
| full-attn heads | 16 Q / 2 KV, head_dim 256, partial RoPE 0.25, theta 1e7 | 32 Q / 2 KV, dim 256 | 16 Q / 2 KV, dim 256 |
| DeltaNet heads | 32 V / 16 K, dims 128/128, conv 4 | 64 V / 16 K, dims 128/128, conv 4 | 32 V, dim 128, conv 4 |
| vocab (untied) | 151,936 | 248,320 | 248,320 |
| context | 262,144 native | 262,144 | 262,144 |
| MTP head | yes (external to config) | yes, 1 layer | check |
| int4 total | ~42-44 GB | ~215-220 GB | ~18 GB |

Repack sources: `lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit`,
`mlx-community/Qwen3.5-397B-A17B-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`.
Kernel-dev fixtures: `tiny-random/qwen3.5-moe` (and a tiny-random qwen3_next, or generate
one with transformers in minutes).

## 3. Memory and speed budget (80B on the M5 / 24 GB dev machine)

Per-expert blob: 3 matrices (gate/up/down) x 2048 x 512 = 3.15M params -> ~1.77 MB at
int4 g64, ~1.8 MB page-aligned stride. Layer file: 512 x 1.8 MB ~ 920 MB; 48 layers ~ 44 GB.

Resident set:
- Dense weights (embeddings in+out, 12 GQA attn layers, 36 DeltaNet projection sets,
  48 shared experts, routers, norms): ~2.8B params -> ~1.6-2 GB at checkpoint precision.
- KV cache: only the 12 full-attention layers keep KV. 2 KV heads x 256 x 2(K,V) x fp16
  = ~25 KB/token -> 100 MB at 4K, 800 MB at 32K. (TF needed a sliding-window ring; we
  mostly don't have the problem: 75% of layers have NO growing KV at all.)
- DeltaNet recurrent state: fixed 36 x 32 x 128x128 x f32 ~ 75 MB, constant at any context.
  Bonus: persist it a la colibri `.coli_kv` and conversations reopen warm instantly.

Total resident ~2.5-3 GB. The rest of RAM is expert cache: on 24 GB, a ~14 GB cache holds
~32% of all 24,576 expert blobs.

IO per token: 10 experts x 48 layers = 480 blobs, ~860 MB worst-case cold. On a ~6 GB/s
internal SSD that is a ~7 tok/s cold floor; with a warm cache and routing locality
(colibri measures 71.6% next-layer predictability; TF's LFU cache converged well above
that on repeated workloads) the realistic warm range on this M5 is well into the teens
to 20+ tok/s. On 8 GB Macs (~4 GB cache, 9% resident) expect TF-like mid-single-digit
tok/s. On 48-64 GB Macs the cache converges to full residency and disk drops out.

397B stretch on the same math: 600 blobs x ~7.3 MB ~ 4.4 GB/token cold, ~6-7 GB dense
resident, ~220 GB container -> external NVMe territory, ~1-1.5 tok/s cold, a few warm.
Patient-use, but "a 397B frontier model on a MacBook" is the headline that markets the app.

## 4. System design

Four products in one SwiftPM package, mirroring TF's proven structure:
`SwiftletCore` (runtime + Metal), `swiftlet-repack` (streaming installer CLI),
`swiftlet` (chat/completion CLI), `SwiftletServer` (loopback OpenAI-compatible),
`Swiftlet.app` (SwiftUI, plus TF's out-of-process decode-service pattern so the GUI
never shares an address space with the model).

### 4.1 Container: `.qpack` (clone of `.gturbo`)

- `manifest.json` (magic, arch block checked field-by-field, quant block, SHA-256 file map),
  `model_weights.bin` (all resident tensors, mmap'd read-only, wrapped
  `makeBuffer(bytesNoCopy:)`), `tokenizer/`, `packed_experts/layout.json` +
  `layer_XX.bin` per layer.
- One expert = one contiguous fixed-stride blob: gate/up/down x {packed int4, scales,
  biases} back-to-back, 16 KB aligned. Fetch = exactly one
  `pread(fd, slot, stride, base + id*stride)`. `layout.json` records intra-blob offsets
  so kernels bind sub-ranges of the slot buffer.
- Installer = TF's `RemoteStreamingRepacker` pattern: read safetensors index + headers
  only, plan byte-range copies, stream bounded HTTP Range requests from the HF MLX-4bit
  repo directly into final file offsets through a 512 KB scratch tile. Never materializes
  a shard, never dequantizes (colibri rule: never re-encode trained/quantized bytes).
  Resumable via checkpointed range digests; atomic promote at the end.

### 4.2 Decode loop (TF's cb1/io/cb2, adapted)

Per layer per token:
1. **cb1 (Metal):** norm -> attention. For 12 GQA layers: QKV + gates, partial RoPE (0.25,
   theta 1e7), fp16 KV append, split-KV online-softmax attention, output gate, O-proj.
   For 36 DeltaNet layers: in-proj, short conv (k=4) state step, L2-norm q/k, gated
   delta-rule state update S <- a*S + b*k(v - Sk)^T, output gate + O-proj. Tiny math
   (~0.5M MAC/head), trivially a Metal kernel. Then router GEMV -> top-10 IDs.
2. **CPU:** read IDs, plan the layer's expert cache (hits/misses/slot assignment under lock).
3. **io:** parallel `pread` of misses into page-aligned slots (`posix_memalign`, bytesNoCopy
   Metal wrapping); meanwhile Metal computes the **shared expert**, hiding read latency.
4. **cb2 (Metal):** routed SwiGLU over 10 slot buffers, weighted reduce, add shared branch,
   layer tail.

Expert cache: per-layer slot pool, LFU with recency tie-break (TF benchmarked LFU > LRU).
TF used 16 slots for 8+1 routing; we route 10+1 of 512, start at 24-32 slots/layer and
let the pool auto-grow to fill free RAM (colibri lesson: the RAM cap, not the disk, is
usually the binding constraint; slot floors that are too low waste a fast NVMe).

Prefill: chunked 64-128 tokens, layer-major, **batch-union** expert loading (each unique
expert read once per chunk, colibri measured 2.7x dedup at chunk 32). DeltaNet prefill v1
is the plain recurrent scan (correct, sequential); the chunked/WY parallel form is a
later optimization with mlx-lm/llama.cpp as reference. GQA prefill via TF's TensorOps
multi-head path.

### 4.3 Metal kernel inventory

Port from TF (same quant format, same patterns): int4/int8 affine g64 GEMV with
alignment-aware loads, RMSNorm, NeoX RoPE (partial factor), split-KV decode attention,
router GEMV + top-k select, MoE two-phase (persistent-workgroup, NOT simd-cooperative;
swap GeGLU -> SwiGLU), fused layer-tail, sampling (top-k/top-p/Gumbel, fused greedy head),
staged affine->fp16 matmul for prefill.

New (the actual work): gated-DeltaNet decode step (conv step + delta-rule state update +
gating), DeltaNet recurrent prefill scan, gated-attention output gate, and the qwen3_5_moe
variants' minor differences (zero-centered layernorm flavor; verify against mlx-lm source).

### 4.4 Correctness harness (non-negotiable, colibri's core rule)

- Python fixture generator: run mlx-lm (and/or transformers) on tiny-random models and on
  the real checkpoint, dump per-layer activations + logits for fixed prompts.
- Every Swift/Metal kernel validated against fixtures before integration; whole-model gate
  is greedy teacher-forcing token match vs mlx-lm running the SAME 4-bit checkpoint
  (apples-to-apples: our quality == MLX ecosystem quality by construction).
- Placement/caching must never change semantics: an expert answers identically from RAM
  or disk. No silent expert substitution on eviction (colibri's Inkling bug: reserve
  slots with sentinels, dedup in-flight loads, round-based acquisition).

## 5. Milestones (focused days)

- **M0 (day 1):** Repo scaffold, products wired. Vendor mlx-lm's two model files +
  llama.cpp's qwen3-next graph into `references/`. Fixture generator + tiny-random
  fixtures. Tokenizer via swift-transformers (Qwen BPE + chat template via swift-jinja,
  exactly TF's dependency set).
- **M1 (days 1-2):** `.qpack` format + repacker v1 from a LOCAL MLX checkpoint directory
  (HTTP-range streaming variant after). Manifest, blob packing, verify mode.
- **M2 (days 2-4):** CPU reference forward pass in Swift (f32 dequant, Accelerate),
  validated per-layer on tiny-random, then real model. Gate: teacher-forcing match.
  This de-risks ALL model math before any Metal is written.
- **M3 (days 4-6):** Metal decode path + streamed expert cache + cb1/io/cb2 overlap.
  Gate: greedy output byte-identical to the CPU reference; first tok/s numbers.
- **M4 (days 6-8):** Prefill (batch-union tiles, recurrent DeltaNet), CLI chat,
  OpenAI-compatible server (port TF's swift-nio loopback server).
- **M5 (days 8-10):** Mac app: SwiftUI, our design system (monochrome near-black accent,
  no grey circles, liquid-glass composer), streaming installer UX with resume, live
  decode/memory stats. Benchmarks doc.
- **Perf passes (ongoing, data-driven):** PILOT-style layer-ahead prefetch (run layer
  L+1's router on L's post-attention state; colibri: 71.6% recall vs 41.3% for
  last-token reuse), learned pinned hot-store from usage history, chunked DeltaNet
  prefill, command-buffer batching (Apple ~5 ms submit latency), passive-wait CPU
  threads (colibri: a spinning CPU throttles the GPU 39% via the shared power envelope),
  `F_NOCACHE` for true zero-copy GPU feeding (macOS has no O_DIRECT).
- **Stretch:** Qwen3.5-397B-A17B (same engine, bigger config; MTP head at int8 for
  speculative decode, never int4: both TF-adjacent projects measured int4 MTP collapse),
  Qwen3.6-35B-A3B for 8 GB Macs, warm-conversation persistence, iOS build.

## 6. Known constraints and risks

| Risk | Mitigation |
|---|---|
| Dev machine has only 18 GB free disk; 80B container is ~44 GB | Kernel dev needs only tiny-random fixtures (MBs). For real-model runs: external NVMe (TB4/5 does 3+ GB/s, also a shipping target config) or free ~50 GB. Decide at M2. |
| DeltaNet chunked prefill complexity | Ship recurrent scan first (correct, slower long prompts); optimize later against two references. |
| 4-bit quality on tiny 3.15M-param experts | We inherit the MLX checkpoint's bytes and validate vs mlx-lm on the same checkpoint; judge on real-text logits, not synthetic SNR (colibri). Dense/router/norm tensors stay at checkpoint precision; embeddings can move to int8 if teacher-forcing shows drift. |
| macOS 26-only (Metal 4) | Same call TF made; Xcode 26.6 on this machine. Enthusiast tool, fine. |
| 8 GB Macs | Target 16-24 GB first; 8 GB is a tuning pass (smaller slot pools, 4K ctx), not a design change. |
| TF dead ends | Do not repeat: quantized KV cache, simd-cooperative expert kernels, monolithic fusions, cross-layer expert prefetch by trace (adjacent layers share ~7% experts; router-lookahead prediction is the way), mmap demand paging for experts (8x slower than explicit pread). |

## 7. Copy list from TurboFieldfare (cloned at scratchpad/turbo-fieldfare)

`Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift` (blob packing),
`.../Remote/RemoteStreamingRepacker.swift` (streaming installer),
`Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` (slot cache),
`.../ModelIO/{PackedExpertsLayout,ManifestReader,Quantization}.swift`,
`Sources/TurboFieldfare/Metal/{MoE/moe.metal,TensorCore/tensorops.metal,Fusions/fused.metal,Logit/logit.metal}`,
server + decode-service IPC (`DecodeProtocol.swift`), `docs/OPTIMIZATION_JOURNEY.md`
(the failure list). Apache 2.0 with attribution; keep a THIRD_PARTY_NOTICES.

Gemma-specific, do NOT port: KVCacheManager's sliding-window ring, attention.metal's
K=V quirk and softcap, ArchConfig constants.

## 8. Policy adoptions from colibri (cloned at scratchpad/colibri)

Auto-growing per-layer cache to fill RAM; learned pinned hot-store with hysteresis
(25%+4, swap cap) to avoid thrash; PILOT layer-ahead router prediction from a dedicated
IO thread; batch-union prefill; one-pread coalesced expert blobs read in disk-offset
order; extract the cache/IO engine as a real module from day one (their 500 KB monolith
regret); decode-only expert-budget trims (never trim the prefill union: corrupts KV);
check router skew before betting on pin caches (flat routers cap hit rates).
