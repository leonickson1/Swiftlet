# Swiftlet

<a href="https://trendshift.io/repositories/102405?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-102405" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/102405/daily?language=Swift" alt="leonickson1%2FSwiftlet | Trendshift" width="250" height="55"/></a>

![Platforms](https://img.shields.io/badge/platforms-macOS%2014%2B%20%7C%20iOS%2017%2B-blue)
![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)
![Metal](https://img.shields.io/badge/GPU-Metal-8A2BE2)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
[![App Store](https://img.shields.io/badge/App%20Store-Priv%20AI-black?logo=apple)](https://apps.apple.com/us/app/priv-ai/id6765706001)
[![35B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen3.6--35B--qpack-yellow)](https://huggingface.co/Leonickson/Qwen3.6-35B-A3B-qpack)
[![80B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen3--Next--80B--qpack-yellow)](https://huggingface.co/Leonickson/Qwen3-Next-80B-A3B-qpack)
[![Built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-D97757)](https://claude.com/claude-code)

**Run 35B and 80B Qwen models on ordinary Apple devices, including iPhones.**

Swiftlet is a Swift + Metal runtime for the Qwen3-Next and Qwen3.5/3.6 MoE
hybrid model family. It keeps only the small dense core of a model resident in
memory and streams the routed Mixture-of-Experts weights from storage on
demand. The result:

| Model | Disk | Peak RAM | Decode speed (M5 Mac) |
|---|---|---|---|
| [Qwen3.6-35B-A3B, 4-bit](https://huggingface.co/Leonickson/Qwen3.6-35B-A3B-qpack) | 18 GB | 2.6 GB | 7 to 11 tok/s |
| [Qwen3.6-35B-A3B, 8-bit](https://huggingface.co/Leonickson/Qwen3.6-35B-A3B-8bit-qpack) | 34 GB | 7.6 GB | 3.5 to 4 tok/s |
| [Qwen3-Next-80B-A3B, 4-bit](https://huggingface.co/Leonickson/Qwen3-Next-80B-A3B-qpack) | 42 GB | 4.3 GB | 4.5 to 5 tok/s |

Those figures are on an M5. For a low end anchor, a base M1 (8-core GPU, 16 GB)
decodes the 4-bit 35B at about 2.45 tok/s and the 8-bit at about 1.74 tok/s, and
prefill runs at roughly decode speed, so a long system prompt is expensive on
this class of machine. On so few GPU cores the decode loop is fully compute
bound, so the expert cache is a memory knob rather than a speed one there:
`--cache-gb 2` matches `--cache-gb 8` while saving several GB of RAM, which is
the better setting on a 16 GB machine.

At the other end, an M4 Max (40-core GPU, 64 GB) decodes the 4-bit 35B at about
19.5 tok/s. The configuration that shows why streaming matters is the 8-bit 80B:
its 78.8 GiB of weights cannot be resident on a 64 GB machine at all, yet
Swiftlet serves it in 6.9 GiB at about 4.8 tok/s, and the 4-bit 397B (207.6 GiB)
runs in 12.6 GiB at about 1.4 tok/s. Decode speed tracks active parameters
(about 3B for both the 35B and 80B, about 17B for the 397B) far more than
container size. Datapoints measured by @Avicennasis (#18).

The 8-bit 35B is the quality tier for Macs: tested head to head on identical
prompts, it removes the repetition artifacts the 4-bit build can show on
longer writing tasks, at the cost of disk, RAM, and speed.

The 35B also runs on an iPhone 17 in about 2.5 GB of RAM, at about 1 tok/s
today. Credit where due: [ANEMLL](https://www.anemll.com/) showed a 397B
MoE streaming on an iPhone 17 Pro as a proof of concept in early 2026.
Swiftlet's aim is the next step, making this class of model an installable
app on a base iPhone, with an open runtime anyone can build on.

Status: working end to end. Both models generate correct, validated output.
The current focus is kernel speed (the decode loop is dispatch bound, not
IO bound, so there is clear headroom). One expectation to set honestly: only
about 3B parameters are active per token, so these models chat and write
like large models but recall facts like small ones.

## Quick start: try it on a Mac

```bash
git clone https://github.com/leonickson1/Swiftlet.git && cd Swiftlet
swift build -c release

# Download the 35B container from the R2 mirror (fast and consistent;
# Hugging Face throttles anonymous downloads harder the longer they run):
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-qpack \
  --output ~/models/qwen3.6-35b.qpack

# Or the 80B (42 GB on disk, still only ~4.3 GB of RAM):
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3-next-80b-qpack \
  --output ~/models/qwen3-next-80b.qpack

# Or the 8-bit 35B (best writing quality, 34 GB disk, ~7.6 GB RAM):
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-8bit-qpack \
  --output ~/models/qwen3.6-35b-8bit.qpack

# The same containers are on Hugging Face (anonymous downloads are
# rate-limited there, so the mirror is usually much faster):
#   --from-hf Leonickson/Qwen3.6-35B-A3B-qpack
#   --from-hf Leonickson/Qwen3.6-35B-A3B-8bit-qpack
#   --from-hf Leonickson/Qwen3-Next-80B-A3B-qpack

# Verify a finished download against the source hashes (optional):
#   python3 scripts/verify_container.py <base-url or org/repo> <container-dir>

# Chat (applies the model chat template, disables the reasoning block,
# keeps conversation state so follow-ups prefill only the new turn):
.build/release/swiftlet chat ~/models/qwen3.6-35b.qpack \
  "What is the capital of Spain?" "And what about France?"

# One-shot generation with stats:
.build/release/swiftlet generate ~/models/qwen3.6-35b.qpack \
  --gpu --chat --prompt "Explain expert streaming in one paragraph."

# Save the conversation state (KV, DeltaNet recurrence, position) and resume
# it later in a new process, continuing exactly from the saved state
# (docs/STATE_FORMAT.md):
.build/release/swiftlet generate ~/models/qwen3.6-35b.qpack \
  --gpu --prompt "The fieldfare is a bird that" --save-state /tmp/s.swlstate
.build/release/swiftlet generate ~/models/qwen3.6-35b.qpack \
  --gpu --prompt " It" --load-state /tmp/s.swlstate

# OpenAI-compatible server (loopback only):
.build/release/swiftlet-server --model ~/models/qwen3.6-35b.qpack --port 8080
```

The same command also repacks raw MLX checkpoints
(`--from-hf mlx-community/...` or `--source /path/to/checkpoint`).

Requirements: Apple Silicon, macOS 14+ or iOS 17+, free SSD space for the
container (18 GB for the 35B, 34 GB for the 8-bit 35B, 42 GB for the 80B).

## Try it on your phone

The 35B runs on iPhone inside
**[Priv AI on the App Store](https://apps.apple.com/us/app/priv-ai/id6765706001)**:
open Settings, then Experimental Models, and download the model. It streams
from storage and chats on-device with no server involved.

The Experimental Models feature ships in the newest app version, which is
still in App Store review, so it may not appear for a couple of days. If you
want the phone experience today, build the app from source: the app is open
source at [leonickson1/localLLM](https://github.com/leonickson1/localLLM).
Clone this repo next to it as `swiftlet`, open the Xcode project, and run it
on your iPhone.

## How it works

These models activate only about 3B of their parameters per token. Each layer
routes every token to 10 of 512 experts (80B) or 8 of 256 (35B). Swiftlet:

- keeps the dense weights resident: attention, DeltaNet projections, routers,
  shared experts, embeddings. About 1.3 GB (35B) or 2.5 GB (80B) at 4-bit;
- repacks the tens of thousands of routed experts into fixed-stride blobs in
  a `.qpack` container, so fetching one expert is exactly one `pread` from
  SSD, no mmap and no page-cache thrash;
- caches hot experts in a bounded pool with LFU plus recency eviction. Cache
  size barely affects speed (measured 43 to 70 percent hit rates at the same
  throughput), because Apple SSDs absorb the misses;
- runs the whole forward pass on Metal with runtime-compiled shaders, so no
  Metal toolchain is needed at build time and the same code ships on iOS.

75 percent of the layers use Gated DeltaNet linear attention with a
fixed-size recurrent state, so there is no growing KV cache for those layers
at any context length.

## Four ways to use it

Swiftlet is a library first:

1. **The Swift package.** Add SwiftletCore to any macOS or iOS app and use
   `SwiftletSession` for chat with streaming deltas, conversation caching,
   sampling with repetition control, and memory-pressure handling built in.
2. **The CLI.** `swiftlet chat` and `swiftlet generate` for local use and
   benchmarking, `swiftlet-repack` to build containers from MLX checkpoints
   (including streaming straight from Hugging Face with resume).
3. **The server.** `swiftlet-server` speaks the OpenAI chat-completions API
   on loopback, so any chat UI that talks to OpenAI-compatible endpoints can
   use a streamed local model.
4. **An app.** [Priv AI](https://apps.apple.com/us/app/priv-ai/id6765706001)
   on iOS embeds SwiftletCore as its streamed-model engine. End users tap
   Download and chat. Nothing here is terminal-only. The app itself is open
   source at [leonickson1/localLLM](https://github.com/leonickson1/localLLM)
   if you want to build it yourself (clone this repo next to it as
   `swiftlet`).

## Correctness

Every layer of the forward pass (Gated DeltaNet recurrence, gated GQA
attention, sparse MoE routing) is validated against mlx-lm reference
implementations with per-layer fixtures, in f32 and int4 quantized form.
Incremental decoding is verified against whole-sequence processing. Metal
kernels are tested against the exact CPU reference, and the fast and scalar
GPU kernels are verified to produce identical outputs. Containers are
byte-verifiable against their source checkpoints. Streaming placement never
changes model semantics: an expert answers identically from cache or disk.

```bash
swift test
```

The test suite uses [Swift Testing](https://developer.apple.com/documentation/testing),
which ships with the full Xcode toolchain (Xcode 16+), with Swift.org
toolchains, and with Command Line Tools 26 and later. On CLT 26.x the framework
is present but not on a default search path, and its interop dylib sits in a
separate directory, so `swift test` reports `no such module 'Testing'` until you
point the compiler and linker at both locations:

```bash
swift test \
  -Xswiftc -F -Xswiftc /Library/Developer/CommandLineTools/Library/Developer/Frameworks \
  -Xlinker -rpath -Xlinker /Library/Developer/CommandLineTools/Library/Developer/Frameworks \
  -Xlinker -rpath -Xlinker /Library/Developer/CommandLineTools/Library/Developer/usr/lib
```

Older Command Line Tools do not ship Swift Testing at all; there, install Xcode
or a Swift.org toolchain, or select an installed Xcode with
`sudo xcode-select -s /Applications/Xcode.app`. Thanks to @Avicennasis for
pinning down the CLT 26.x path on multiple machines (#12).

## Roadmap

- Batched prefill: long prompts currently process at decode speed, so agent
  clients with big system prompts are slow. Top of the list.
- Phone speed pass: fp16 activations and fewer GPU dispatches per token.
- A 6-bit container tier between the 4-bit and 8-bit ones.

## Relationship to TurboFieldfare

[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) proved the
expert-streaming thesis for Gemma on Macs, and Swiftlet adopts several of its
published design lessons with gratitude: stream experts with `pread` into a
bounded slot pool instead of mmap, evict with LFU plus recency, pack experts
at fixed stride so one fetch is one read, install by routing downloaded bytes
straight into their final container positions, and compile shaders at
runtime.

Everything else is built here, from scratch, in about 10k lines of Swift and
Metal written against mlx-lm references rather than TurboFieldfare code:

- support for a different model family with a fundamentally different
  architecture: the Qwen hybrid stack with Gated DeltaNet linear attention,
  gated GQA, and high-sparsity MoE with a shared expert (TurboFieldfare runs
  Gemma, a classical dense transformer);
- MLX affine int4/int8 group quantization compute in Metal, byte-addressed
  kernels with 64-bit offsets for multi-gigabyte shards, a cooperative
  simdgroup GEMV fast path, and explicit hazard management;
- a validated CPU reference implementation and the fixture infrastructure
  that gates every kernel change;
- the `.qpack` container and repacker, the resumable Hugging Face streaming
  installer with stall recovery, and download cancellation;
- the chat session layer: template handling for thinking and non-thinking
  Qwen variants, sampling with presence and frequency penalties and
  minimum-length and sentence-completion stopping, conversation caching with
  delta prefill, and iOS memory-pressure coordination;
- iPhone support end to end, including the app engine integration.

[colibrì](https://github.com/JustVugg/colibri) informed the caching and
placement policy thinking. mlx-lm is the correctness reference throughout.

Swiftlet was built with
[Claude Code](https://claude.com/claude-code).

## License

Apache 2.0. Model weights are downloaded separately and remain governed by
their own terms (Qwen models: Apache 2.0). See THIRD_PARTY_NOTICES.md.
