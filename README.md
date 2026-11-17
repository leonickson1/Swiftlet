# Swiftlet

<a href="https://trendshift.io/repositories/102405?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-102405" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/102405/daily?language=Swift" alt="leonickson1%2FSwiftlet | Trendshift" width="250" height="55"/></a>
<a href="https://news.ycombinator.com/item?id=49158333" target="_blank" rel="noopener noreferrer"><img src="https://hackergit.org/badge/leonickson1/Swiftlet.svg?style=card" alt="Hacker News: 312 points, 141 comments" width="250" height="55"/></a>

![Platforms](https://img.shields.io/badge/platforms-macOS%2014%2B%20%7C%20iOS%2017%2B-blue)
![Swift](https://img.shields.io/badge/Swift-6.2%2B-orange)
![Metal](https://img.shields.io/badge/GPU-Metal-8A2BE2)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
[![Release](https://img.shields.io/github/v/release/leonickson1/Swiftlet)](https://github.com/leonickson1/Swiftlet/releases)
[![35B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen3.6--35B--qpack-yellow)](https://huggingface.co/Leonickson/Qwen3.6-35B-A3B-qpack)
[![80B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen3--Next--80B--qpack-yellow)](https://huggingface.co/Leonickson/Qwen3-Next-80B-A3B-qpack)
[![Built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-D97757)](https://claude.com/claude-code)

**Run 35B and 80B Qwen models on ordinary Apple devices, including iPhones.**

Swiftlet is a Swift + Metal runtime for the Qwen3-Next and Qwen3.5/3.6 MoE
hybrid model family. It keeps only the small dense core of a model resident in
memory and streams the routed Mixture-of-Experts weights from storage on
demand. On a 24 GB M5 MacBook Air with v0.1.0:

| Model | Disk | Dense core (always in RAM) | Expert cache | Peak RAM | Prompt (prefill) | Decode |
|---|---|---|---|---|---|---|
| [Qwen3.6-35B-A3B, 4-bit](https://huggingface.co/Leonickson/Qwen3.6-35B-A3B-qpack) | 18 GB | 1.3 GB | 2 GB | 3.5 GiB | 70 tok/s | 16 tok/s |
| | | | 8 GB (default) | 9.5 GiB | 69 tok/s | 17 tok/s |
| [Qwen3-Next-80B-A3B, 4-bit](https://huggingface.co/Leonickson/Qwen3-Next-80B-A3B-qpack) | 42 GB | 1.3 GB | 2 GB | 3.6 GiB | 50 tok/s | 10.7 tok/s |
| | | | 8 GB (default) | 9.6 GiB | 48 tok/s | 10.5 tok/s |

The dense core (attention, DeltaNet, routers, shared experts, embeddings) is
loaded once; the routed experts stay on the SSD and pass through the expert
cache. Peak RAM is the dense core plus the cache plus working buffers.
A 512-token prompt, then 128 generated tokens, with the model files dropped from
macOS's page cache first; mean of two runs. On this Mac there is RAM to spare,
so the page cache absorbs most expert misses and the cache size barely changes
speed. Where RAM cannot hold the model file (a phone, or a Mac running other
work), an 8 GB expert cache makes decoding 2 to 2.8 times faster.

That is also where Swiftlet beats a general-purpose runtime. The 35B on the
same Air, with 8 or 12 GB of its memory locked away to act like a smaller
machine:

| Memory available | Prompt, Swiftlet | Prompt, llama.cpp | Decode, Swiftlet | Decode, llama.cpp |
|---|---|---|---|---|
| 24 GB | 70 tok/s | **118 tok/s** | 15.5 tok/s | **21.6 tok/s** |
| 16 GB | **68 tok/s** | 31 tok/s | **13.8 tok/s** | 3.8 tok/s |
| 12 GB | **62 tok/s** | 23 tok/s | **12.4 tok/s** | 3.1 tok/s |

With the full 24 GB, llama.cpp is faster. With 16 or 12 GB, about what a
phone has, llama.cpp has to keep rereading weights it cannot hold, while
Swiftlet decodes 3.6 to 4 times faster. llama.cpp b12359 with Q4_K_M weights,
file-backed (`-lm mmap --repack 0`) and fitted to the GPU automatically; warm
requests, mean of three runs. Swiftlet's outputs are token-for-token identical
with the expert cache on or off.

The [8-bit 35B](https://huggingface.co/Leonickson/Qwen3.6-35B-A3B-8bit-qpack)
(34 GB on disk) is the quality tier for Macs: on identical prompts it removes
the repetition artifacts the 4-bit build can show on longer writing tasks.
Before v0.1.0 it measured about 7.6 GB of RAM and 3.5 to 4 tok/s on an M5.

Community datapoints from before v0.1.0, by @Avicennasis (#18): a base M1
(8-core GPU, 16 GB) decodes the 4-bit 35B at about 2.45 tok/s and the 8-bit at
about 1.74 tok/s, and there `--cache-gb 2` matched `--cache-gb 8` on short
prompts while using several GB less RAM. An M4 Max (40-core GPU, 64 GB) decodes
the 4-bit 35B at about 19.5 tok/s, serves the 8-bit 80B (78.8 GiB of weights,
more than the machine's RAM) in 6.9 GiB at about 4.8 tok/s, and runs the 4-bit
397B (207.6 GiB) in 12.6 GiB at about 1.4 tok/s. Decode speed tracks active
parameters (about 3B for the 35B and 80B, about 17B for the 397B) far more than
container size.

The 35B also runs on an iPhone 17 in about 2.5 GB of RAM, at about 1 tok/s
(measured before v0.1.0). [ANEMLL](https://www.anemll.com/) showed a 397B MoE
streaming on an iPhone 17 Pro as a proof of concept in early 2026.

Status: working end to end. Both models generate correct, validated output.
Only about 3B parameters are active per token, so these models chat and write
like large models but recall facts like small ones.

## Quick start: try it on a Mac

Prebuilt binaries for Apple Silicon, no Xcode needed:

```bash
curl -L https://github.com/leonickson1/Swiftlet/releases/download/v0.1.0/swiftlet-v0.1.0-macos-arm64.tar.gz | tar xz
cd swiftlet-v0.1.0-macos-arm64

# Download the 35B container (18 GB) from the R2 mirror (fast and consistent;
# Hugging Face throttles anonymous downloads harder the longer they run):
./swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-qpack \
  --output ~/models/qwen3.6-35b.qpack

# Chat:
./swiftlet chat ~/models/qwen3.6-35b.qpack "What is the capital of Spain?"
```

Downloading with `curl` avoids the macOS warning for unnotarized binaries. If
you downloaded the archive in a browser, run `xattr -dr com.apple.quarantine .`
inside the folder first.

Or build from source (needs Swift 6.2, which comes with Xcode 26):

```bash
git clone https://github.com/leonickson1/Swiftlet.git && cd Swiftlet
swift build -c release

# The 35B, as above:
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-qpack \
  --output ~/models/qwen3.6-35b.qpack

# Or the 80B (42 GB on disk, about 3.6 GiB of RAM with a 2 GB cache):
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3-next-80b-qpack \
  --output ~/models/qwen3-next-80b.qpack

# Or the 8-bit 35B (best writing quality, 34 GB on disk):
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

# OpenAI-compatible server (loopback only):
.build/release/swiftlet-server --model ~/models/qwen3.6-35b.qpack --port 8080
```

The same command also repacks raw MLX checkpoints
(`--from-hf mlx-community/...` or `--source /path/to/checkpoint`).
[INSTALL.md](INSTALL.md) walks through every step, including chat clients.

Requirements: Apple Silicon, macOS 14+ or iOS 17+, and free SSD space for the
container (18 GB for the 35B, 34 GB for the 8-bit 35B, 42 GB for the 80B).

## How it works

These models activate only about 3B of their parameters per token. Each layer
routes every token to 10 of 512 experts (80B) or 8 of 256 (35B). Swiftlet:

- keeps the dense weights resident: attention, DeltaNet projections, routers,
  shared experts, embeddings. About 1.3 GB for either model at 4-bit;
- repacks the tens of thousands of routed experts into fixed-stride blobs in
  a `.qpack` container, so fetching one expert is exactly one `pread` from
  SSD, with no mmap;
- caches experts in a bounded pool with least-recently-used eviction. Expert
  routing is strongly local within a response, so an 8 GB cache serves 92
  percent (35B) and 83 percent (80B) of expert loads;
- reads a prompt layer by layer in chunks of up to 512 tokens with tiled
  kernels, so each layer's experts are read once per prompt, with output
  identical bit for bit to processing the prompt token by token;
- runs the whole forward pass on Metal with runtime-compiled shaders, so no
  Metal toolchain is needed at build time and the same code ships on iOS.

75 percent of the layers use Gated DeltaNet linear attention with a
fixed-size recurrent state, so there is no growing KV cache for those layers
at any context length.

## Four ways to use it

Swiftlet is a library first:

1. **The Swift package.** Add SwiftletCore to any macOS or iOS app and use
   `SwiftletSession` for chat with streaming deltas, conversation caching,
   sampling with repetition control, cooperative cancellation, and
   memory-pressure handling built in.
2. **The CLI.** `swiftlet chat` and `swiftlet generate` for local use and
   benchmarking, `swiftlet-repack` to build containers from MLX checkpoints
   (including streaming straight from Hugging Face with resume).
3. **The server.** `swiftlet-server` speaks the OpenAI chat-completions API
   on loopback, so any chat UI that talks to OpenAI-compatible endpoints can
   use a streamed local model. It honors string/array `stop`, reports `stop`
   versus `length`, and cancels queued or active generation when the client
   disconnects.
4. **An iPhone app.** The open-source app
   [leonickson1/localLLM](https://github.com/leonickson1/localLLM) embeds
   SwiftletCore.

## Correctness

Every layer of the forward pass (Gated DeltaNet recurrence, gated GQA
attention, sparse MoE routing) is validated against mlx-lm reference
implementations with per-layer fixtures, in f32 and int4 quantized form.
Incremental decoding is verified against whole-sequence processing. Metal
kernels are tested against the exact CPU reference, and the fast, scalar and
batched prefill GPU kernels are verified to produce identical outputs.
Containers are byte-verifiable against their source checkpoints. Streaming
placement never changes model semantics: an expert answers identically from
cache or disk.

```bash
swift test
```

The test suite uses [Swift Testing](https://developer.apple.com/documentation/testing),
which ships with the full Xcode toolchain (Xcode 26+), with Swift.org
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

- Decode right after a long prompt: reading the prompt in one chunk leaves the
  page cache holding mostly late layers' experts, which costs the 35B up to
  about 8 percent over a short answer.
- Stop cleanly after very short answers: a stray special token can follow a
  one-word answer.
- Phone speed pass: fp16 activations and fewer GPU dispatches per token.
- A 6-bit container tier between the 4-bit and 8-bit ones.

## Credits

[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) proved expert
streaming for Gemma on Macs, and Swiftlet adopts its published design lessons:
`pread` into a bounded slot pool instead of mmap, fixed-stride expert packing,
installing downloads straight into their final positions, and runtime-compiled
shaders. Everything specific to the Qwen hybrid architecture (the DeltaNet,
gated attention and MoE forward pass and its Metal kernels, the `.qpack`
container and repacker, the CPU reference, the chat session layer, and the
iPhone integration) is written here against mlx-lm references.
[colibrì](https://github.com/JustVugg/colibri) informed the caching ideas.
Swiftlet was built with [Claude Code](https://claude.com/claude-code).

## License

Apache 2.0. Model weights are downloaded separately and remain governed by
their own terms (Qwen models: Apache 2.0). See THIRD_PARTY_NOTICES.md.
