# Third-party notices

Swiftlet builds on the work of several open-source projects.

## Design references

- **TurboFieldfare** (Apache 2.0), Andrey Mikhaylov,
  https://github.com/drumih/turbo-fieldfare: the `.qpack` container layout,
  single-pread expert blobs, streaming installer design, expert cache, and
  runtime-compiled Metal library pattern are modeled on TurboFieldfare's
  `.gturbo` format and runtime.
- **colibrì** (Apache 2.0), https://github.com/JustVugg/colibri: cache and
  placement policies (learned pinning, router-lookahead prefetch, batch-union
  prefill) and the correctness-first measurement discipline.

## Reference implementations

- **mlx-lm** (MIT), Apple Inc., https://github.com/ml-explore/mlx-lm: the
  `qwen3_next` model implementation is the correctness oracle for this
  project; the gated-delta Metal kernel is a port of mlx-lm's
  `gated_delta.py` kernel. Vendored reference copies live in `references/`.
- **llama.cpp** (MIT), https://github.com/ggml-org/llama.cpp: secondary
  reference for the Qwen3-Next graph.

## Dependencies

- **swift-transformers** (Apache 2.0), Hugging Face: tokenization and chat
  templates.
- **swift-nio** (Apache 2.0), Apple Inc.: the loopback HTTP server.

## Model weights

`fixtures/prompt-snapshot-tokenizer/` contains Qwen3.6 tokenizer metadata
from Alibaba's Qwen team (Apache 2.0): the chat template and tokenizer
configuration, and a reduced vocabulary/merge table for the header test.
Its README records the source and reductions. No model weights are included.

Model weights are not distributed with this project. The Qwen3-Next and
Qwen3.5/3.6 checkpoints are released by Alibaba's Qwen team under Apache 2.0;
quantized community conversions are downloaded from their respective Hugging
Face repositories.
