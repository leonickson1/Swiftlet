# Conversation-state snapshot format (v1)

A snapshot is the byte image of one `QwenInferenceContext`: everything a Qwen
engine needs to continue a sequence exactly where it stopped — the attention
KV rows, each DeltaNet layer's conv history and delta recurrence, and the
position. Written by `PersistableInferenceModel.snapshot(of:)`, read by
`restoreContext(from:)`, on either engine (CPU reference or Metal), in the
same process or a later one. Implementation: `Sources/SwiftletCore/ConversationState.swift`;
byte offsets pinned by `ConversationStateTests.headerLayoutIsPinned`.

What a snapshot is **not**: it holds no pending logits, no sampler or
repetition-penalty history, no chat-template history, and no expert-cache
contents. Continuing needs at least one new token (`step` on the restored
context). `SwiftletSession`'s message-level continuation matching is a layer
above this and is not persisted here.

## Container

All integers little-endian. Offsets in bytes.

```
Header (128 bytes)
  0    8   magic          "SWLSTATE"
  8    4   version        u32  = 1
  12   4   headerLength   u32  = 128
  16   4   dtype          u32  1 = float32 (the only value in v1)
  20   4   identityScheme u32  how `identity` was derived (see below)
  24   32  identity       SHA-256 of the model, per `identityScheme`
  56   44  geometry       11 x u32, in this order:
             numHiddenLayers, fullAttentionInterval, numKeyValueHeads,
             headDim, linearNumValueHeads, linearNumKeyHeads,
             linearKeyHeadDim, linearValueHeadDim, linearConvKernelDim,
             hiddenSize, vocabSize
  100  8   position       u64  tokens the context has consumed
  108  4   sectionCount   u32
  112  8   totalLength    u64  whole file, header through trailer
  120  8   reserved       zero
Sections (sectionCount of, back to back, from offset 128)
  0    4   layer          u32  0-based decoder layer
  4    4   kind           u32  1 = K rows, 2 = V rows, 3 = conv tail, 4 = delta state
  8    8   count          u64  float32 elements that follow
  16   4*count  float32 data
Trailer (32 bytes)
  SHA-256 over every byte before it (header + sections)
```

Section layouts are the ones both engines hold in memory (the Metal fast
path's GPU buffers use the same ones, so a Metal snapshot is a memcpy of the
GPU state and a CPU restore of it needs no transposition):

| kind | shape | expected `count` |
|---|---|---|
| 1 K rows | `[position][numKeyValueHeads][headDim]` | `position * numKeyValueHeads * headDim` |
| 2 V rows | same as K | same |
| 3 conv tail | `[linearConvKernelDim - 1][convDim]`, `convDim = 2 * linearNumKeyHeads * linearKeyHeadDim + linearNumValueHeads * linearValueHeadDim` | `(linearConvKernelDim - 1) * convDim` |
| 4 delta state | `[linearNumValueHeads][linearValueHeadDim][linearKeyHeadDim]` | product of the three |

Layer `i` is a DeltaNet layer unless `(i + 1) % fullAttentionInterval == 0`.
DeltaNet layers carry kinds 3 and 4; attention layers carry kinds 1 and 2.
Each `(layer, kind)` appears at most once. Once `position > 0` every layer's
sections must be present; at `position == 0` the section table is empty.

## Model identity

Snapshots must never be restored into a different model. The identity is a
SHA-256 that reads no weight bytes, so it is cheap on an 18 GB container, and
is as strong as the container allows:

- **scheme 1 — qpack container with `hashes.json`**: digest over `config.json`
  and `hashes.json`. `hashes.json` lists a SHA-256 per file in the container,
  so this transitively pins every weight byte.
- **scheme 2 — plain checkpoint, or a container without `hashes.json`**:
  digest over `config.json`, `manifest.json` if present, and for each
  safetensors shard (sorted by name) its file name, header JSON, and byte
  size. This pins the architecture, tensor names, dtypes, shapes, and sizes,
  **not the weight values**: two checkpoints that differ only in weight bytes
  but share their headers would share an identity under scheme 2.

Each label and length is fed to the hash before its bytes, so a shard rename
or a boundary shift changes the digest.

## Refusals

`restoreContext(from:)` throws `ConversationStateError`, in this order, and
restores nothing on any failure:

| check | error |
|---|---|
| first 8 bytes are not the magic | `badMagic` |
| `version != 1` | `unsupportedVersion(found:supported:)` |
| fewer bytes than `totalLength` | `truncated` |
| more bytes than `totalLength`, bad `headerLength`, trailer digest mismatch | `corrupt(...)` |
| `dtype != 1` | `dtypeMismatch(found:)` |
| a section whose layer, kind, or size does not fit the header's own geometry and position; a duplicate; a missing layer at `position > 0` | `corrupt(...)` |
| identity scheme or digest differs from the model's | `modelMismatch(expected:found:)` |
| any geometry field differs from the model's | `geometryMismatch(field:expected:found:)` |

The digest is checked before any header field beyond version and length is
interpreted, so everything after it is what the writer wrote. Geometry is
compared after identity: a foreign model reports `modelMismatch`, not the
first dimension that happens to differ.

## Engines

- **CPU reference**: the context's fields are the snapshot's sections,
  directly.
- **Metal fast path**: the bound context's conv history and delta recurrence
  live in per-layer GPU buffers; `snapshot(of:)` reads them back into the
  context first. KV comes from the context's mirror, which the engine copies
  out of the GPU KV rows after every attention layer.
  `ConversationStateTests.metalSnapshotMatchesDirectGPUReadback` proves both
  paths bitwise against direct GPU readback. A restored context is unbound;
  its first step loads it into the GPU buffers through `bindContext`, which
  also validates every layer's shape against the model.

Correctness bar (tests in `ConversationStateTests`): save after N greedy
tokens → restore into a fresh model instance → continue → logits and greedy
tokens bitwise identical to the uninterrupted run, on CPU and on Metal,
including across a layer-major chunked prefill. A Metal snapshot restored
into the CPU reference continues within the engines' usual parity band
(2e-3), not bitwise, because the state itself carries GPU-vs-CPU
accumulation noise.

## CLI

```
swiftlet generate <model> --gpu --prompt "..." --max-new 32 --save-state /tmp/s.swlstate
swiftlet generate <model> --gpu --ids 1234,567 --max-new 32 --load-state /tmp/s.swlstate
```

`--save-state` writes the context as it stands after the last generated token
was fed (prompt plus every generated token). `--load-state` restores it and
feeds `--prompt`/`--ids` as the continuation — raw tokens, no chat template
is applied to a resumed sequence, and at least one token is required.
`--max-new 0` saves right after the prompt.

## Versioning

`version` bumps on any change to the header layout, section kinds, layouts,
or the trailer. A v1 reader refuses anything else. Adding a section kind or a
dtype is a version bump, not a flag; the reserved bytes stay zero in v1.

## Validation log

2026-09-02, macmini2 (Apple M1, 8-core GPU, 16 GB, macOS 26.6.2, Swift
6.3.3), `swift build -c release`, Qwen3.6-35B-A3B 4-bit qpack, `--gpu
--cache-gb 4`, greedy, prompt "The fieldfare is a bird that" (7 tokens).
Log: `~/bench-logs/s6-state-check-20260902-105315.log` on macmini2.

| run | command | generated ids |
|---|---|---|
| A | `--max-new 32` | `369,1669,303,4357,11,13229,11,321,4634,4994,13,1049,369,264,4314,314,279,8541,1080,2902,11,321,424,369,14706,5265,310,279,3567,95180,13,561` |
| B | `--max-new 16 --save-state s` | `369,...,4314,314` = A[1..16]; wrote 66 807 200 bytes at position 23, sha256 `e85f4c11…eae26e4` |
| C | fresh process, `--load-state s --ids 279 --max-new 15` (279 = A[17]) | `8541,1080,2902,11,321,424,369,14706,5265,310,279,3567,95180,13,561` = A[18..32] |

Snapshot identity scheme 1 (the container ships `hashes.json`). Decode
2.9–3.3 tok/s in every run; the resumed run's first step processed one token
(41 command buffers, position 23 → 24). Tiny-fixture equivalents, including
the CPU engine resuming a Metal snapshot and the `modelMismatch` /
`truncated` refusals through the CLI, ran the same day.
