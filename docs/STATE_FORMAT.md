# Conversation-state snapshot format (v2)

A snapshot is the byte image of one `QwenCPUModel.DecodeState`: everything a
Qwen engine needs to continue a sequence exactly where it stopped — the
attention KV rows, each DeltaNet layer's conv history and delta recurrence,
and the position. Written by `PersistableInferenceModel.snapshot(of:)`, read
by `restoreState(from:)`, on either engine (CPU reference or Metal), in the
same process or a later one. Implementation:
`Sources/SwiftletCore/ConversationState.swift`; byte offsets pinned by
`ConversationStateTests.headerLayoutIsPinned`.

What a snapshot is **not**: it holds no pending logits, no sampler or
repetition-penalty history, no chat-template history, no token ids, and no
expert-cache contents. Continuing needs at least one new token (`step` on the
restored state). `SwiftletSession` is a layer above this and is not persisted
here: its conversation cache decides whether a turn can reuse the live state
from a record of what it has fed, so a session-level restore needs its own
envelope (this snapshot, the token ids it was built from, the generation
options) and must *replace* that record with the restored sequence's rather
than merely reset it — a reset record reads as "nothing cached" and the next
turn starts a fresh state, discarding the restored one.

## Container

All integers little-endian. Offsets in bytes.

```
Header (144 bytes)
  0    8   magic                  "SWLSTATE"
  8    4   version                u32  = 2
  12   4   headerLength           u32  = 144
  16   4   dtype                  u32  1 = float32 (the only value in v2)
  20   4   identityScheme         u32  how `identity` was derived (see below)
  24   32  identity               SHA-256 of the model, per `identityScheme`
  56   44  geometry               11 x u32, in this order:
             numHiddenLayers, fullAttentionInterval, numKeyValueHeads,
             headDim, linearNumValueHeads, linearNumKeyHeads,
             linearKeyHeadDim, linearValueHeadDim, linearConvKernelDim,
             hiddenSize, vocabSize
  100  8   position               u64  tokens the state has consumed
  108  4   sectionCount           u32
  112  8   totalLength            u64  whole file, header through trailer
  120  4   rotaryDims             u32  Int(headDim * partialRotaryFactor): the
                                       rotated prefix of every cached K row
  124  4   writerContextCapacity  u32  contextCapacity of the engine that
                                       wrote the file (informational)
  128  8   ropeTheta              f64  bit pattern of the RoPE base
  136  8   reserved               zero
Sections (sectionCount of, back to back, from offset 144)
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

K rows are cached **after** RoPE is applied, so a K section is only
meaningful under the rotary split and base that produced it; `rotaryDims`
and `ropeTheta` make that contract explicit in the header (today both are
also pinned transitively by the identity digest, which covers
`config.json`).

**Range rule.** Each of the eleven geometry fields, and `sectionCount`, is
between 1 and 2^24 (16 777 216) inclusive. The trailer digest proves only
that the bytes were written as read — anyone can compute it over their own
bytes — so a reader must treat every header field as hostile until it has
been compared with a model: it refuses an out-of-range field before doing
any arithmetic with it, and every section size it derives (`position ×
numKeyValueHeads × headDim`, the conv-tail and delta-state products) is
overflow-checked, so a crafted file is refused as `corrupt` rather than
trapping the process. Real geometries are far inside the rule: the tiny
test fixture's largest field is its vocabulary, 128; Qwen3.6-35B-A3B's is
its vocabulary, 248 320 (40 layers, 2 KV heads × 256, 32/16 DeltaNet heads
× 128, hidden 2048). `rotaryDims` and `writerContextCapacity` are only
compared, never multiplied, and need only be positive.

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

`restoreState(from:)` throws `ConversationStateError`, in this order, and
restores nothing on any failure:

| check | error |
|---|---|
| first 8 bytes are not the magic | `badMagic` |
| `version != 2` | `unsupportedVersion(found:supported:)` |
| fewer bytes than `totalLength` | `truncated` |
| more bytes than `totalLength`, bad `headerLength`, trailer digest mismatch | `corrupt(...)` |
| `dtype != 1` | `dtypeMismatch(found:)` |
| a geometry field or `sectionCount` outside 1…2^24; a non-positive `rotaryDims` or `writerContextCapacity`; a derived section size that does not fit 64 bits ("geometry overflow") | `corrupt(...)` |
| a section whose layer, kind, or size does not fit the header's own geometry and position; a duplicate; a missing layer at `position > 0` | `corrupt(...)` |
| identity scheme or digest differs from the model's | `modelMismatch(expected:found:)` |
| any geometry field, then `rotaryDims`, differs from the model's | `geometryMismatch(field:expected:found:)` |
| `ropeTheta` differs from the model's (compared bitwise) | `ropeThetaMismatch(expected:found:)` |
| `position` exceeds the restoring engine's `contextCapacity` | `capacityExceeded(position:capacity:)` |

The digest is checked before any header field beyond version and length is
interpreted, so everything after it is what the writer wrote. Geometry is
compared after identity: a foreign model reports `modelMismatch`, not the
first dimension that happens to differ. `writerContextCapacity` is never
compared: an engine with a larger capacity may legitimately restore. A
`position` equal to the capacity restores — the state is valid — and the next
token is then refused by admission exactly as it is today.

`snapshot(of:)` refuses, as `inconsistentState(...)`, a live state no reader
could restore, instead of writing a file the reader would call `corrupt`:

| live state | why |
|---|---|
| its last step threw after a layer had begun mutating it and before `position` advanced (`DecodeState.tainted`) | the CPU engine appends KV rows per layer and the Metal engine advances the GPU recurrence per layer and the KV mirror per attention layer, with `position` advanced only after the whole token: a cancelled or failed token leaves some layers one step ahead of the rest. The flag is never cleared; such a state keeps the existing contract that its caller discards it |
| another engine instance holds it bound (`boundOn`) | its conv history and delta recurrence live in that instance's GPU buffers; the CPU fields are a stale copy. Only the holding instance may write it, until a switch or its deinit releases the state |
| a field does not fit its position, or a layer is missing | the same shape rules the reader applies |

## Engines

- **CPU reference**: the state's fields are the snapshot's sections,
  directly.
- **Metal fast path**: the bound state's conv history and delta recurrence
  live in per-layer GPU buffers; `snapshot(of:)` reads them back into the
  state first. KV comes from the state's mirror, which the engine copies out
  of the GPU KV rows after every attention layer (`appendKVMirror`: exactly
  `count * kvRowFloats` floats per call, once per attention layer per token
  and once per chunk, so after a completed step every attention layer's
  mirror holds `position` rows).
  `ConversationStateTests.metalSnapshotMatchesDirectGPUReadback` proves both
  paths bitwise against direct GPU readback. A restored state is unbound; its
  first step loads it into the GPU buffers through `bindState`, which also
  validates every layer's shape against the model and refuses a tainted
  state (`QwenMetalModel.RuntimeError.inconsistentState`) before touching any
  buffer or the state that was bound. Stepping any other state on the same
  instance captures the bound state's recurrence first, so one instance
  serves any number of live states; when an instance is deallocated it hands
  its bound state the recurrence the same way.

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

`--save-state` writes the state as it stands after the last generated token
was fed (prompt plus every generated token). `--load-state` restores it,
prints the file's position and the capacity it was written at beside this
engine's, admits `--max-new` against the capacity *remaining after the
restored position*, and feeds `--prompt`/`--ids` as the continuation — raw
tokens, at least one; `--chat` is refused together with `--load-state`
rather than silently wrapping the continuation in a template turn.
`--max-new 0` saves right after the prompt. `generate` also prints the raw
generated ids to stderr so a resumed run can be compared token for token
with an uninterrupted one.

## Versioning

`version` bumps on any change to the header layout, section kinds, layouts,
or the trailer. A v2 reader refuses anything else. Adding a section kind or a
dtype is a version bump, not a flag; the reserved bytes stay zero in v2.

Version 1 was a 128-byte-header draft (no `rotaryDims`,
`writerContextCapacity`, `ropeTheta`) that was validated but never released.
Widening a header is a layout change under the rule above, and the reader
would otherwise have reported a v1 file as `corrupt("header length 128")`
rather than by its version, so the released format is v2 with no v1 reader:
a v1 file reports `unsupportedVersion(found: 1, supported: 2)`.

## Validation log

2026-10-16, macbook3 (Apple M1 Max, 64 GB, macOS 27.0.1, Command Line
Tools, Swift 6.3.3), fixture gate on a fresh tree of this branch at
`9a59219` — the commit adding this file changes documentation only —
`swift test` with the README's `Testing.framework` flags:

- `ConversationStateTests`: 23 tests in 1 suite passed;
  `MetalStateSwitchTests`: 4 tests in 1 suite passed; full suite 228 tests
  in 35 suites passed (`main` `de1a942`: 201 in 33), no skips;
  `swift build -c release` 30 warnings on both trees.
- The interleaved-states requirement, run as a probe over `main`'s sources:
  10 issues on Metal (KV layers 3 and 7 of both sequences diverged; logits
  diverged from the second step of each sequence on), none on CPU; 0 issues
  on this branch.
- CLI, release binary, `fixtures/tiny-model-q4`, `--ids 1,5,9,42,7`,
  greedy, on each engine: A = `--max-new 8` → `121,107,41,67,53,37,107,82`;
  B = `--max-new 4 --save-state` → `121,107,41,67` = A[1..4], 15 792 bytes
  at position 9, "written at capacity 512"; C = a fresh process,
  `--load-state --ids 53 --max-new 3` → `37,107,82` = A[6..8]. The CPU
  engine resuming the Metal snapshot produced the same three ids (parity
  band, not a bitwise guarantee). Refused through the CLI: `modelMismatch`
  (loading into `fixtures/tiny-model`), `truncated` (a 300-byte prefix),
  `--chat` together with `--load-state`, an empty continuation.

**Real-model resume check: pending.** The resume check on the production
container (A = N tokens uninterrupted; B = fewer tokens then `--save-state`;
C = a fresh process with `--load-state` fed A's next id; B == A's prefix and
C == A's remainder, id for id, plus `modelMismatch` / `truncated` through the
CLI) has not yet been run on this build. The 2026-09-02 run recorded against
the v1 draft is deliberately not carried here; this section is to be replaced
by a run of this build before merge.
