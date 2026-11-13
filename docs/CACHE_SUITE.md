# Does the expert cache help across a real session, and is the output still good?

`docs/CACHE_AB.md` measures the cache inside one answer: a fresh process, an
empty pool, one prompt. A real session is different. Prompts arrive back to
back in one process, and whatever the pool holds when one answer ends is
what the next one starts with. This experiment runs a fixed suite of
prompts that way and answers three questions:

1. **Efficiency.** Over the whole session, how much decode time does the
   cache save against a cache that is effectively off, with the page cache
   cold and warm? Per prompt, how much of the hit rate comes from what
   earlier prompts left behind?
2. **Headroom.** How close is the cache's policy to the best possible
   eviction (Belady's optimum) on the same routing, and what would other
   budgets have scored?
3. **Quality.** Is the output still right? The cache cannot change it by
   construction (see below), and the same suite is graded automatically and
   can be compared against a reference implementation.

## The suite

`scripts/suites/core.jsonl`: 18 prompts in a fixed, interleaved order, so
the session crosses domains the way a real one does instead of running all
the math and then all the code.

| Category | Prompts | Graded by |
|---|---|---|
| svg | pelican riding a bicycle (Simon Willison's prompt, verbatim), fox under a tree, rainfall bar chart | well-formed SVG; the chart must contain its six values; rendered side by side in the gallery |
| math | three GSM8K test questions, the bat and ball problem | final `Answer:` line against the known answer |
| code | `roman_to_int`, `merge_intervals` | the function is run against test cases |
| fact | gold's symbol, author of Pride and Prejudice, Apollo 11 year | the reply contains the answer |
| instruction | five red fruits as a lowercase numbered list; bare JSON for Tokyo | IFEval-style verifiable rules |
| summarization | three bullets from a 250-word passage (also the long prefill) | exactly three bullets, nothing else |
| writing | the `CACHE_AB.md` mixture-of-experts explanation, a 400-word story | not graded; read them, and see the repetition score |
| translation | one paragraph into French | not graded |

Each prompt has its own `max_new` (1024 for SVGs, 32 for facts), up to
7168 new tokens per pass. Add or change prompts by editing the JSONL; each
line is `{"id", "category", "prompt", "max_new", "check"}` (and optionally
`system`). `scripts/suites/tiny.jsonl` is a smoke suite for the fixture model.

Why this mix, and not one prompt repeated: the MoE cache literature finds
that a cache measured on one prompt type flatters it, because similar
prompts route to similar experts. Zhang (arXiv:2608.07911) goes further and
shows that probe sets built from one instruction template per category
produce identical opening tokens and inflate measured locality by 19 to 32
points. Every prompt here is written separately, the chat template is
applied, and the report prints that paper's reporting checklist, so the
numbers can be compared with published work.

## How a pass runs

`swiftlet generate <model> --gpu --prompts-file suite.jsonl --answers a.jsonl --trace-steps t.jsonl`
loads the model once and runs every prompt in order. Each prompt is its own
fresh conversation, rendered through the chat template with reasoning
disabled, exactly as the app renders a new chat. The decode state resets
between prompts; the expert cache does not. Decoding is greedy. The trace is
the same per-step format as `CACHE_AB.md`, with each record tagged by the
prompt it belongs to.

`scripts/cache_suite.sh` runs the design:

- Floor arm, `--cache-gb 0.03` (18 slots, effectively no cache), against
  the full arm, `--cache-gb 8` (the CLI default).
- Each arm runs once cold and once warm (straight after, nothing
  evicted). Cold means the model's files have just been dropped from the
  page cache by `scripts/evict_page_cache.py`: it maps each file and calls
  `msync(MS_INVALIDATE)`, which needs no sudo and leaves the rest of the
  system's cache alone. A layer file read at 16 GiB/s warm and 6.3 GiB/s
  straight after eviction, so the eviction is real. `--cold-by purge` uses
  `sudo purge` instead.
- `--reps N` repeats the whole thing (default 1).

Prefill is token-major in both arms, for the reason given in `CACHE_AB.md`.

### Emulating a machine that cannot hold the model: `--no-page-cache`

On a Mac whose RAM can hold much of the expert file, the OS page cache
already answers most misses from memory. Swiftlet's pool then only saves a
syscall and a copy per hit, so the measured gap is small. That is a true
result for that machine, but it says nothing about the case the cache
exists for: a phone, or a model several times larger than RAM.

An arm label ending in `nc` (for example `8nc`) runs that budget with
`--no-page-cache`. The layer files are opened with `F_NOCACHE`, so what a
miss reads is not kept, and the next miss on the same blob goes to the SSD
again. Each `nc` arm is compared against the smallest `nc` arm, so
`--arms "0.03 8 0.03nc 8nc"` gives two comparisons:

- `0.03` against `8`: Swiftlet's pool on top of the OS page cache. This is
  what a Mac user sees.
- `0.03nc` against `8nc`: Swiftlet's pool when nothing else caches. This
  emulates a device whose RAM cannot hold the expert file.

`F_NOCACHE` does not evict pages the OS already holds, so these arms are
only meaningful after an eviction (the default). A run with `--cold-by none` looks exactly like the
page cache arms. The miss tiers in the pass lines show whether the bypass
took: an `nc` pass should report almost all misses as device reads. Label
it as an emulation in any write-up. The real-world confirmation is a model
larger than RAM (the 8-bit 35B or the 80B) run with the ordinary arms.

### Eviction policy: `--cache-policy lru` (now the default)

The cache's default policy is LFU with history. It evicts the expert used
least often since launch, with recency breaking ties. On the first 35B
campaign the offline replay showed what that costs across a session:

- At 8 GiB the policy hit 77.8%, plain LRU 93.2%, and Belady's optimum 97.1%.
- Each prompt replayed alone from an empty cache hit 91.7%. The same
  prompts in a session hit only 77.8%, because the frequency history keeps
  earlier prompts' favourites resident.

LRU is now the default; `--cache-policy lfu` restores the old rule (the
history is still kept; only the victim choice changes). An arm label with
`lru` or `lfu` before any `nc` (`8lfu`, `8lfunc`) picks the policy
explicitly. Every log states the policy it ran with, and the report reads it
from there; a log without that line predates it and ran LFU. The report checks an LRU arm's
traced hit count against an exact slot-for-slot LRU replay.

## Before running it on a 24 GB Mac

- **Disk.** The 4-bit 35B container is 18 GB. Leave a few GB beyond that
  for swap.
- **Memory.** Reboot, or at least quit browsers, Xcode and simulators. Then
  check `sysctl vm.swapusage`: it should be close to zero. When this was
  set up, the machine already had 15.4 GB of swap in use. Under that load,
  an 8 GB pool would be paged out by the kernel, and the comparison would
  measure swap rather than the cache. The runner records swap before every
  pass, and the report prints it beside each pass so a contaminated run is
  visible.
- **Power.** Stay on mains power. Don't run anything else during the
  campaign.
- **Storage.** Keep the container on the internal disk.

## Running it

```
swift build -c release
.build/release/swiftlet-repack --from-hf Leonickson/Qwen3.6-35B-A3B-qpack --output ~/models/qwen3.6-35b.qpack
scripts/cache_suite.sh ~/models/qwen3.6-35b.qpack ~/cache-suite
```

At 7 to 11 tok/s the four passes (floor and full, cold and warm) take
about an hour. The floor arm is the slow one. For a first look, add
`--max-new-cap 256`, which takes about 45 minutes with all four arms. Other
options:

- `--arms "0.03 2 8"` adds a middle budget.
- `--shuffle SEED` changes the order, to test whether the result depends on it.
- `--suite FILE` runs a different prompt set.

The run leaves:

- `report.txt` and `cache-report.html`: the cache analysis.
- `quality.txt` and `quality/gallery.html`: the graded answers, with every
  SVG rendered.
- One trace, answers file, log and memory snapshot per pass.

Rerun the reports with:
`scripts/cache_suite_report.py ~/cache-suite --html ~/cache-suite/cache-report.html`.

## Reading the cache report

- **Passes.** Each pass's decode speed, hit rate, and miss tiers (page
  cache copies against device reads), plus swap.
- **Identity.** Every pass must produce the same tokens for every prompt.
  If any differ, the arms did not do the same work, and the comparison is
  void.
- **Per prompt.** Milliseconds per token with the cache off and on, the
  saving, and the hit rate. "Hit first 16" is the hit rate over a prompt's
  first 16 decode steps: what the previous prompts left in the pool.
  "Session decode" is the headline number: total decode time off against on.
- **Offline replay.** The recorded routing is run through the cache's own
  policy, LRU and Belady. The own-policy replay must reproduce the traced
  hit count exactly; the report says whether it does. The carry-over table
  replays each prompt alone from an empty cache, so the "gain" column is
  what keeping the cache between prompts is worth, prompt by prompt.
  Budgets that were not run are replayed too, which is free because the
  routing is fixed.
- **Reporting block.** Union-to-capacity ratio, bytes read per output
  token, decode steps recorded and so on, after the checklist in
  arXiv:2608.07911 section 10.

What would show the cache is helping:

- The session decode saving is positive in both page cache states.
- The full arm's device reads per token fall well below the floor arm's.
- The carry-over gain is positive.

The warm cells are the steady-state answer, the cold cells the first
session after launch. If the warm saving is small, report that. On a
machine whose page cache can hold most of the 16.9 GiB expert file, the
kernel already does most of the cache's job. `CACHE_AB.md` explains how to
read the four cells.

## Quality: three separate claims

**1. The cache does not change the output.** It changes where expert bytes
come from (pool, page cache or SSD), never what they are. Under greedy
decoding, the same prompt must therefore produce byte-identical tokens with
any budget. The report checks this across every pass, so each campaign
proves it again.

**2. Streaming matches the same weights held fully in memory.** The
container is byte-identical to `mlx-community/Qwen3.6-35B-A3B-4bit`. Run
that checkpoint through mlx-lm:

```
pip install mlx-lm
scripts/suite_reference.py mlx --model mlx-community/Qwen3.6-35B-A3B-4bit --out ref-mlx.jsonl
scripts/suite_score.py scripts/suites/core.jsonl ~/cache-suite/answers-warm-8-1.jsonl \
    --reference ref-mlx.jsonl --names swiftlet,mlx-4bit --out ~/cache-suite/vs-mlx
```

The score prints, per prompt, whether the two token streams are identical
or where they diverge. Small kernel differences in floating point can flip
a near-tie between two tokens after many steps, so a late divergence is
expected. An early one on many prompts would be a bug. mlx-lm needs about
20 GB for these weights, more than a 24 GB Mac can spare, so run this
step on a 32 GB+ machine and copy the JSONL back.

**3. What 4-bit costs against full precision.** Run the same suite against
the original model behind any OpenAI-compatible endpoint (vLLM on a lab
GPU, or a hosted provider):

```
scripts/suite_reference.py openai --base-url http://gpu-box:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' --out ref-bf16.jsonl
```

Compare with `suite_score.py` as above. This measures quantization, not
Swiftlet, and the write-up should keep the two apart.

The gallery puts every answer side by side and renders every SVG. That is
the page to show for the pelican.

## Sources

- Simon Willison, the pelican-on-a-bicycle test: https://simonwillison.net/tags/pelican-riding-a-bicycle/
  He has also cautioned that it no longer separates frontier models well. Here it is a visual check, not a ranking.
- Y. Zhang, "Reproducible Evaluation of MoE Expert Caching", arXiv:2608.07911 (2026): replay semantics, template contamination, reporting checklist.
- Eliseev and Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading", arXiv:2312.17238: LRU expert cache hit rates on OpenAssistant conversations.
- fMoE, arXiv:2502.05370: expert hit rates on LMSYS-Chat-1M and ShareGPT prompts.
- GSM8K (Cobbe et al. 2021, MIT licence), the source of the three math questions.
