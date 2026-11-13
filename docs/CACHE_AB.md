# Does the expert cache help, and where in an answer?

The expert cache (`--cache-gb`) keeps recently routed expert blobs in GPU
shared memory so the next token that routes to them skips the pread. The
earlier sweeps (`BENCH_QWEN36_35B_M1_MINI.md`, `BENCH_QWEN36_35B_M4MAX.md`)
showed decode speed moving modestly with the budget, and also showed the
catch: a miss is not one thing. On a Mac the file the cache misses into is
itself cached by the kernel, so a miss is either a pread the page cache
answers from RAM (a syscall plus a 1.7 MiB copy) or a pread that waits on
the SSD. Hit rate alone cannot tell those apart, and on a 16 GB machine
every GB handed to the pool was a GB taken from the page cache, so the two
were close to zero sum.

This experiment answers two questions directly, from one prompt:

1. Over one whole answer, how much time does the cache save against a cache
   that is effectively off, with the page cache cold and with it warm?
2. Where in the answer does the saving come from: do the first tokens pay
   while the pool fills, and when does it start paying back?

## What is measured

`swiftlet generate --gpu ... --trace-steps out.jsonl` writes one JSON line
per step. Each decode record has the step's wall time and its S3a/S3c split
(blocking wait, encode, CPU gap, GPU time, time inside the expert fetch
scope), plus the step's fetches folded by tier:

- `hits`: served from the pool, no syscall.
- `miss_page_cache`: pread issued, every page already resident in the page
  cache, so the cost is the copy.
- `miss_device`: pread issued, at least one page not resident, so the SSD
  was involved.
- `pread_s`, `pread_max_s`, `pread_ms_hist`: each missed blob's own pread
  wall time, summed, max, and a histogram with edges at 0.25, 0.5, 1, 2,
  4, 8 and 16 ms. A page cache copy of 1.7 MiB sits in the first buckets;
  a device read sits to the right.
- `layers`: for every layer, the experts the router asked for and one
  letter per expert (H hit, P page cache miss, D device miss, U unprobed).
  This is the routing itself, so the run can be replayed offline against
  another policy.

The page cache tier comes from a read-only `mmap` of each layer file plus
`mincore(2)` on the blob's pages the instant before its pread. Neither call
faults a page in, so the probe does not disturb what it measures. The
mapping is only created when tracing; an untraced run is byte for byte the
same code path as before.

## The design

Two arms, each run with the page cache in two states, three repeats:

- Floor arm, `--cache-gb 0.03`. That is 18 slots, the smallest budget the
  pool accepts (it refuses fewer than 16). Every expert comes through
  pread; the only thing standing between the model and the SSD is the
  page cache.
- Full arm, `--cache-gb 8`, the CLI default.
- Cold: `sudo purge` right before the run.
- Warm: the run immediately after the cold one, same arm, nothing purged.

Every arm prefills token-major (`--prefill-chunk 0`). The default
layer-major schedule asks one layer for the union of a 32 token chunk's
experts in a single batch, up to 256 blobs, and the floor arm's 18 slots
cannot hold that many at once. Token-major keeps every batch at top-k. It
costs TTFT, equally in both arms, and changes nothing about decode, which
is what the experiment measures.

Decoding in `generate` is always greedy (argmax), so every run produces the
identical token sequence and therefore the identical routing. The runner
checks that all answers are byte identical before it summarises anything.
If they are not, the arms did not see the same work and the comparison is
void.

Why floor against full rather than another sweep: the sweep already exists.
What it never had was an "off" arm, so it could only speak to the marginal
GB, not to whether the pool is worth having at all. Why 256 tokens rather
than 64: the LFU needs steps to build a working set, and the question is
when it starts paying.

## Running it

```
swift build -c release
scripts/cache_ab.sh /path/to/qwen3.6-35b.qpack ~/Downloads/cache-ab
```

Options: `--arms "0.03 2 8"`, `--reps 3`, `--max-new 256`, `--prompt "..."`,
`--chat`, `--prefill-chunk N`, `--no-purge` (no sudo; the "cold" cell is
then just the first run of each pair, with whatever the kernel held),
`--bin PATH`.

Twelve runs of 256 tokens. At 7 to 11 tok/s on an M-series laptop that is
about 10 minutes; at 2.5 tok/s on a base M1 mini it is about 25.

The report runs automatically at the end and can be rerun on the directory:

```
scripts/cache_ab_report.py ~/Downloads/cache-ab --window 16 --png ~/Downloads/cache-ab/plot.png
```

## Reading the report

The run lines give each run's decode speed, pool hit rate, the share of
each step spent inside the fetch path, and what fraction of misses were
page cache copies versus device reads.

The window table is the answer to "where". For each page cache state it
lays the floor arm and the full arm side by side in windows of 16 steps:
milliseconds per step for each, the difference, the full arm's hit rate and
device misses per step in that window, and the floor arm's fetch time and
device and page cache misses in the same window. Expect the first window or
two to show the pool filling (low hit rate, little or no saving) and the
saving to grow as the hit rate climbs. The last line gives the whole answer
total, the split between the first and second half, and the step from which
the cumulative saving stays positive.

What the four cells mean:

- Warm floor against warm full is the steady state value of the pool. The
  kernel already holds as much of the file as it can in both arms; the full
  arm only adds skipping a syscall and a copy per hit, 320 times per token
  at top-8 over 40 layers. If this gap is small, the pool is not buying much
  on top of the page cache on this machine.
- Cold floor against cold full is what the pool is worth for the first
  answer after launch.
- Cold full slower than cold floor would mean the pool is double buffering
  bytes the page cache already holds and starving it. The floor arm's
  `miss_device` column against the full arm's says so directly. The fix in
  that case is a no-cache hint on the pool's preads, not a bigger budget.

The per-rep pairing compares floor and arm inside the same rep, which is
the fair comparison when the machine drifts over a campaign. The per-step
breakdown splits each cell's step into blocking wait on the GPU, encode,
and the CPU gap, with the fetch scope inside the gap. This is the line to
read when hit rate and wall time disagree: if fetch fell and wait rose by
about the same amount, the pool paid for its RAM in GPU stalls, which on a
machine already swapping is exactly what happens. The pread histogram
under it is where the two kinds of miss separate: page cache copies sit
under 0.5 ms, device reads sit in the 1 to 4 ms band. The runner records
swap and page cache state before every run in `mem-*.txt`, and the report
prints the swap figure per run.

The offline replay at the end takes the recorded routing and runs it
through the cache's own policy (which must reproduce the traced hit count
exactly, or something is off), through LRU, and through Belady's optimal
with perfect knowledge of the future, all at the full arm's slot count. The
gap between the policy and Belady is the most any smarter eviction could
gain at that budget. The replay also reports how many distinct experts the
answer touched, which is the budget at which every repeat would be a hit,
and the policy's hit rate at the standard budgets that were not run, since
the routing is the same and a replay is free.

## Machine notes

The split between page cache and device misses depends on how much of the
16.9 GiB expert file fits in RAM beside the process. On 16 GB the floor arm
cannot hold it all and pays device reads throughout; on 24 GB the floor
arm's warm run can be served almost entirely from the page cache, which
makes warm floor the cleanest "the kernel does everything" baseline. The
full arm's 8 GB pool comes out of the same RAM. Both situations are
informative; the report shows which one you are in through the tier
columns rather than asking you to guess.

If the model lives on an external drive, device misses cost more than they
would on the internal SSD and the cold cells will flatter the pool. Keep
the container on the internal disk for numbers that mean something.
