#!/usr/bin/env python3
"""Report for the traces scripts/cache_ab.sh leaves behind. See docs/CACHE_AB.md.

usage: cache_ab_report.py <out-dir> [--window 16] [--png plot.png] [--no-oracle]

Reads every trace-<state>-<arm>-<rep>.jsonl in the directory, folds decode
steps per (state, arm) across reps, and prints:

  1. one line per run: tokens, decode speed, hit rate, where the misses
     were served from, and how much of the step the fetch path took;
  2. for each page cache state, a window table that puts the floor arm and
     each larger arm side by side over the course of the answer, so you can
     see where in the answer the cache starts paying and how much;
  3. the cumulative saving over the answer and the break-even step;
  4. unless --no-oracle, an offline replay of the recorded routing against
     the cache's own policy (as a check), LRU, and Belady's optimal, at the
     larger arm's slot count, plus the size of the answer's working set.

Only the standard library is required; --png uses matplotlib if present.
"""
import glob
import heapq
import json
import os
import re
import statistics
import sys
from collections import defaultdict

NAME = re.compile(r"trace-(cold|warm)-([0-9.]+)-(\d+)\.jsonl$")


def load(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    kinds = defaultdict(list)
    for r in recs:
        kinds[r["kind"]].append(r)
    return {
        "run": kinds["run"][0] if kinds["run"] else {},
        "end": kinds["end"][0] if kinds["end"] else None,
        "prefill": kinds["prefill"][0] if kinds["prefill"] else None,
        "decode": kinds["decode"],
        "path": path,
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def per_step_series(runs, field):
    """Mean across reps of a per-step field, truncated to the shortest run."""
    n = min(len(r["decode"]) for r in runs)
    return [mean(r["decode"][i][field] for r in runs) for i in range(n)]


def swap_note(t):
    """'swap 16.5 GB' from the mem-<tag>.txt the runner writes beside each trace."""
    mem = t["path"].replace("trace-", "mem-").replace(".jsonl", ".txt")
    try:
        for line in open(mem):
            if "used =" in line:
                used = line.split("used =")[1].split()[0]
                return f"  swap {used}"
    except OSError:
        pass
    return ""


def run_line(state, arm, rep, t):
    d = t["decode"]
    if not d:
        return f"  {state:4} {arm:>6} #{rep}  no decode steps"
    wall = sum(s["wall_s"] for s in d)
    fetch = sum(s["fetch_s"] for s in d)
    hits = sum(s["hits"] for s in d)
    misses = sum(s["misses"] for s in d)
    page = sum(s["miss_page_cache"] for s in d)
    dev = sum(s["miss_device"] for s in d)
    unp = sum(s["miss_unprobed"] for s in d)
    decode_s = t["end"]["decode_s"] if t["end"] else wall
    pre = t["prefill"]["wall_s"] if t["prefill"] else 0.0
    tiers = (f"misses {pct(page, misses):3.0f}% page cache / {pct(dev, misses):3.0f}% device"
             + (f" / {pct(unp, misses):3.0f}% unprobed" if unp else ""))
    return (f"  {state:4} {arm:>6} #{rep}  {len(d):4d} tok  {decode_s:7.1f} s  "
            f"{len(d) / decode_s if decode_s else 0:5.2f} tok/s  prefill {pre:5.1f} s  "
            f"hit {pct(hits, hits + misses):5.1f}%  fetch {pct(fetch, wall):4.1f}% of step  {tiers}{swap_note(t)}")


def window_table(state, floor_arm, arm, floor_runs, arm_runs, window):
    f_wall = per_step_series(floor_runs, "wall_s")
    a_wall = per_step_series(arm_runs, "wall_s")
    n = min(len(f_wall), len(a_wall))
    if n == 0:
        return []
    a_hits = per_step_series(arm_runs, "hits")
    a_miss = per_step_series(arm_runs, "misses")
    a_fetch = per_step_series(arm_runs, "fetch_s")
    a_dev = per_step_series(arm_runs, "miss_device")
    f_fetch = per_step_series(floor_runs, "fetch_s")
    f_dev = per_step_series(floor_runs, "miss_device")
    f_page = per_step_series(floor_runs, "miss_page_cache")
    lines = [
        f"\n{state} page cache: floor {floor_arm} GiB vs {arm} GiB, {n} steps, windows of {window}",
        f"  {'steps':>9}  {'floor ms':>8}  {'arm ms':>7}  {'saved ms':>8}  {'saved':>6}  "
        f"{'arm hit':>7}  {'arm fetch':>9}  {'arm dev':>7}  {'floor fetch':>11}  {'floor dev':>9}  {'floor page':>10}",
    ]
    for w0 in range(0, n, window):
        w1 = min(n, w0 + window)
        fm = mean(f_wall[w0:w1]) * 1000
        am = mean(a_wall[w0:w1]) * 1000
        lines.append(
            f"  {w0 + 1:4d}-{w1:4d}  {fm:8.1f}  {am:7.1f}  {fm - am:8.1f}  {pct(fm - am, fm):5.1f}%  "
            f"{pct(sum(a_hits[w0:w1]), sum(a_hits[w0:w1]) + sum(a_miss[w0:w1])):6.1f}%  "
            f"{mean(a_fetch[w0:w1]) * 1000:7.1f}ms  {mean(a_dev[w0:w1]):7.1f}  "
            f"{mean(f_fetch[w0:w1]) * 1000:9.1f}ms  {mean(f_dev[w0:w1]):9.1f}  {mean(f_page[w0:w1]):10.1f}")
    total_f = sum(f_wall[:n])
    total_a = sum(a_wall[:n])
    saved = total_f - total_a
    cum = 0.0
    break_even = None
    for i in range(n):
        cum += f_wall[i] - a_wall[i]
        if cum > 0 and break_even is None:
            break_even = i + 1
        if cum <= 0:
            break_even = None
    half = n // 2
    first = sum(f_wall[:half]) - sum(a_wall[:half])
    second = saved - first
    lines.append(
        f"  whole answer: floor {total_f:.1f} s, arm {total_a:.1f} s, saved {saved:.1f} s ({pct(saved, total_f):.1f}%); "
        f"first half saved {first:.1f} s, second half {second:.1f} s"
        + (f"; cumulative saving positive for good from step {break_even}" if break_even else "; never ahead for good"))
    return lines


def pairing_lines(cells, floor, arms):
    """Floor vs arm inside the same rep and state: the fairest pairing when the
    machine drifts over the campaign (each rep runs its arms back to back)."""
    lines = ["\nper-rep pairing (same rep, same page cache state, decode seconds)"]
    for state in ("cold", "warm"):
        for arm in arms[1:]:
            f = {t["rep"]: t for t in cells.get((state, floor), [])}
            a = {t["rep"]: t for t in cells.get((state, arm), [])}
            for rep in sorted(set(f) & set(a)):
                fs = f[rep]["end"]["decode_s"] if f[rep]["end"] else sum(x["wall_s"] for x in f[rep]["decode"])
                as_ = a[rep]["end"]["decode_s"] if a[rep]["end"] else sum(x["wall_s"] for x in a[rep]["decode"])
                lines.append(f"  {state:4} rep {rep}: floor {fs:5.1f} s, {arm} GiB {as_:5.1f} s, "
                             f"arm {'faster' if as_ < fs else 'slower'} by {abs(pct(fs - as_, fs)):4.1f}%")
    return lines


def breakdown_lines(cells):
    """Where a decode step's wall time goes, per arm and state, averaged over
    reps: blocking wait on the GPU, CPU encode, the CPU gap, and inside the gap
    the expert fetch scope. Plus the arm's pread latency histogram, which is
    where page cache copies (left buckets) and device reads (right) separate."""
    edges = [0.25, 0.5, 1, 2, 4, 8, 16]
    labels = [f"<{e:g}" for e in edges] + [f">={edges[-1]:g}"]
    lines = ["\nper-step breakdown, ms, mean over decode steps and reps (wall = wait + encode + gap; fetch is inside gap)",
             f"  {'cell':>14}  {'wall':>6}  {'wait':>6}  {'encode':>6}  {'gap':>6}  {'fetch':>6}  {'other gap':>9}  {'gpu':>6}  "
             f"{'misses/step':>11}  {'pread ms/miss':>13}"]
    hist_lines = ["\npread latency histogram, all decode misses of the cell (ms buckets): " + "  ".join(labels)]
    for (state, arm), runs in sorted(cells.items(), key=lambda kv: (kv[0][0] != "cold", float(kv[0][1]))):
        steps = [st for t in runs for st in t["decode"]]
        if not steps:
            continue
        n = len(steps)
        wall = mean(st["wall_s"] for st in steps) * 1000
        wait = mean(st["wait_s"] for st in steps) * 1000
        enc = mean(st["encode_s"] for st in steps) * 1000
        gap = mean(st["gap_s"] for st in steps) * 1000
        fetch = mean(st["fetch_s"] for st in steps) * 1000
        gpu = mean(st["gpu_s"] for st in steps) * 1000
        misses = sum(st["misses"] for st in steps)
        pread = sum(st["pread_s"] for st in steps)
        lines.append(f"  {state + ' ' + arm + ' GiB':>14}  {wall:6.1f}  {wait:6.1f}  {enc:6.1f}  {gap:6.1f}  {fetch:6.1f}  "
                     f"{gap - fetch:9.1f}  {gpu:6.1f}  {misses / n:11.1f}  {pread * 1000 / misses if misses else 0:13.2f}")
        hist = [0] * 8
        for st in steps:
            for i, c in enumerate(st.get("pread_ms_hist", [])):
                hist[i] += c
        total = sum(hist) or 1
        hist_lines.append(f"  {state + ' ' + arm + ' GiB':>14}  " + "  ".join(f"{100 * h / total:5.1f}%" for h in hist))
    return lines + hist_lines


# ---- offline replay of the routing -------------------------------------

def access_stream(trace):
    """[(layer, [experts])] in call order, prefill first, then decode."""
    calls = []
    marks = []  # index of the first decode call
    if trace["prefill"]:
        for L in trace["prefill"]["layers"]:
            calls.append((L["l"], L["e"]))
    marks.append(len(calls))
    for s in trace["decode"]:
        for L in s["layers"]:
            calls.append((L["l"], L["e"]))
    return calls, marks[0]


def replay_lfu_recency(calls, slots):
    """The cache's own policy: LFU with history, recency tie-break, lowest slot
    on full ties, batch members protected, grow on demand, free slots first."""
    slot_key = []
    key_slot = {}
    freq = defaultdict(int)
    last_use = {}
    tick = 0
    hits = []
    for layer, experts in calls:
        tick += 1
        protected = set()
        h = 0
        for e in experts:
            k = (layer, e)
            freq[k] += 1
            if k in key_slot:
                s = key_slot[k]
                last_use[s] = tick
                protected.add(s)
                h += 1
                continue
            if len(slot_key) < slots:
                slot_key.append(None)
                s = len(slot_key) - 1
            else:
                free = [i for i, kk in enumerate(slot_key) if kk is None]
                if free:
                    s = free[0]
                else:
                    s = min((i for i in range(len(slot_key)) if i not in protected),
                            key=lambda i: (freq[slot_key[i]], last_use.get(i, 0), i))
            if slot_key[s] is not None:
                del key_slot[slot_key[s]]
            slot_key[s] = k
            key_slot[k] = s
            last_use[s] = tick
            protected.add(s)
        hits.append(h)
    return hits


def replay_lru(calls, slots):
    tick = 0
    last = {}
    hits = []
    for layer, experts in calls:
        tick += 1
        protected = set()
        h = 0
        for e in experts:
            k = (layer, e)
            if k in last:
                h += 1
            elif len(last) >= slots:
                victim = min((kk for kk in last if kk not in protected), key=lambda kk: last[kk])
                del last[victim]
            last[k] = tick
            protected.add(k)
        hits.append(h)
    return hits


def replay_belady(calls, slots):
    """Evict the resident key whose next use is farthest away (never used again
    counts as infinitely far). Batch members are protected like the real cache."""
    flat = [(ci, (layer, e)) for ci, (layer, experts) in enumerate(calls) for e in experts]
    INF = float("inf")
    next_use = [INF] * len(flat)
    seen = {}
    for i in range(len(flat) - 1, -1, -1):
        k = flat[i][1]
        next_use[i] = seen.get(k, INF)
        seen[k] = i
    resident = {}  # key -> its recorded next use (stale heap entries differ)
    heap = []      # (-next_use, key)
    hits = [0] * len(calls)
    i = 0
    for ci, (layer, experts) in enumerate(calls):
        protected = set()
        for e in experts:
            k = (layer, e)
            if k in resident:
                hits[ci] += 1
            elif len(resident) >= slots:
                parked = []
                while True:
                    nu, victim = heapq.heappop(heap)
                    if victim not in resident or resident[victim] != -nu:
                        continue
                    if victim in protected:
                        parked.append((nu, victim))
                        continue
                    del resident[victim]
                    break
                for p in parked:
                    heapq.heappush(heap, p)
            resident[k] = next_use[i]
            heapq.heappush(heap, (-next_use[i], k))
            protected.add(k)
            i += 1
    return hits


def oracle_section(traces_by_cell, arms):
    # Routing is identical across arms and reps (greedy); take any run with the
    # longest decode as the stream, and check the replay against the arm's own trace.
    any_runs = [t for runs in traces_by_cell.values() for t in runs]
    if not any_runs:
        return []
    src = max(any_runs, key=lambda t: len(t["decode"]))
    calls, decode_start = access_stream(src)
    if not calls:
        return []
    decode_calls = calls[decode_start:]
    decode_fetches = sum(len(e) for _, e in decode_calls)
    unique = len({(l, e) for l, es in decode_calls for e in es})
    stride = src["run"].get("stride_bytes", 0)
    lines = ["\noffline replay of the recorded routing (decode steps only)",
             f"  {len(src['decode'])} steps, {decode_fetches} expert fetches, "
             f"{unique} distinct (layer, expert) pairs touched"
             + (f" = {unique * stride / 2**30:.2f} GiB if every one were resident" if stride else "")]
    for arm in arms:
        runs = [t for (state, a), ts in traces_by_cell.items() if a == arm for t in ts]
        if not runs:
            continue
        slots = runs[0]["run"].get("slots", 0)
        if not slots:
            continue
        traced = sum(s["hits"] for s in runs[0]["decode"])
        own = replay_lfu_recency(calls, slots)
        lru = replay_lru(calls, slots)
        bel = replay_belady(calls, slots)
        own_d = sum(own[decode_start:])
        lines.append(
            f"  {arm} GiB = {slots} slots: traced hit rate {pct(traced, decode_fetches):5.1f}%, "
            f"replayed policy {pct(own_d, decode_fetches):5.1f}% "
            f"({'matches' if own_d == traced else 'DIFFERS from'} the trace), "
            f"LRU {pct(sum(lru[decode_start:]), decode_fetches):5.1f}%, "
            f"Belady optimal {pct(sum(bel[decode_start:]), decode_fetches):5.1f}%")
    # Budgets that were not run: the routing is the same, so their hit rates
    # are a replay away. Cheap guidance for picking the next campaign's arms.
    if stride:
        ran = {float(a) for a in arms}
        extra = [gb for gb in (1, 2, 4, 6, 8, 12) if gb not in ran]
        cells_ = []
        for gb in extra:
            slots = int(gb * 2**30 // stride)
            own = replay_lfu_recency(calls, slots)
            cells_.append(f"{gb} GiB ({slots} slots) {pct(sum(own[decode_start:]), decode_fetches):.1f}%")
        if cells_:
            lines.append("  policy hit rate at budgets not run, replayed from the same routing: " + ", ".join(cells_))
    return lines


def maybe_png(path, cells, window):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"\n(no plot: matplotlib not importable: {exc})")
        return
    states = sorted({s for s, _ in cells})
    fig, axes = plt.subplots(len(states), 2, figsize=(13, 4.2 * len(states)), squeeze=False)
    for row, state in enumerate(states):
        ax_t, ax_h = axes[row]
        for (s, arm), runs in sorted(cells.items(), key=lambda kv: float(kv[0][1])):
            if s != state:
                continue
            wall = [w * 1000 for w in per_step_series(runs, "wall_s")]
            hits = per_step_series(runs, "hits")
            miss = per_step_series(runs, "misses")
            dev = per_step_series(runs, "miss_device")
            xs = list(range(1, len(wall) + 1))
            sm = [mean(wall[max(0, i - window // 2):i + window // 2 + 1]) for i in range(len(wall))]
            ax_t.plot(xs, sm, label=f"{arm} GiB")
            hr = [pct(h, h + m) for h, m in zip(hits, miss)]
            ax_h.plot(xs, [mean(hr[max(0, i - window // 2):i + window // 2 + 1]) for i in range(len(hr))],
                      label=f"{arm} GiB hit rate")
            ax_h.plot(xs, [mean(dev[max(0, i - window // 2):i + window // 2 + 1]) for i in range(len(dev))],
                      linestyle=":", label=f"{arm} GiB device misses / step")
        ax_t.set_title(f"{state} page cache: ms per decode step (smoothed over {window})")
        ax_t.set_xlabel("decode step")
        ax_t.set_ylabel("ms")
        ax_t.legend()
        ax_h.set_title(f"{state} page cache: pool hit rate (%) and device misses per step")
        ax_h.set_xlabel("decode step")
        ax_h.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nplot written to {path}")


def main(argv):
    if len(argv) < 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    out = argv[1]
    window = 16
    png = None
    oracle = True
    i = 2
    while i < len(argv):
        if argv[i] == "--window":
            window = int(argv[i + 1]); i += 2
        elif argv[i] == "--png":
            png = argv[i + 1]; i += 2
        elif argv[i] == "--no-oracle":
            oracle = False; i += 1
        else:
            print(f"unknown option {argv[i]}"); return 2

    cells = defaultdict(list)
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "trace-*.jsonl"))):
        m = NAME.search(path)
        if not m:
            continue
        state, arm, rep = m.group(1), m.group(2), int(m.group(3))
        t = load(path)
        t["rep"] = rep
        cells[(state, arm)].append(t)
        rows.append((state, float(arm), rep, t))
    if not cells:
        print(f"no trace-*.jsonl in {out}")
        return 1

    run_txt = os.path.join(out, "run.txt")
    if os.path.exists(run_txt):
        print(open(run_txt).read().rstrip())
    any_run = rows[0][3]["run"]
    print(f"\nmodel: {any_run.get('layers')} layers x {any_run.get('experts_per_layer')} experts, "
          f"top-{any_run.get('top_k')}, stride {any_run.get('stride_bytes', 0) / 2**20:.2f} MiB, "
          f"prompt {any_run.get('prompt_tokens')} tokens, "
          f"page cache probe {'on' if any_run.get('page_cache_probe') else 'OFF (misses unclassified)'}")

    texts = sorted(glob.glob(os.path.join(out, "text-*.txt")))
    if texts:
        digests = {open(p, "rb").read() for p in texts}
        print(f"answers: {len(texts)} runs, {'all identical' if len(digests) == 1 else str(len(digests)) + ' DIFFERENT texts (routing differed between runs)'}")

    print("\nruns")
    for state, arm, rep, t in sorted(rows, key=lambda r: (r[0] != "cold", r[1], r[2])):
        print(run_line(state, f"{arm:g}", rep, t))

    arms = sorted({arm for _, arm in cells}, key=float)
    floor = arms[0]
    for state in ("cold", "warm"):
        if (state, floor) not in cells:
            continue
        for arm in arms[1:]:
            if (state, arm) in cells:
                for line in window_table(state, floor, arm, cells[(state, floor)], cells[(state, arm)], window):
                    print(line)

    for line in pairing_lines(cells, floor, arms):
        print(line)
    for line in breakdown_lines(cells):
        print(line)

    if oracle:
        for line in oracle_section(cells, arms):
            print(line)

    if png:
        maybe_png(png, cells, window)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
