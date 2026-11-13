#!/usr/bin/env python3
"""Report for the suite runs scripts/cache_suite.sh leaves behind. See docs/CACHE_SUITE.md.

usage: cache_suite_report.py <out-dir> [--html report.html] [--no-oracle] [--self-test]

Reads every suite-<state>-<arm>-<rep>.jsonl (one trace per pass through the
whole prompt suite, one process, one expert cache) and prints:

  1. one line per pass: tokens, decode speed, hit rate, miss tiers, swap;
  2. whether every pass produced the same tokens for every prompt (greedy
     decoding makes this a requirement, not a hope);
  3. per prompt, for each page cache state: ms per token with the cache off
     (floor arm) and on, the saving, the hit rate, the hit rate in the
     prompt's first 16 decode steps (what the previous prompts left behind),
     and device reads per token;
  4. the session totals: decode and prefill time, floor against full;
  5. unless --no-oracle, an offline replay of the session's routing: the
     cache's own policy (must match the trace), LRU and Belady at the full
     arm's slot count, the hit rate each prompt would have had with an empty
     cache (so the value of carrying the cache across prompts is explicit),
     and the policy's hit rate at budgets that were not run;
  6. a reporting block after the checklist in Zhang, "Reproducible
     Evaluation of MoE Expert Caching" (arXiv:2608.07911, section 10).

Only the standard library is required. --html writes a self-contained page
with the same tables and inline charts.
"""
import glob
import heapq
import html
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_ab_report import mean, pct, replay_belady, replay_lfu_recency, replay_lru  # noqa: E402

NAME = re.compile(r"suite-(cold|warm)-([0-9.]+(?:lru|lfu)?(?:nc)?)-(\d+)\.jsonl$")


# Policy each arm actually ran with, read from its logs by main(); a log
# without an "expert cache policy:" line predates that line, when the
# default was LFU.
ARM_POLICY = {}


def arm_parts(arm):
    """(GiB, policy, page cache bypassed) of an arm label: 8, 8nc, 8lru, 8lfunc."""
    nc = arm.endswith("nc")
    a = arm[:-2] if nc else arm
    label = a[-3:] if a[-3:] in ("lru", "lfu") else None
    gb = float(a[:-3] if label else a)
    return gb, ARM_POLICY.get(arm, label or "lfu"), nc


def logged_policy(trace_path):
    try:
        for line in open(sibling(trace_path, "log-", ".txt")):
            if line.startswith("expert cache policy:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def arm_gb(arm):
    return arm_parts(arm)[0]


def arm_name(arm):
    gb, policy, nc = arm_parts(arm)
    return f"{gb:g} GiB " + policy.upper() + (", no page cache" if nc else "")
ARRIVAL_STEPS = 16


# ---- loading ---------------------------------------------------------------

def load(path):
    run, end = {}, None
    prompts = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r["kind"]
        if k == "run":
            run = r
        elif k == "end":
            end = r
        elif k == "prompt":
            prompts[r["prompt"]] = {"meta": r, "prefill": None, "decode": [], "end": None}
        elif k == "prompt_end":
            prompts[r["prompt"]]["end"] = r
        elif k in ("prefill", "decode"):
            p = prompts[r["prompt"]]
            if k == "prefill":
                p["prefill"] = r
            else:
                p["decode"].append(r)
    order = [prompts[i] for i in sorted(prompts)]
    return {"run": run, "end": end, "prompts": order, "path": path}


def sibling(trace_path, prefix, ext=".jsonl"):
    """answers-/log-/mem- file beside a suite- trace (the directory name may
    itself contain "suite-", so only the file name is rewritten)."""
    d, name = os.path.split(trace_path)
    return os.path.join(d, prefix + name[len("suite-"):-len(".jsonl")] + ext)


def load_answers(trace_path):
    path = sibling(trace_path, "answers-")
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def memory_note(trace_path):
    """'peak 3.1 GiB' from the "memory:" line generate prints at the end of a suite."""
    log = sibling(trace_path, "log-", ".txt")
    try:
        for line in open(log):
            if line.startswith("memory: peak footprint"):
                return "peak " + line.split("peak footprint")[1].split(",")[0].strip()
    except OSError:
        pass
    return ""


def swap_note(trace_path):
    mem = sibling(trace_path, "mem-", ".txt")
    try:
        for line in open(mem):
            if "used =" in line:
                return "swap " + line.split("used =")[1].split()[0]
    except OSError:
        pass
    return ""


# ---- per pass and per prompt numbers ----------------------------------------

def fold(steps):
    out = defaultdict(float)
    for s in steps:
        for f in ("wall_s", "fetch_s", "hits", "misses", "miss_page_cache", "miss_device", "miss_unprobed"):
            out[f] += s[f]
    out["n"] = len(steps)
    return out


def pass_line(state, arm, rep, t):
    dec = [s for p in t["prompts"] for s in p["decode"]]
    f = fold(dec)
    decode_s = sum(p["end"]["decode_s"] for p in t["prompts"] if p["end"])
    prefill_s = sum(p["end"]["prefill_s"] for p in t["prompts"] if p["end"])
    tokens = sum(p["end"]["generated"] for p in t["prompts"] if p["end"])
    m = f["misses"]
    tiers = f"misses {pct(f['miss_page_cache'], m):3.0f}% page cache / {pct(f['miss_device'], m):3.0f}% device"
    if f["miss_unprobed"]:
        tiers += f" / {pct(f['miss_unprobed'], m):3.0f}% unprobed"
    return (f"  {state:4} {arm:>6} #{rep}  {len(t['prompts'])} prompts {tokens:5d} tok  "
            f"decode {decode_s:7.1f} s {tokens / decode_s if decode_s else 0:5.2f} tok/s  prefill {prefill_s:6.1f} s  "
            f"hit {pct(f['hits'], f['hits'] + m):5.1f}%  fetch {pct(f['fetch_s'], f['wall_s']):4.1f}% of step  "
            f"{tiers}  {swap_note(t['path'])}  {memory_note(t['path'])}")


def prompt_stats(runs, i):
    """Mean over reps of one prompt's numbers in one (state, arm) cell."""
    rows = []
    for t in runs:
        p = t["prompts"][i]
        f = fold(p["decode"])
        early = fold(p["decode"][:ARRIVAL_STEPS])
        n = max(f["n"], 1)
        rows.append({
            "ms_tok": 1000 * f["wall_s"] / n,
            "hit": pct(f["hits"], f["hits"] + f["misses"]),
            "hit_arrival": pct(early["hits"], early["hits"] + early["misses"]),
            "dev_tok": f["miss_device"] / n,
            "prefill_s": p["end"]["prefill_s"] if p["end"] else 0.0,
            "tokens": f["n"],
        })
    return {k: mean(r[k] for r in rows) for k in rows[0]} if rows else None


# ---- fast replay for long sessions ----------------------------------------
# cache_ab_report's replays scan every slot per eviction, fine for one 256
# token answer, too slow for a whole suite at thousands of slots. These are
# the same policies with heaps; --self-test checks them against the originals.

def replay_lfu_fast(calls, slots):
    """Same policy as replay_lfu_recency: evict min (freq, last use, slot),
    batch members protected, grow on demand."""
    slot_key = []
    key_slot = {}
    freq = defaultdict(int)
    last_use = []
    heap = []  # (freq, last_use, slot, key), stale entries skipped
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
                heapq.heappush(heap, (freq[k], tick, s, k))
                h += 1
                continue
            if len(slot_key) < slots:
                slot_key.append(None)
                last_use.append(0)
                s = len(slot_key) - 1
            else:
                parked = []
                while True:
                    f, lu, s, kk = heapq.heappop(heap)
                    if slot_key[s] != kk or freq[kk] != f or last_use[s] != lu:
                        continue
                    if s in protected:
                        parked.append((f, lu, s, kk))
                        continue
                    break
                for p in parked:
                    heapq.heappush(heap, p)
                del key_slot[slot_key[s]]
            slot_key[s] = k
            key_slot[k] = s
            last_use[s] = tick
            protected.add(s)
            heapq.heappush(heap, (freq[k], tick, s, k))
        hits.append(h)
    return hits


def replay_lru_slots(calls, slots):
    """The cache's own LRU mode, slot for slot: evict min (last use, slot),
    batch members protected, grow on demand. (replay_lru breaks ties by
    admission order instead; the two agree on every hit except exact ties.)"""
    slot_key = []
    key_slot = {}
    last_use = []
    heap = []
    tick = 0
    hits = []
    for layer, experts in calls:
        tick += 1
        protected = set()
        h = 0
        for e in experts:
            k = (layer, e)
            if k in key_slot:
                s = key_slot[k]
                h += 1
            elif len(slot_key) < slots:
                slot_key.append(None)
                last_use.append(0)
                s = len(slot_key) - 1
            else:
                parked = []
                while True:
                    lu, s = heapq.heappop(heap)
                    if last_use[s] != lu:
                        continue
                    if s in protected:
                        parked.append((lu, s))
                        continue
                    break
                for p in parked:
                    heapq.heappush(heap, p)
                del key_slot[slot_key[s]]
            if slot_key[s] != k:
                slot_key[s] = k
                key_slot[k] = s
            last_use[s] = tick
            heapq.heappush(heap, (tick, s))
            protected.add(s)
        hits.append(h)
    return hits


def replay_lru_fast(calls, slots):
    """Same policy as replay_lru: evict the least recent call's key; keys
    touched by the same call tie, and the tie goes to the key admitted first
    (the reference's dict order), which a hit does not change."""
    resident = {}  # key -> (tick of last use, admission sequence)
    heap = []      # (tick, seq, key), stale entries skipped
    seq = 0
    tick = 0
    hits = []
    for layer, experts in calls:
        tick += 1
        protected = set()
        h = 0
        for e in experts:
            k = (layer, e)
            if k in resident:
                h += 1
                s = resident[k][1]
            else:
                if len(resident) >= slots:
                    parked = []
                    while True:
                        entry = heapq.heappop(heap)
                        t, s_, victim = entry
                        if resident.get(victim) != (t, s_):
                            continue
                        if victim in protected:
                            parked.append(entry)
                            continue
                        break
                    for p in parked:
                        heapq.heappush(heap, p)
                    del resident[victim]
                seq += 1
                s = seq
            resident[k] = (tick, s)
            heapq.heappush(heap, (tick, s, k))
            protected.add(k)
        hits.append(h)
    return hits


def self_test():
    rng = random.Random(7)
    for trial in range(200):
        layers, experts, k = rng.randint(1, 4), rng.randint(3, 12), rng.randint(1, 3)
        k = min(k, experts)
        calls = [(rng.randrange(layers), rng.sample(range(experts), k)) for _ in range(rng.randint(1, 120))]
        slots = rng.randint(k, layers * experts + 2)
        assert replay_lfu_fast(calls, slots) == replay_lfu_recency(calls, slots), ("lfu", trial)
        assert replay_lru_fast(calls, slots) == replay_lru(calls, slots), ("lru", trial)
    print("self-test: fast LFU and LRU replays match the reference replays on 200 random traces")


# ---- the report -------------------------------------------------------------

def session_calls(t):
    """[(layer, experts)] in call order plus each call's (prompt index, is_decode)."""
    calls, owner = [], []
    for i, p in enumerate(t["prompts"]):
        for kind, steps in (("prefill", [p["prefill"]] if p["prefill"] else []), ("decode", p["decode"])):
            for s in steps:
                for L in s["layers"]:
                    calls.append((L["l"], L["e"]))
                    owner.append((i, kind == "decode"))
    return calls, owner


def oracle_section(cells, arms, out):
    src = next((ts[0] for ts in cells.values() if ts), None)
    if src is None:
        return []
    calls, owner = session_calls(src)
    if not calls:
        return []
    run = src["run"]
    stride = run.get("stride_bytes", 0)
    n_prompts = len(src["prompts"])
    dec_idx = [j for j, (_, d) in enumerate(owner) if d]
    dec_fetch = sum(len(calls[j][1]) for j in dec_idx)
    unique = len({(l, e) for l, es in calls for e in es})
    lines = ["\noffline replay of the session's routing (prefill and decode, all prompts in order)",
             f"  {len(calls)} cache calls, {sum(len(e) for _, e in calls)} expert fetches, "
             f"{unique} distinct (layer, expert) pairs"
             + (f" = {unique * stride / 2**30:.2f} GiB to hold them all" if stride else "")]
    out["replay"] = []
    # A floor, or an "nc" arm, replays exactly like the budget it shares
    # slots with: replay each distinct slot count once, floors excluded.
    floor_slots = {t["run"].get("slots") for (s, a), ts in cells.items()
                   if a in (arms[0], arms[0] + "nc") for t in ts} if len(arms) > 1 else set()
    replayed = set(floor_slots)
    for arm in arms[1:] or arms:
        runs = [t for (s, a), ts in cells.items() if a == arm for t in ts]
        if not runs or not runs[0]["run"].get("slots"):
            continue
        slots = runs[0]["run"]["slots"]
        traced = runs[0]["end"]["hits"] if runs[0]["end"] else None
        if arm_parts(arm)[1] == "lru":
            # Check the LRU arm's trace against the LRU mode replayed slot for
            # slot; the LFU replay at this budget is reported by its own arm.
            mine = sum(replay_lru_slots(calls, slots))
            lines.append(f"  {arm_name(arm)} = {slots} slots: replayed LRU {mine} hits "
                         f"({'matches' if traced == mine else 'DIFFERS from'} the trace's {traced})")
            continue
        if slots in replayed:
            continue
        replayed.add(slots)
        own = replay_lfu_fast(calls, slots)
        lru = replay_lru_fast(calls, slots)
        bel = replay_belady(calls, slots)
        dsum = lambda hs: sum(hs[j] for j in dec_idx)  # noqa: E731
        lines.append(
            f"  {arm_name(arm)} = {slots} slots: replayed policy {sum(own)} hits "
            f"({'matches' if traced == sum(own) else 'DIFFERS from'} the trace's {traced}); decode hit rate: "
            f"policy {pct(dsum(own), dec_fetch):5.1f}%, LRU {pct(dsum(lru), dec_fetch):5.1f}%, "
            f"Belady optimal {pct(dsum(bel), dec_fetch):5.1f}%")
        # Carry-over: each prompt replayed alone from an empty cache.
        lines.append(f"  carry-over at {arm} GiB: decode hit rate per prompt with the cache the previous prompts "
                     f"left (session) against an empty cache (isolated)")
        lines.append(f"    {'#':>2} {'prompt':24} {'session':>8} {'isolated':>9} {'gain':>6}  "
                     f"{'first ' + str(ARRIVAL_STEPS) + ' steps: session':>24} {'isolated':>9}")
        tot_s = tot_i = tot_f = 0
        rows = []
        for i in range(n_prompts):
            idx = [j for j, (pi, _) in enumerate(owner) if pi == i]
            sub = [calls[j] for j in idx]
            iso = replay_lfu_fast(sub, slots)
            d_local = [n for n, j in enumerate(idx) if owner[j][1]]
            fetch = sum(len(sub[n][1]) for n in d_local)
            s_h = sum(own[idx[n]] for n in d_local)
            i_h = sum(iso[n] for n in d_local)
            layers = run.get("layers", 1) or 1
            early = d_local[:ARRIVAL_STEPS * layers]
            e_fetch = sum(len(sub[n][1]) for n in early)
            e_s = sum(own[idx[n]] for n in early)
            e_i = sum(iso[n] for n in early)
            tot_s += s_h; tot_i += i_h; tot_f += fetch
            name = src["prompts"][i]["meta"]["id"]
            rows.append({"prompt": name, "session": pct(s_h, fetch), "isolated": pct(i_h, fetch),
                         "arrival_session": pct(e_s, e_fetch), "arrival_isolated": pct(e_i, e_fetch)})
            lines.append(f"    {i + 1:>2} {name[:24]:24} {pct(s_h, fetch):7.1f}% {pct(i_h, fetch):8.1f}% "
                         f"{pct(s_h, fetch) - pct(i_h, fetch):+5.1f}  {pct(e_s, e_fetch):23.1f}% {pct(e_i, e_fetch):8.1f}%")
        lines.append(f"    all prompts: session {pct(tot_s, tot_f):.1f}% against isolated {pct(tot_i, tot_f):.1f}%: "
                     f"carrying the cache across prompts is worth {pct(tot_s, tot_f) - pct(tot_i, tot_f):+.1f} points of hit rate")
        out["replay"].append({"arm": arm, "slots": slots, "policy": pct(dsum(own), dec_fetch),
                              "lru": pct(dsum(lru), dec_fetch), "belady": pct(dsum(bel), dec_fetch),
                              "session": pct(tot_s, tot_f), "isolated": pct(tot_i, tot_f), "rows": rows})
    if stride:
        ran = {arm_gb(a) for a in arms}
        extra = []
        for gb in (0.5, 1, 2, 4, 6, 8, 12, 16):
            if gb in ran:
                continue
            slots = int(gb * 2**30 // stride)
            if slots < 16:
                continue
            own = replay_lfu_fast(calls, slots)
            extra.append((gb, slots, pct(sum(own[j] for j in dec_idx), dec_fetch)))
        if extra:
            lines.append("  policy decode hit rate at budgets not run (same routing, replayed): "
                         + ", ".join(f"{gb:g} GiB {h:.1f}%" for gb, _, h in extra))
        out["budgets"] = extra
    return lines


def checklist(src, arms, cells, identical):
    run = src["run"]
    L, k, slots = run.get("layers", 0), run.get("top_k", 0), None
    stride = run.get("stride_bytes", 0)
    lines = ["\nreporting block (after arXiv:2608.07911 section 10)"]
    lines.append(f"  A1 execution unit: one cache call per (step, layer); batch size 1, so a decode call asks for top-{k} "
                 f"experts and a token-major prefill call likewise; real end-to-end system, so A2-A4 are not applicable")
    for arm in arms:
        runs = [t for (s, a), ts in cells.items() if a == arm for t in ts]
        if not runs:
            continue
        slots = runs[0]["run"].get("slots", 0)
        per_layer = slots / L if L else 0
        lines.append(f"  C1/C2 {arm_name(arm)}: one global pool of {slots} slots across {L} layers "
                     f"({per_layer:.1f} per layer on average); per-step union {k} per layer, "
                     f"union/capacity r = {k / per_layer if per_layer else float('inf'):.2f}")
        dec = [s for t in runs for p in t["prompts"] for s in p["decode"]]
        if dec and stride:
            f = fold(dec)
            n = len(dec)
            lines.append(f"  D1 {arm_name(arm)}: {f['misses'] * stride / n / 2**20:.1f} MiB read per output token "
                         f"({f['miss_device'] * stride / n / 2**20:.1f} MiB of it from the device)")
    lines.append("  C3 miss denominator: every expert a call asks for (no deduplication at batch size 1)")
    lines.append("  B2 chat template: applied, reasoning disabled (same rendering as the app's fresh conversation)")
    n_steps = sum(len(p["decode"]) for p in src["prompts"])
    lines.append(f"  B3 decode steps recorded: {n_steps} over {len(src['prompts'])} prompts "
                 f"(min {min(len(p['decode']) for p in src['prompts'])}, max {max(len(p['decode']) for p in src['prompts'])})")
    cats = defaultdict(int)
    for p in src["prompts"]:
        cats[p["meta"]["category"]] += 1
    lines.append("  B1 prompts per category: " + ", ".join(f"{c} {n}" for c, n in sorted(cats.items()))
                 + "; every prompt is written separately, no shared instruction template beyond the math suffix")
    lines.append(f"  B5 generated sequences identical across passes: {identical}")
    return lines


def write_html(path, title, sections, per_prompt, out):
    def bars(rows, key_a, key_b, label_a, label_b, unit):
        w, rh = 640, 18
        h = rh * len(rows) + 30
        vmax = max([r[key_a] for r in rows] + [r[key_b] for r in rows] + [1e-9])
        parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
        for n, r in enumerate(rows):
            y = 20 + n * rh
            parts.append(f'<text x="0" y="{y + 11}" class="lbl">{html.escape(r["prompt"][:22])}</text>')
            for off, key, cls in ((0, key_a, "a"), (8, key_b, "b")):
                bw = (w - 200) * r[key] / vmax
                parts.append(f'<rect x="170" y="{y + off - 1}" width="{bw:.1f}" height="7" class="{cls}"/>')
            parts.append(f'<text x="{w - 2}" y="{y + 11}" class="val" text-anchor="end">'
                         f'{r[key_a]:.1f} / {r[key_b]:.1f} {unit}</text>')
        parts.append(f'<text x="170" y="12" class="key"><tspan class="ka">■</tspan> {label_a}  '
                     f'<tspan class="kb">■</tspan> {label_b}</text></svg>')
        return "".join(parts)

    charts = []
    for state, rows in per_prompt.items():
        if rows:
            charts.append(f"<h2>{state}: ms per token, cache off vs on</h2>"
                          + bars(rows, "floor_ms", "full_ms", "cache off (floor)", "cache on", "ms"))
    for rep in out.get("replay", []):
        charts.append(f"<h2>Carry-over at {rep['arm']} GiB: decode hit rate, session vs empty cache</h2>"
                      + bars(rep["rows"], "session", "isolated", "cache carried from earlier prompts",
                             "empty cache per prompt", "%"))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --a:#b4532a; --b:#2f6f8f; --rule:#e2dfd7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#171716; --fg:#ecebe6; --muted:#a3a29b; --a:#e07a4f; --b:#6fb3d2; --rule:#33322f; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; max-width:960px; margin:0 auto; padding:24px 16px; }}
pre {{ font:12px/1.45 ui-monospace, Menlo, monospace; overflow-x:auto; border-top:1px solid var(--rule); padding-top:12px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin-top:28px; }}
svg .lbl, svg .val, svg .key {{ font:11px ui-monospace, Menlo, monospace; fill:var(--fg); }}
svg .val {{ fill:var(--muted); }} svg .a, .ka {{ fill:var(--a); }} svg .b, .kb {{ fill:var(--b); }}
</style></head><body><h1>{html.escape(title)}</h1>
{''.join(charts)}
<h2>Full report</h2><pre>{html.escape(chr(10).join(sections))}</pre></body></html>"""
    with open(path, "w") as f:
        f.write(doc)


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    if len(argv) < 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    outdir = argv[1]
    html_path = None
    oracle = True
    i = 2
    while i < len(argv):
        if argv[i] == "--html":
            html_path = argv[i + 1]; i += 2
        elif argv[i] == "--no-oracle":
            oracle = False; i += 1
        else:
            print(f"unknown option {argv[i]}"); return 2

    cells = defaultdict(list)
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, "suite-*.jsonl"))):
        m = NAME.search(path)
        if not m:
            continue
        t = load(path)
        if not t["prompts"]:
            continue
        state, arm, rep = m.group(1), m.group(2), int(m.group(3))
        t["answers"] = load_answers(path)
        explicit = arm.removesuffix("nc")[-3:]
        ARM_POLICY[arm] = logged_policy(path) or (explicit if explicit in ("lru", "lfu") else "lfu")
        cells[(state, arm)].append(t)
        rows.append((state, arm, rep, t))
    if not cells:
        print(f"no suite-*.jsonl in {outdir}")
        return 1

    lines = []
    run_txt = os.path.join(outdir, "run.txt")
    if os.path.exists(run_txt):
        lines.append(open(run_txt).read().rstrip())
    src = rows[0][3]
    run = src["run"]
    lines.append(f"\nmodel: {run.get('layers')} layers x {run.get('experts_per_layer')} experts, top-{run.get('top_k')}, "
                 f"stride {run.get('stride_bytes', 0) / 2**20:.2f} MiB, {len(src['prompts'])} prompts, "
                 f"page cache probe {'on' if run.get('page_cache_probe') else 'OFF (misses unclassified)'}")

    # Identity: same tokens for every prompt in every pass.
    sigs = set()
    for _, _, _, t in rows:
        if t["answers"]:
            sigs.add(tuple(tuple(a["tokens"]) for a in t["answers"]))
    if not sigs:
        identical = "unknown (no answers files)"
    elif len(sigs) == 1:
        identical = f"yes, all {len(rows)} passes"
    else:
        identical = f"NO: {len(sigs)} different token streams, so the arms did not see the same routing"
    lines.append(f"answers identical across passes: {identical}")

    lines.append("\npasses")
    for state, arm, rep, t in sorted(rows, key=lambda r: (r[0] != "cold", arm_gb(r[1]), r[1], r[2])):
        lines.append(pass_line(state, arm, rep, t))

    arms = sorted({a for _, a in cells}, key=lambda a: (arm_gb(a), a))
    per_prompt = {}
    out = {}
    for state in ("cold", "warm"):
        for full in arms:
            # Each arm against the smallest budget of its own kind: a page
            # cache arm against the plain floor, an "nc" arm against the
            # smallest "nc" arm (nothing cached anywhere) when one ran.
            family = [a for a in arms if a.endswith("nc") == full.endswith("nc")] or arms
            floor = family[0] if len(family) > 1 or not full.endswith("nc") else arms[0]
            if full == floor:
                continue
            if (state, floor) not in cells or (state, full) not in cells:
                continue
            fr, ur = cells[(state, floor)], cells[(state, full)]
            lines.append(f"\n{state} page cache, per prompt: cache off ({arm_name(floor)}) against {arm_name(full)}, mean over reps")
            lines.append(f"  {'#':>2} {'prompt':24} {'tok':>4} {'off ms/tok':>10} {'on ms/tok':>9} {'saving':>7} "
                         f"{'hit':>6} {'hit first ' + str(ARRIVAL_STEPS):>12} {'dev/tok off':>11} {'dev/tok on':>10} {'prefill off/on s':>17}")
            prow = []
            tot_f = tot_u = 0.0
            for i in range(len(src["prompts"])):
                a, b = prompt_stats(fr, i), prompt_stats(ur, i)
                name = src["prompts"][i]["meta"]["id"]
                tot_f += a["ms_tok"] * a["tokens"]
                tot_u += b["ms_tok"] * b["tokens"]
                prow.append({"prompt": name, "floor_ms": a["ms_tok"], "full_ms": b["ms_tok"]})
                lines.append(f"  {i + 1:>2} {name[:24]:24} {int(b['tokens']):4d} {a['ms_tok']:10.1f} {b['ms_tok']:9.1f} "
                             f"{pct(a['ms_tok'] - b['ms_tok'], a['ms_tok']):6.1f}% {b['hit']:5.1f}% {b['hit_arrival']:11.1f}% "
                             f"{a['dev_tok']:11.1f} {b['dev_tok']:10.1f} {a['prefill_s']:8.1f}/{b['prefill_s']:.1f}")
            lines.append(f"  session decode: {tot_f / 1000:.1f} s off, {tot_u / 1000:.1f} s on, "
                         f"saving {pct(tot_f - tot_u, tot_f):.1f}% ({(tot_f - tot_u) / 1000:+.1f} s)")
            per_prompt[f"{state} page cache, {arm_name(full)}"] = prow

    if oracle:
        lines += oracle_section(cells, arms, out)
    lines += checklist(src, arms, cells, identical)

    print("\n".join(lines))
    if html_path:
        write_html(html_path, "Expert cache: suite report", lines, per_prompt, out)
        print(f"\nhtml report written to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
