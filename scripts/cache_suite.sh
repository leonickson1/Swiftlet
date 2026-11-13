#!/usr/bin/env bash
# Back-to-back prompt suite through one process and one expert cache, cache
# off (floor) against on, with the page cache cold and warm. Every pass
# leaves a per-step trace and the answers; the report and the quality score
# run at the end. See docs/CACHE_SUITE.md.
#
#   scripts/cache_suite.sh <model.qpack> <out-dir> [options]
#
#   --suite FILE      prompt suite, JSONL (default scripts/suites/core.jsonl)
#   --arms "0.03 8"   cache budgets in GiB; the first is the floor (cache off).
#                     A trailing "nc" (e.g. 8nc) runs that budget with
#                     --no-page-cache: misses bypass the OS file cache, so
#                     each one is a device read, emulating a machine whose
#                     RAM cannot hold the expert file. "lru" or "lfu"
#                     before any "nc" (8lfu, 8lfunc) picks the eviction
#                     policy explicitly; a bare budget uses the binary's
#                     default (LRU since the policy study)
#   --reps 1          repeats of every (arm, cold/warm) pass
#   --max-new-cap N   clamp every prompt's max_new to N (quick runs)
#   --shuffle SEED    run the prompts in a seeded random order instead of file order
#   --prefill-chunk N prefill schedule for every arm (default 0, token-major;
#                     the floor arm's 18 slots cannot hold a layer-major chunk)
#   --cold-by MODE    how a "cold" pass starts: evict (default) drops only the
#                     model's files from the page cache, no sudo needed
#                     (scripts/evict_page_cache.py); purge runs sudo purge,
#                     emptying the whole cache; none skips it, and "cold" is
#                     then just the first pass of a pair
#   --no-purge        same as --cold-by none
#   --cold-only       skip the warm pass after each cold one (for a model far
#                     larger than RAM, where nothing stays cached between passes)
#   --bin PATH        swiftlet binary (default .build/release/swiftlet)
#
# Each (rep, arm) does: [evict or purge] -> cold pass -> warm pass. A pass runs
# the whole suite in one `swiftlet generate --prompts-file` process and
# leaves suite-<state>-<arm>-<rep>.jsonl (trace), answers-*.jsonl, text-*.txt
# (stdout), log-*.txt (stderr) and mem-*.txt (swap and page cache before it).
set -euo pipefail

MODEL=${1:?usage: cache_suite.sh <model.qpack> <out-dir> [options]}
OUT=${2:?usage: cache_suite.sh <model.qpack> <out-dir> [options]}
shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE="$ROOT/scripts/suites/core.jsonl"
COLD_ONLY=0; ARMS="0.03 8"; REPS=1; CAP=""; SHUFFLE=""; PREFILL_CHUNK=0; COLD_BY=evict
BIN="$ROOT/.build/release/swiftlet"
while [ $# -gt 0 ]; do
  case "$1" in
    --suite) SUITE=$2; shift 2 ;;
    --arms) ARMS=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --max-new-cap) CAP=$2; shift 2 ;;
    --shuffle) SHUFFLE=$2; shift 2 ;;
    --prefill-chunk) PREFILL_CHUNK=$2; shift 2 ;;
    --cold-by) COLD_BY=$2; shift 2 ;;
    --no-purge) COLD_BY=none; shift ;;
    --cold-only) COLD_ONLY=1; shift ;;
    --bin) BIN=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "no binary at $BIN (run: swift build -c release)" >&2; exit 1; }
[ -f "$MODEL/manifest.json" ] || { echo "$MODEL is not a .qpack container" >&2; exit 1; }
[ -f "$SUITE" ] || { echo "no suite at $SUITE" >&2; exit 1; }
mkdir -p "$OUT"

# The suite as run, after any cap or shuffle, so the directory is self-describing.
python3 - "$SUITE" "$OUT/suite.jsonl" "$CAP" "$SHUFFLE" <<'EOF'
import json, random, sys
src, dst, cap, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [json.loads(l) for l in open(src) if l.strip() and not l.lstrip().startswith(("#", "//"))]
if cap:
    for r in rows:
        r["max_new"] = min(int(r.get("max_new", 256)), int(cap))
if seed:
    random.Random(int(seed)).shuffle(rows)
with open(dst, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"suite: {len(rows)} prompts, up to {sum(int(r.get('max_new', 256)) for r in rows)} new tokens per pass", file=sys.stderr)
EOF

case "$COLD_BY" in evict|purge|none) ;; *) echo "--cold-by must be evict, purge or none" >&2; exit 2 ;; esac
if [ "$COLD_BY" = purge ]; then
  if ! sudo -v; then
    echo "sudo is needed for purge; use --cold-by evict (no sudo) instead" >&2
    exit 1
  fi
  ( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) &
  KEEPALIVE=$!
  trap 'kill $KEEPALIVE 2>/dev/null || true' EXIT
fi

{
  echo "model=$MODEL"
  echo "suite=$SUITE"
  echo "arms=$ARMS"
  echo "reps=$REPS"
  echo "max_new_cap=${CAP:-none}"
  echo "order=${SHUFFLE:+shuffled seed $SHUFFLE}${SHUFFLE:-file order}"
  echo "prefill_chunk=$PREFILL_CHUNK"
  echo "cold_by=$COLD_BY"
  echo "cold_only=$COLD_ONLY"
  echo "bin=$BIN"
  echo "git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo '+dirty')"
  echo "started=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "os=$(sw_vers -productVersion)"
  echo "cpu=$(sysctl -n machdep.cpu.brand_string)"
  echo "ram_bytes=$(sysctl -n hw.memsize)"
} > "$OUT/run.txt"

run() { # state arm rep
  local tag="$1-$2-$3"
  echo "[$(date +%H:%M:%S)] $tag" >&2
  { sysctl -n vm.swapusage; vm_stat | head -14; uptime; } > "$OUT/mem-$tag.txt"
  local gb=$2 nocache="" policy=""
  case "$gb" in *nc) nocache="--no-page-cache"; gb=${gb%nc} ;; esac
  case "$gb" in
    *lru) policy="--cache-policy lru"; gb=${gb%lru} ;;
    *lfu) policy="--cache-policy lfu"; gb=${gb%lfu} ;;
  esac
  # shellcheck disable=SC2086  # $nocache and $policy are empty or flags
  "$BIN" generate "$MODEL" --gpu --prompts-file "$OUT/suite.jsonl" --cache-gb "$gb" $nocache $policy \
      --prefill-chunk "$PREFILL_CHUNK" \
      --answers "$OUT/answers-$tag.jsonl" \
      --trace-steps "$OUT/suite-$tag.jsonl" > "$OUT/text-$tag.txt" 2> "$OUT/log-$tag.txt" \
      || { echo "pass $tag failed; see $OUT/log-$tag.txt" >&2; exit 1; }
  grep -h "^suite: [0-9]\|^expert cache:" "$OUT/log-$tag.txt" | sed 's/^/    /' >&2
}

for rep in $(seq 1 "$REPS"); do
  for arm in $ARMS; do
    case "$COLD_BY" in
      purge) sudo purge; sleep 2 ;;
      evict) python3 "$ROOT/scripts/evict_page_cache.py" "$MODEL" >&2 \
               || { echo "eviction left pages resident; cold cell would not be cold" >&2; exit 1; } ;;
    esac
    run cold "$arm" "$rep"
    [ "$COLD_ONLY" = 1 ] || run warm "$arm" "$rep"
  done
done

echo >&2
python3 "$ROOT/scripts/cache_suite_report.py" "$OUT" --html "$OUT/cache-report.html" | tee "$OUT/report.txt"
FIRST=$(ls "$OUT"/answers-*.jsonl | head -1)
python3 "$ROOT/scripts/suite_score.py" "$OUT/suite.jsonl" "$FIRST" --out "$OUT/quality" | tee "$OUT/quality.txt"
