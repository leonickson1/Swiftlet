#!/usr/bin/env bash
# Floor-vs-full expert cache A/B over one whole answer, with the page cache
# cold and warm, and a per-step trace of every run. See docs/CACHE_AB.md.
#
#   scripts/cache_ab.sh <model.qpack> <out-dir> [options]
#
#   --arms "0.03 8"   cache budgets in GiB to compare; the first is the floor
#   --reps 3          repeats per arm
#   --max-new 256     tokens per answer
#   --prompt "..."    the prompt (a long explanatory question by default)
#   --ids 1,2,3       raw token ids instead of a prompt (fixtures without a tokenizer)
#   --prefill-chunk N prefill schedule for every arm (default 0, token-major):
#                     layer-major prefill asks one layer for the union of a
#                     chunk's experts in one batch, and the floor arm's 18
#                     slots cannot hold that; token-major keeps every batch at
#                     top-k. Decode is unaffected either way, and both arms
#                     share whatever schedule you pick.
#   --chat            wrap the prompt in the chat template
#   --no-purge        skip sudo purge; the "cold" cell is then just the first
#                     run of each pair, with whatever the page cache held
#   --bin PATH        swiftlet binary (default .build/release/swiftlet)
#
# Each (rep, arm) does: [sudo purge] -> cold run -> warm run. Every run leaves
# text-<state>-<arm>-<rep>.txt (the answer), log-*.txt (stderr) and
# trace-*.jsonl (one JSON line per step). Decoding is greedy, so every run
# must print the same answer; the script checks that before summarising.
set -euo pipefail

MODEL=${1:?usage: cache_ab.sh <model.qpack> <out-dir> [options]}
OUT=${2:?usage: cache_ab.sh <model.qpack> <out-dir> [options]}
shift 2
ARMS="0.03 8"; REPS=3; MAX_NEW=256; PURGE=1; CHAT=""; IDS=""; PREFILL_CHUNK=0
BIN="$(cd "$(dirname "$0")/.." && pwd)/.build/release/swiftlet"
PROMPT="Explain, in plain language and in some detail, how a mixture of experts language model decides which experts handle each token, why that makes the model cheaper to run than a dense model of the same size, and what it costs in memory and on disk. Then give two concrete examples."
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --max-new) MAX_NEW=$2; shift 2 ;;
    --prompt) PROMPT=$2; shift 2 ;;
    --chat) CHAT="--chat"; shift ;;
    --ids) IDS=$2; shift 2 ;;
    --prefill-chunk) PREFILL_CHUNK=$2; shift 2 ;;
    --no-purge) PURGE=0; shift ;;
    --bin) BIN=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "no binary at $BIN (run: swift build -c release)" >&2; exit 1; }
[ -f "$MODEL/manifest.json" ] || { echo "$MODEL is not a .qpack container" >&2; exit 1; }
mkdir -p "$OUT"

if [ "$PURGE" = 1 ]; then
  if ! sudo -v; then
    echo "sudo is needed for purge; rerun with --no-purge for warm cells only" >&2
    exit 1
  fi
  # Keep the sudo timestamp alive for the whole run so purge never prompts
  # in the middle of a cell.
  ( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) &
  KEEPALIVE=$!
  trap 'kill $KEEPALIVE 2>/dev/null || true' EXIT
fi

{
  echo "model=$MODEL"
  echo "arms=$ARMS"
  echo "reps=$REPS"
  echo "max_new=$MAX_NEW"
  echo "chat=${CHAT:-no}"
  echo "prefill_chunk=$PREFILL_CHUNK"
  echo "purge=$PURGE"
  echo "bin=$BIN"
  echo "started=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "os=$(sw_vers -productVersion)"
  echo "cpu=$(sysctl -n machdep.cpu.brand_string)"
  echo "ram_bytes=$(sysctl -n hw.memsize)"
  if [ -n "$IDS" ]; then echo "ids=$IDS"; else echo "prompt=$PROMPT"; fi
} > "$OUT/run.txt"

if [ -n "$IDS" ]; then INPUT=(--ids "$IDS"); else INPUT=(--prompt "$PROMPT"); fi

run() { # state arm rep
  local tag="$1-$2-$3"
  echo "[$(date +%H:%M:%S)] $tag" >&2
  # Memory state the run starts in. Swap in use and compressor size decide
  # whether an 8 GB pool stays resident; the report prints the swap figure.
  { sysctl -n vm.swapusage; vm_stat | head -14; uptime; } > "$OUT/mem-$tag.txt"
  "$BIN" generate "$MODEL" --gpu $CHAT "${INPUT[@]}" --max-new "$MAX_NEW" --cache-gb "$2" \
      --prefill-chunk "$PREFILL_CHUNK" \
      --trace-steps "$OUT/trace-$tag.jsonl" > "$OUT/text-$tag.txt" 2> "$OUT/log-$tag.txt" \
      || { echo "run $tag failed; see $OUT/log-$tag.txt" >&2; exit 1; }
  grep -h "^decode:\|^expert cache:" "$OUT/log-$tag.txt" | sed 's/^/    /' >&2
}

for rep in $(seq 1 "$REPS"); do
  for arm in $ARMS; do
    if [ "$PURGE" = 1 ]; then sudo purge; sleep 2; fi
    run cold "$arm" "$rep"
    run warm "$arm" "$rep"
  done
done

echo >&2
distinct=$(shasum "$OUT"/text-*.txt | awk '{print $1}' | sort -u | wc -l | tr -d ' ')
if [ "$distinct" = 1 ]; then
  echo "all $(ls "$OUT"/text-*.txt | wc -l | tr -d ' ') answers identical (greedy, as required)" >&2
else
  echo "WARNING: answers differ between runs, so the arms did not see the same routing:" >&2
  shasum "$OUT"/text-*.txt >&2
fi
python3 "$(dirname "$0")/cache_ab_report.py" "$OUT" | tee "$OUT/report.txt"
