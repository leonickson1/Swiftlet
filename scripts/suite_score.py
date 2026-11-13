#!/usr/bin/env python3
"""Score a suite's answers, optionally against a reference run of the same suite.

usage: suite_score.py <suite.jsonl> <answers.jsonl> [--reference ref.jsonl] [--names swiftlet,reference]
                      [--out DIR]

<answers.jsonl> is what `swiftlet generate --prompts-file --answers` writes;
--reference is the same format, typically from scripts/suite_reference.py
(mlx-lm with the same weights held fully in memory, or a full precision
model behind an OpenAI-compatible API).

Every prompt's `check` decides how it is graded:
  number     the last 'Answer: <n>' line (else the last number) equals `answer`
  python     the reply's code defines `entry`, and every test in `tests` passes
             (run in a separate python3 -I process with a 10 s timeout)
  contains   the reply contains one of `any`
  rules      verifiable format rules: numbered_items, bullet_items, lowercase
  json       the reply is exactly one JSON object with `keys`
  svg        the reply holds a well-formed <svg> document (and `must_contain`)
  open       not graded; read it, and see the repetition column

Every answer also gets a repetition score: the share of its word 4-grams
that repeat an earlier 4-gram (0 for fresh text; looping text climbs fast).
With a reference: whether the two answers are token-identical (when both
carry the same tokenizer's ids), else the first differing character, and a
similarity ratio.

--out DIR writes DIR/svg/*.svg, DIR/scores.json and DIR/gallery.html, a
self-contained side-by-side page of every answer with SVGs rendered.
"""
import difflib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip() and not l.lstrip().startswith(("#", "//"))]


# ---- checks -----------------------------------------------------------------

NUM = r"-?\$?\s*-?[0-9][0-9,]*(?:\.[0-9]+)?"


def to_number(s):
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def check_number(text, spec):
    lines = re.findall(r"answer\s*[:=]\s*\**\s*(" + NUM + ")", text, flags=re.I)
    source = "Answer line"
    if not lines:
        lines = re.findall(NUM, text)
        source = "last number (no Answer line)"
    if not lines:
        return False, "no number in the reply"
    got = to_number(lines[-1])
    ok = got is not None and abs(got - float(spec["answer"])) < 1e-6
    return ok, f"{source}: {lines[-1].strip()} (expected {spec['answer']})"


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def check_python(text, spec):
    code = extract_code(text)
    harness = (code + "\n\nimport json, sys\n"
               "def _norm(x):\n"
               "    if isinstance(x, (list, tuple)): return [_norm(v) for v in x]\n"
               "    return x\n"
               f"_tests = json.loads({json.dumps(json.dumps(spec['tests']))})\n"
               "_fails = []\n"
               "for _args, _want in _tests:\n"
               "    try:\n"
               f"        _got = _norm({spec['entry']}(*_args))\n"
               "    except Exception as e:\n"
               "        _got = f'raised {type(e).__name__}: {e}'\n"
               "    if _got != _norm(_want): _fails.append((_args, _want, _got))\n"
               "print(json.dumps({'n': len(_tests), 'fails': _fails}, default=str))\n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "answer.py")
        with open(path, "w") as f:
            f.write(harness)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timed out"
    if r.returncode != 0:
        err = (r.stderr.strip().splitlines() or ["failed"])[-1]
        return False, f"did not run: {err[:120]}"
    res = json.loads(r.stdout.strip().splitlines()[-1])
    if res["fails"]:
        a, w, g = res["fails"][0]
        return False, f"{res['n'] - len(res['fails'])}/{res['n']} tests pass; e.g. {a} -> {g!r}, expected {w!r}"
    return True, f"{res['n']}/{res['n']} tests pass"


def check_contains(text, spec):
    hit = [a for a in spec["any"] if a in text]
    return bool(hit), f"found {hit[0]!r}" if hit else f"none of {spec['any']} in the reply"


def check_rules(text, spec):
    body = [l for l in text.strip().splitlines() if l.strip()]
    notes, ok = [], True
    if "numbered_items" in spec:
        items = [l for l in body if re.match(r"\s*\d+[.)]\s+\S", l)]
        good = len(items) == spec["numbered_items"] and len(items) == len(body)
        ok &= good
        notes.append(f"{len(items)} numbered items, {len(body) - len(items)} other lines")
    if "bullet_items" in spec:
        items = [l for l in body if re.match(r"\s*[-*\u2022]\s+\S", l)]
        good = len(items) == spec["bullet_items"] and len(items) == len(body)
        ok &= good
        notes.append(f"{len(items)} bullets, {len(body) - len(items)} other lines")
    if spec.get("lowercase"):
        upper = [c for c in text if c.isupper()]
        ok &= not upper
        notes.append("all lowercase" if not upper else f"{len(upper)} uppercase letters")
    return ok, "; ".join(notes)


def check_json(text, spec):
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError as e:
        return False, f"not bare JSON ({e.msg})"
    if not isinstance(obj, dict):
        return False, "JSON but not an object"
    keys = set(obj)
    want = set(spec["keys"])
    return keys == want, "keys match" if keys == want else f"keys {sorted(keys)}, expected {sorted(want)}"


def extract_svg(text):
    m = re.search(r"<svg\b.*?</svg>", text, flags=re.S | re.I)
    return m.group(0) if m else None


def check_svg(text, spec):
    svg = extract_svg(text)
    if not svg:
        return False, "no <svg>...</svg> in the reply"
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return False, f"<svg> present but not well-formed XML ({e})"
    shapes = sum(1 for el in root.iter() if el.tag.split("}")[-1] in
                 ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text"))
    missing = [s for s in spec.get("must_contain", []) if s not in svg]
    if missing:
        return False, f"well-formed, {shapes} shapes, missing {missing}"
    return True, f"well-formed, {shapes} shapes, {len(svg)} bytes"


CHECKS = {"number": check_number, "python": check_python, "contains": check_contains,
          "rules": check_rules, "json": check_json, "svg": check_svg}


def grade(text, spec):
    kind = (spec or {}).get("type", "open")
    if kind == "open" or kind not in CHECKS:
        return None, "not graded (read it)"
    return CHECKS[kind](text, spec)


def repetition(text, n=4):
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


# ---- report -----------------------------------------------------------------

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def mark(ok):
    return "n/a " if ok is None else ("PASS" if ok else "FAIL")


def main(argv):
    if len(argv) < 3 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    suite = {p["id"]: p for p in load_jsonl(argv[1])}
    runs = [load_jsonl(argv[2])]
    names = ["swiftlet", "reference"]
    out = None
    i = 3
    while i < len(argv):
        if argv[i] == "--reference":
            runs.append(load_jsonl(argv[i + 1])); i += 2
        elif argv[i] == "--names":
            names = argv[i + 1].split(","); i += 2
        elif argv[i] == "--out":
            out = argv[i + 1]; i += 2
        else:
            print(f"unknown option {argv[i]}"); return 2
    names = (names + ["reference"])[:len(runs)]
    by_id = [{a["id"]: a for a in r} for r in runs]
    order = [a["id"] for a in runs[0]]

    results = []
    print(f"{'prompt':20} {'category':14} " + " ".join(f"{n[:10]:>10} {'rep4':>5}" for n in names)
          + ("  agreement" if len(runs) > 1 else ""))
    tally = [defaultdict(lambda: [0, 0]) for _ in runs]
    for pid in order:
        spec = suite.get(pid, {}).get("check")
        cat = suite.get(pid, {}).get("category", "?")
        row = {"id": pid, "category": cat, "prompt": suite.get(pid, {}).get("prompt", ""), "runs": []}
        cells = []
        for r, ans in enumerate(by_id):
            a = ans.get(pid)
            if a is None:
                row["runs"].append(None)
                cells.append(f"{'missing':>10} {'':>5}")
                continue
            text = a.get("text") or ""
            ok, detail = grade(text, spec)
            rep = repetition(text)
            row["runs"].append({"name": names[r], "ok": ok, "detail": detail, "repetition": rep,
                                "text": text, "generated": a.get("generated"), "hit_eos": a.get("hit_eos"),
                                "tok_s": a.get("decode_tok_s")})
            if ok is not None:
                tally[r][cat][0] += ok
                tally[r][cat][1] += 1
            cells.append(f"{mark(ok):>10} {rep:5.2f}")
        agree = ""
        if len(runs) > 1 and all(x is not None for x in row["runs"][:2]):
            a0, a1 = by_id[0][pid], by_id[1][pid]
            if a0.get("tokens") and a1.get("tokens"):
                d = first_diff(a0["tokens"], a1["tokens"])
                agree = "identical tokens" if d is None else f"tokens diverge at {d}"
            else:
                t0, t1 = row["runs"][0]["text"], row["runs"][1]["text"]
                d = first_diff(t0, t1)
                agree = "identical text" if d is None else f"text diverges at char {d}"
            ratio = difflib.SequenceMatcher(None, row["runs"][0]["text"], row["runs"][1]["text"]).ratio()
            agree += f", similarity {ratio:.2f}"
            row["agreement"] = agree
        print(f"{pid[:20]:20} {cat[:14]:14} " + " ".join(cells) + (f"  {agree}" if agree else ""))
        for x in row["runs"]:
            if x and x["ok"] is not None:
                print(f"    {x['name']}: {x['detail']}")
            if x and x["generated"] and not x["hit_eos"]:
                print(f"    {x['name']}: stopped at max_new ({x['generated']} tokens), answer may be cut off")
        results.append(row)

    print("\nby category (graded prompts only)")
    cats = sorted({c for t in tally for c in t})
    for c in cats:
        print(f"  {c:14} " + "  ".join(f"{names[r]} {t[c][0]}/{t[c][1]}" for r, t in enumerate(tally) if c in t))
    for r, t in enumerate(tally):
        p = sum(v[0] for v in t.values())
        n = sum(v[1] for v in t.values())
        reps = [x["runs"][r]["repetition"] for x in results if len(x["runs"]) > r and x["runs"][r]]
        print(f"  {names[r]}: {p}/{n} graded checks pass; mean repetition {sum(reps) / max(len(reps), 1):.3f}")

    if out:
        os.makedirs(os.path.join(out, "svg"), exist_ok=True)
        for row in results:
            for x in row["runs"]:
                if not x:
                    continue
                svg = extract_svg(x["text"])
                if svg:
                    name = f"{row['id']}-{re.sub(r'[^A-Za-z0-9_.-]', '_', x['name'])}.svg"
                    with open(os.path.join(out, "svg", name), "w") as f:
                        f.write(svg)
                    x["svg_file"] = "svg/" + name
        with open(os.path.join(out, "scores.json"), "w") as f:
            json.dump(results, f, indent=1)
        write_gallery(os.path.join(out, "gallery.html"), results, names)
        print(f"\nwrote {out}/gallery.html, {out}/scores.json and {out}/svg/")
    return 0


def write_gallery(path, results, names):
    cards = []
    for row in results:
        cols = []
        for x in row["runs"]:
            if not x:
                cols.append('<div class="col"><p class="muted">missing</p></div>')
                continue
            badge = {None: "open", True: "pass", False: "fail"}[x["ok"]]
            img = (f'<img src="{html.escape(x["svg_file"])}" alt="{html.escape(row["id"])} as drawn by {html.escape(x["name"])}">'
                   if x.get("svg_file") else "")
            cols.append(
                f'<div class="col"><div class="head"><b>{html.escape(x["name"])}</b>'
                f'<span class="badge {badge}">{badge}</span></div>'
                f'<p class="muted">{html.escape(x["detail"])} · repetition {x["repetition"]:.2f}'
                + (f' · {x["tok_s"]:.1f} tok/s' if x.get("tok_s") else "") + "</p>"
                f'{img}<details{" open" if not img else ""}><summary>text</summary><pre>{html.escape(x["text"])}</pre></details></div>')
        agree = f'<p class="muted">{html.escape(row.get("agreement", ""))}</p>' if row.get("agreement") else ""
        cards.append(f'<section><h2>{html.escape(row["id"])} <span class="muted">{html.escape(row["category"])}</span></h2>'
                     f'<p class="prompt">{html.escape(row["prompt"][:600])}</p>{agree}<div class="cols">{"".join(cols)}</div></section>')
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Suite Answers</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --rule:#e2dfd7; --card:#ffffff; --pass:#2e7d4f; --fail:#b3261e; --open:#6b6a65; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#171716; --fg:#ecebe6; --muted:#a3a29b; --rule:#33322f; --card:#1f1f1d; --pass:#6cc58f; --fail:#f2867d; --open:#a3a29b; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; max-width:1200px; margin:0 auto; padding:24px 16px; }}
section {{ border-top:1px solid var(--rule); padding:16px 0; }}
h2 {{ font-size:17px; margin:0 0 6px; }} .muted {{ color:var(--muted); font-weight:400; font-size:13px; }}
.prompt {{ font-style:italic; }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }}
.col {{ background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:12px; min-width:0; }}
.head {{ display:flex; justify-content:space-between; align-items:center; }}
.badge {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.pass {{ color:var(--pass); }} .fail {{ color:var(--fail); }} .open {{ color:var(--open); }}
img {{ width:100%; max-height:420px; object-fit:contain; background:#fff; border-radius:4px; }}
pre {{ white-space:pre-wrap; word-break:break-word; font:12px/1.45 ui-monospace, Menlo, monospace; }}
</style></head><body><h1>Suite answers: {html.escape(" vs ".join(names))}</h1>{"".join(cards)}</body></html>"""
    with open(path, "w") as f:
        f.write(doc)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
