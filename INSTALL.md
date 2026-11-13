# Installing and testing Swiftlet on a Mac

This guide takes you from a Mac with nothing installed to a working local chat
with a 35B or 80B model. Follow it top to bottom. Every step says what you
should see, so you can tell straight away whether it worked.

## What you are installing

Swiftlet is a Swift and Metal runtime that runs large Qwen Mixture-of-Experts
models on ordinary Apple hardware. It keeps only the small dense core of the
model in memory and streams the routed expert weights from the SSD as they are
needed. That is why an 80B model whose weights are 42 GB on disk runs in about
4.3 GB of RAM.

There are four ways to use it once it is built:

1. The command line, with `swiftlet chat` and `swiftlet generate`.
2. A local OpenAI-compatible HTTP server, `swiftlet-server`, which any tool that
   speaks the OpenAI API can point at.
3. As a Swift package inside your own macOS or iOS app.
4. On iPhone, through the Priv AI app or by building
   [leonickson1/localLLM](https://github.com/leonickson1/localLLM) yourself.

This guide covers 1 and 2.

## Before you start

You need:

- An **Apple Silicon Mac** (M1 or later). Intel Macs are not supported.
- **macOS 14 or later.**
- **Free SSD space** for the model container. 18 GB for the 4-bit 35B, 34 GB for
  the 8-bit 35B, 42 GB for the 4-bit 80B. This is on top of the roughly 40 GB
  Xcode itself takes.
- **Time.** Xcode is a large download, the first build takes a few minutes, and
  the model download takes anywhere from 15 minutes to a couple of hours
  depending on your connection.

You do not need a Metal toolchain, a Python environment, or any model conversion
tooling. Shaders are compiled at runtime and the containers are prebuilt.

---

## 1. Install the Swift toolchain

Swiftlet is a Swift package, so you need a Swift compiler. The cleanest way to
get one that also runs the test suite without extra flags is the full Xcode.

Install **Xcode 16 or later** from the Mac App Store. Wait until the App Store
reports it as installed and `/Applications/Xcode.app` exists.

Xcode being present is not enough. The command line still points at whatever
developer tools were there before, so you have to select it, accept the license,
and let it finish its first-launch component install. You can do all three from
the terminal without ever opening the app:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch
```

Each of these asks for your password. The last one can take a minute or two.

Now confirm it worked:

```bash
swift --version
xcrun --find swift
```

**You should see** a Swift 6.x version string, and a path that goes through
`Xcode.app/Contents/Developer`. If the path still says
`/Library/Developer/CommandLineTools`, the `xcode-select` line did not take, so
run it again and recheck.

> **A note on Command Line Tools.** You do not need to install the standalone
> Command Line Tools package separately. Full Xcode contains everything,
> including git. If you only have CLT 26.x installed instead of Xcode, `swift
> build` still works, but `swift test` fails with `no such module 'Testing'`
> until you pass the extra `-F` and `-rpath` flags documented in the README.
> Installing Xcode avoids that entirely, which is why this guide recommends it.

---

## 2. Clone and build

```bash
cd ~
git clone https://github.com/leonickson1/Swiftlet.git
cd Swiftlet
swift build -c release
```

The `-c release` matters. A debug build runs the inference loop far slower and
will give you misleading performance numbers.

**You should see** a long stream of `Compiling ...` lines followed by
`Build complete!`. The first build pulls dependencies and takes a few minutes.
Later builds are much faster.

Check that the three binaries exist:

```bash
ls -lh .build/release/swiftlet .build/release/swiftlet-repack .build/release/swiftlet-server
```

**You should see** three files. Briefly, they are:

| Binary | What it does |
|---|---|
| `swiftlet` | The CLI. `chat` and `generate` subcommands. |
| `swiftlet-repack` | Downloads and builds `.qpack` model containers. |
| `swiftlet-server` | The OpenAI-compatible HTTP server. |

### Optional: run the test suite

If you are here to verify the build rather than just use it, run the tests now.
They are fast and they exercise the kernels against reference fixtures.

```bash
swift test
```

**You should see** roughly 175 tests across 30 suites, all passing, in a couple
of seconds. This runs entirely on fixtures, so you do not need a model
downloaded yet.

---

## 3. Download a model container

Models are distributed as prebuilt `.qpack` containers. You do not convert
anything yourself. Pick one tier:

| Model | Disk | Peak RAM | Decode speed (M5) | Use it when |
|---|---|---|---|---|
| 35B 4-bit | 18 GB | 2.6 GB | 7 to 11 tok/s | Default. Best speed and smallest footprint. |
| 35B 8-bit | 34 GB | 7.6 GB | 3.5 to 4 tok/s | Best writing quality. Removes repetition artifacts the 4-bit build can show on long tasks. |
| 80B 4-bit | 42 GB | 4.3 GB | 4.5 to 5 tok/s | Largest model. Note RAM is barely higher than the 35B, because only about 3B parameters are active per token either way. |

If you are not sure, take the **35B 4-bit**.

Make somewhere to put it:

```bash
mkdir -p ~/models
```

### Option A: the R2 mirror (recommended)

This is usually much faster, because Hugging Face throttles anonymous downloads
progressively the longer they run.

```bash
# 35B 4-bit
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-qpack \
  --output ~/models/qwen3.6-35b.qpack

# 35B 8-bit
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-8bit-qpack \
  --output ~/models/qwen3.6-35b-8bit.qpack

# 80B 4-bit
.build/release/swiftlet-repack \
  --from-url https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3-next-80b-qpack \
  --output ~/models/qwen3-next-80b.qpack
```

### Option B: Hugging Face

The same containers, if you prefer to pull from the source of record or the
mirror is unavailable.

```bash
# 35B 4-bit
.build/release/swiftlet-repack \
  --from-hf Leonickson/Qwen3.6-35B-A3B-qpack \
  --output ~/models/qwen3.6-35b.qpack

# 35B 8-bit
.build/release/swiftlet-repack \
  --from-hf Leonickson/Qwen3.6-35B-A3B-8bit-qpack \
  --output ~/models/qwen3.6-35b-8bit.qpack

# 80B 4-bit
.build/release/swiftlet-repack \
  --from-hf Leonickson/Qwen3-Next-80B-A3B-qpack \
  --output ~/models/qwen3-next-80b.qpack
```

**You should see** a live progress readout as bytes are downloaded and routed
straight into their final positions in the container. Nothing is staged and then
copied, so the download is the install.

**If it stalls or you cancel it**, rerun the exact same command. Both sources are
resumable and it picks up where it left off. This is worth knowing for the 80B,
where a 42 GB download over a flaky connection may need a couple of restarts.

Confirm what landed:

```bash
du -sh ~/models/*.qpack
```

**You should see** a size close to the disk figure in the table above.

### Optional: verify the download

If you want to be certain the container matches the published hashes rather than
trusting the transfer, there is a checker script:

```bash
# against the R2 mirror
python3 scripts/verify_container.py \
  https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-qpack \
  ~/models/qwen3.6-35b.qpack

# or against Hugging Face
python3 scripts/verify_container.py \
  Leonickson/Qwen3.6-35B-A3B-qpack \
  ~/models/qwen3.6-35b.qpack
```

This reads every file and compares SHA-256 against the source manifest, so it
takes a while on a large container.

---

## 4. Chat from the terminal

The simplest way to confirm everything works. Pass the container path, then one
or more turns as separate arguments.

```bash
.build/release/swiftlet chat ~/models/qwen3.6-35b.qpack \
  "What is the capital of Spain?" "And what about France?"
```

**You should see** tokens stream out as they are generated. The second turn only
prefills the new text, because conversation state is kept between turns, so
follow-ups start answering much faster than the first one.

The 80B works identically, just point at its container:

```bash
.build/release/swiftlet chat ~/models/qwen3-next-80b.qpack \
  "Explain MoE routing simply."
```

You can set a system prompt, allow longer answers, and size the expert cache:

```bash
.build/release/swiftlet chat ~/models/qwen3.6-35b.qpack \
  --system "You are terse." \
  --max-new 512 \
  --cache-gb 8 \
  "Summarize expert streaming."
```

For a single prompt with performance numbers at the end, use `generate` instead:

```bash
.build/release/swiftlet generate ~/models/qwen3.6-35b.qpack \
  --gpu --chat --max-new 256 \
  --prompt "Explain expert streaming in one paragraph."
```

**You should see** the answer followed by timing stats, including tokens per
second. Note that `generate` defaults to `--max-new 32`, which truncates almost
any real answer, so raise it as shown. `--gpu` selects the Metal runtime and
`--chat` applies the model's chat template.

---

## 5. Run the OpenAI-compatible server

This is how you connect Swiftlet to everything else. Open a terminal tab and
leave this running:

```bash
.build/release/swiftlet-server \
  --model ~/models/qwen3.6-35b.qpack \
  --port 8080 \
  --cache-gb 2
```

Swap the `--model` path for the 8-bit 35B or the 80B container as you like.

**You should see**
`swiftlet-server listening on http://127.0.0.1:8080/v1 (model: ...)`.

Two things to know about this server:

- It binds to **loopback only** (`127.0.0.1`). It is not reachable from other
  machines on your network, by design.
- It handles **one request at a time**, because it holds shared conversation
  state. It is a personal endpoint, not a multi-user gateway.

Leave that tab alone and open a second tab for everything below.

---

## 6. Connect clients

### curl

Check which model id the server reports:

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

Send a chat completion. The `model` field is not enforced, so any string works:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "swiftlet",
    "messages": [{"role": "user", "content": "One sentence about Metal."}]
  }' | python3 -m json.tool
```

Stream the response instead, so tokens arrive as they are produced:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "swiftlet",
    "stream": true,
    "messages": [{"role": "user", "content": "Count 1 to 5."}]
  }'
```

**You should see** a sequence of `data:` lines. The server also emits SSE comment
heartbeats during long prefills so idle connections are not dropped, and standard
parsers ignore those.

### Python

```bash
pip3 install openai
```

```bash
python3 - <<'PY'
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")
reply = client.chat.completions.create(
    model="swiftlet",
    messages=[{"role": "user", "content": "Say hi in five words."}],
)
print(reply.choices[0].message.content)
PY
```

The API key is required by the client library but ignored by the server, so any
non-empty string is fine.

### Any OpenAI-compatible CLI

Tools like aider and the OpenAI CLI read these two variables:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="sk-local"
```

### Open WebUI, a browser chat interface

A quick note on what Ollama can and cannot do here: you **cannot** load a
`.qpack` container into Ollama, because Ollama runs its own GGUF models through
llama.cpp. What Swiftlet gives you is the same OpenAI API that Ollama also
exposes, so any front end built for an OpenAI-compatible endpoint works with
Swiftlet. Open WebUI is the common one.

Docker is not installed on macOS by default. Install it once, then the
`docker run` pulls Open WebUI automatically on first use:

```bash
brew install --cask docker
open -a Docker
```

Wait for the whale icon in the menu bar to stop animating, then confirm the
daemon is up:

```bash
docker --version && docker info --format '{{.ServerVersion}}'
```

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8080/v1" \
  -e OPENAI_API_KEY="sk-local" \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

open http://localhost:3000
```

Note the container reaches your Mac through `host.docker.internal`, not
`127.0.0.1`, because `127.0.0.1` inside a container means the container itself.

If you would rather not install Docker at all:

```bash
pip3 install open-webui
export OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="sk-local"
open-webui serve --port 3000
```

The explicit `--port 3000` matters, because Open WebUI also defaults to 8080 and
`swiftlet-server` is already sitting there.

### Editor and agent tools

For Continue, Cursor, or anything else with a custom OpenAI endpoint setting:

- Provider: OpenAI-compatible
- Base URL: `http://127.0.0.1:8080/v1`
- API key: `sk-local`
- Model: `swiftlet`

---

## Flag reference

Defaults differ between the CLI and the server, which surprises people:

| Flag | `swiftlet chat` | `swiftlet generate` | `swiftlet-server` |
|---|---|---|---|
| `--max-new` | 256 | 32 | via `max_tokens`, default 512 |
| `--cache-gb` | 8 | 8 | 2 |
| `--port` | n/a | n/a | 8080 |

`--cache-gb` sets the expert cache budget. Treat it as a memory knob first and a
throughput knob second. On a 16 GB Mac, raising it takes RAM directly from the OS
page cache, so the gain is modest, roughly 8 to 14 percent, and the low setting
is usually the better trade. On a 64 GB Mac there is no such contention and
sizing the cache to hold the whole pool is worth about 11 percent. The detailed
measurements are in `docs/BENCH_QWEN36_35B_M1_MINI.md` and
`docs/BENCH_QWEN36_35B_M4MAX.md`.

---

## Troubleshooting

**`swift --version` shows the wrong path, or `no such module 'Testing'`**
Your tools are still pointed at the standalone Command Line Tools. Rerun
`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` and check with
`xcrun --find swift`.

**`git` pops up an "install command line developer tools" dialog**
This happens on a fresh Mac before Xcode is selected. Once step 1 is done, git
comes from Xcode and the dialog stops. If you are blocked before that,
`xcode-select --install` unblocks you, but you will not need it afterwards.

**The download stopped partway**
Rerun the identical `swiftlet-repack` command. It resumes rather than starting
over.

**Generation is slow to produce the first token on a long prompt**
That is prefill. Long prompts currently process at roughly decode speed, so a
large system prompt costs real time before the first token appears. This is the
top item on the project roadmap.

**The model gets facts wrong**
Expected, and worth setting expectations on. Only about 3B parameters are active
per token, so these models write and converse like large models but recall facts
like small ones.

**Port 8080 is already in use**
Either something else has it, or you started a second `swiftlet-server`. Pass a
different `--port` and update the client base URL to match.

---

## Where to go next

- `README.md` for the architecture, the correctness story, and the roadmap.
- `docs/BENCH_QWEN36_35B_M1_MINI.md` and `docs/BENCH_QWEN36_35B_M4MAX.md` for
  measured performance across hardware and cache settings.
- [leonickson1/localLLM](https://github.com/leonickson1/localLLM) to run the same
  engine on an iPhone. Clone this repo next to it as `swiftlet` first.
