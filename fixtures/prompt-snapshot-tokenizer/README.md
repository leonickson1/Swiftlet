# Qwen3.6 prompt snapshot tokenizer fixture

Apache 2.0, Alibaba's Qwen team. Source: the tokenizer metadata accompanying
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0), read from the
installed `Qwen3.6-35B-A3B-4bit-qpack` checkpoint on 2026-11-04.
The chat template also matches that upstream revision byte for byte.

`chat_template.jinja` and `tokenizer_config.json` are unchanged. The
`tokenizer.json` retains the original preprocessing/decoding configuration,
the first 256 vocabulary entries, and the 33 merges whose joined pieces are
substrings of `assistant`, `user`, `hi`, or one/two byte-level newlines (`Ċ`).
The resulting vocabulary has 289 entries, with the original token ids and
relative merge order. Only the four ChatML/think added tokens are retained.
This fixture is only for the `hi` transcript in the header test; it cannot
tokenize arbitrary conversation text.

Original files' SHA-256:

- `tokenizer.json`: `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`
- `tokenizer_config.json`: `e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182`
- `chat_template.jinja`: `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`
