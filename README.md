# ground-truth-labeler

A small standalone tool for generating complete, checkpointed boolean labels for a
Parquet dataset and publishing the merged result as a Hugging Face dataset.

The base Parquet file is always read-only. Labeling output, response recovery state,
merged data, and publish staging files are written elsewhere.

## Installation

Install the pinned project dependencies:

```bash
uv sync
```

Or install them directly from `pyproject.toml`:

```bash
pip install .
```

Both scripts load a local `.env` file. For the API backend, set the credentials required
by the LiteLLM model string. Publishing uses `HF_TOKEN`. For vLLM model downloads,
Hugging Face also reads `HF_TOKEN` and honors `HF_HOME` when it is set.

## Configuration

Both scripts read `config.toml`. The query manifest and base dataset are defined once in
`[shared]`. Given `datasets/amazon_products.parquet`, the scripts derive:

```text
datasets/amazon_products/labels/    # checkpoints, manifest, response cache
datasets/amazon_products/publish/   # complete Hugging Face upload tree
```

To process another dataset, create another query manifest and TOML file, then select it
with `--config`:

```bash
python label.py --config config-reviews.toml
python publish.py --config config-reviews.toml
```

An empty `queries` list means all available queries.

Generation parameters are also configured in TOML. Every key is optional and only
uncommented values are passed directly. One deliberate vLLM exception is recorded in
the effective fingerprint: when `max_tokens` is omitted, the labeler passes
`max_tokens=None` to avoid vLLM 0.25.1's 16-token output default. Generation is then
bounded by the engine's `max_model_len`, which covers input plus output.

For local inference, `[label.vllm_engine_parameters]` is passed directly to `vllm.LLM`.
Set `tensor_parallel_size` manually to the fewest GPUs required to fit the model; vLLM
does not automatically choose that number. GPU visibility remains controlled by the
runtime environment, such as `CUDA_VISIBLE_DEVICES`. Omitted engine arguments use vLLM
defaults, and configured engine arguments are also included in the query fingerprint.

## Query manifest

Use one JSONL manifest per base dataset. The base dataset belongs in `config.toml`, so a
query requires only `id` and `filter`:

```json
{"id": 1, "type": "Simple Base-Table Predicate", "category": ["equality"], "filter": "The review mentions shipping trouble: {review_text}"}
```

`system_prompt` is optional per query. When absent, `[label].system_prompt` from TOML is
used. When present, the query value overrides that default. Other JSON fields are
ignored by labeling. The optional reporting-only fields `type` and `category` are shown
in the published dataset card; `category`, when present, is a list of strings. Adding,
removing, or changing either field does not affect prompts, fingerprints, caching,
resume behavior, labels, or merged Parquet columns. All alignment and merging are
permanently based on the base file's zero-based row order.

Templates use Python `str.format` fields. Only referenced Parquet columns are loaded.
A rendering error aborts the run.

## Labeling

Edit `config.toml`, then run:

```bash
python label.py
```

The only other command-line option is the one-off `--force` action:

```bash
python label.py --force
```

`--force` discards and regenerates selected query results only when the current
configuration fingerprint matches their existing fingerprint. A changed model, prompt,
backend, generation parameters, vLLM engine parameters, or base dataset is refused while
fragments exist. To make a deliberate methodology change, delete that query's
`query_id=N/` directory and run again.

Each completed checkpoint becomes an atomic Parquet fragment:

```text
datasets/amazon_products/labels/
├── manifest.json
├── .cache/                     # local recovery state; never published
└── query_id=1/
    ├── part-....parquet
    └── part-....parquet
```

Re-running automatically skips positions already present in fragments. With caching
enabled, successful responses in an interrupted checkpoint are also reused. Once a
fragment is committed, its response-cache entries are removed.

The parser ignores any surrounding reasoning and selects the last complete
`<answer>...</answer>` block. After trimming and case-folding its contents, `true`, `1`,
and `yes` map to true; `false`, `0`, and `no` map to false. A missing or invalid final
tag becomes a null label. A response whose finish reason is `length` also becomes null,
even if an earlier complete answer tag exists. Rendering, missing-response,
model-call, output-count, storage, and schema failures abort the run.

API concurrency applies only to independent LiteLLM requests. The vLLM backend instead
passes up to `vllm_checkpoint_every` prompts to one synchronous `LLM.chat()` call, and
vLLM schedules that group internally. The next group is submitted after the current
group returns and its fragment is written atomically.

## Publishing

Edit `config.toml`, then run:

```bash
python publish.py
```

Before writing or uploading anything, publishing verifies the base file's full SHA-256
against every selected manifest entry and verifies that each query contains every
position from `0` through `len(base)-1` exactly once. Null labels count as completed
rows. Any mismatch refuses the entire publish.

The derived publish staging directory contains:

```text
datasets/amazon_products/publish/
├── README.md
├── amazon_products.parquet
├── amazon_products.merged.parquet
└── labels/
    ├── manifest.json
    └── query_id=1/*.parquet
```

The local `.cache/` directory is intentionally excluded. `publish.py` creates the Hub
dataset repository as private when needed, uploads the staging directory in one commit,
and prints its commit hash.

Use one Hugging Face dataset repository per base dataset. Pin the resulting Hub commit
and the corresponding `ground-truth-labeler` Git tag when consuming published data.
