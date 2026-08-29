# ground-truth-labeler

A small standalone tool for generating complete, checkpointed boolean labels for a
Parquet dataset and publishing the merged result as a Hugging Face dataset.

The base Parquet file is always read-only. Labeling output, response recovery state,
merged data, and publish staging files are written elsewhere.

## Public Datasets Generated (Aug 2026):

- [Abdullah-Taha/reviews-filtered-labels](https://huggingface.co/datasets/Abdullah-Taha/reviews-filtered-labels)
- [Abdullah-Taha/products-filtered-labels](https://huggingface.co/datasets/Abdullah-Taha/products-filtered-labels)


## Installation

Install the pinned project dependencies using uv (recommended):

```bash
uv sync
```

Or install them directly from `pyproject.toml`:

```bash
pip install .
```

For the API backend, set the credentials required by the LiteLLM model string. Publishing uses `HF_TOKEN`. For vLLM model downloads,
Hugging Face also reads `HF_TOKEN` if needed and  `HF_HOME` for .cache directory.

## Configuration

Both scripts read `config.toml`. The query manifest and base dataset are defined once in
`[shared]`. Given `datasets/amazon_products.parquet`, the scripts derive:

```text
datasets/amazon_products/labels/    # per-query checkpoints, manifests, caches, locks
datasets/amazon_products/publish/   # complete Hugging Face upload tree
```

To process another dataset, create another query manifest and TOML file, then select it
with `--config`:

```bash
python label.py --config config-reviews.toml
python publish.py --config config-reviews.toml
```

To select specific queries

```bash
python label.py --config config.toml --queries 0 2 3
python label.py --config config.toml --queries 4 5 6
```

## Query manifest

Use one JSONL manifest per base dataset. The base dataset belongs in `config.toml`, so a
query requires only `id` and `filter`:

```json
{"id": 1, "type": "Simple Base-Table Predicate", "category": ["equality"], "filter": "The review mentions shipping trouble: {review_text}"}
```

## Labeling

Edit `config.toml`, then run:

```bash
python label.py
```

The one-off `--force` action discards selected results:

```bash
python label.py --force
```

`--force` discards and regenerates selected query results only when the current
configuration fingerprint matches their existing fingerprint. A changed model, prompt,
backend, generation parameters, fingerprinted vLLM engine parameters, or base dataset is
refused while fragments exist. To make a deliberate methodology change, delete that query's
`query_id=N/` directory and run again.


Each completed checkpoint becomes an atomic Parquet fragment:

```text
datasets/amazon_products/labels/
├── .locks/                         # persistent files; OS locks release on process exit
└── query_id=1/
    ├── manifest.json               # contains exactly the Q1 entry
    ├── cache.sqlite3               # optional local recovery state; never published
    ├── part-....parquet
    ├── part-....parquet
    ├── deferred.parquet            # present only while positions remain pending
    └── finalized_deferred.parquet  # explicit capacity-null audit, when applicable
```


## Publishing

Set `[publish].title`, `citation_author`, and `citation_year` for the release. The
repository ID is inserted into both the loading example and the generated dataset
citation.

Edit `config.toml`, then run:

```bash
python publish.py --config config.toml
```

The derived publish staging directory contains:

```text
datasets/amazon_products/publish/
├── README.md
├── amazon_products.parquet
├── amazon_products_labeled.parquet
└── labels/
    ├── manifest.json
    └── label_q1.parquet
```

<!-- 
## Addittional (commented)

Generation parameters are also configured in TOML. Every key is optional and only
uncommented values are passed directly. The checked-in vLLM configuration uses
`temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`,
`presence_penalty = 0.0`, `repetition_penalty = 1.0`, and `seed = 42`. These settings
are part of the labeling fingerprint. One deliberate vLLM exception is also recorded
in the effective fingerprint: when `max_tokens` is omitted, the labeler passes
`max_tokens=None` to avoid vLLM 0.25.1's 16-token output default. Generation is then
bounded by the engine's `max_model_len`, which covers input plus output.

For local inference, `[label.vllm_engine_parameters]` is passed directly to `vllm.LLM`.
Set `tensor_parallel_size` manually to the fewest GPUs required to fit the model; vLLM
does not automatically choose that number. GPU visibility remains controlled by the
runtime environment, such as `CUDA_VISIBLE_DEVICES`. Omitted engine arguments use vLLM
defaults, and configured engine arguments are also included in the query fingerprint.
Four capacity/placement settings are operational exceptions and are excluded:
`tensor_parallel_size`, `max_model_len`, `max_num_seqs`, and
`gpu_memory_utilization`. They remain recorded in each manifest's embedded effective
configuration, but changing them does not block resume or change the cache namespace.
All other configured vLLM engine parameters remain fingerprinted.

`system_prompt` is optional per query. When absent, `[label].system_prompt` from TOML is
used. When present, the query value overrides that default. Other JSON fields are
ignored by labeling. The optional reporting-only fields `type` and `category` are shown
in the published dataset card; `category`, when present, is a list of strings. Adding,
removing, or changing either field does not affect prompts, fingerprints, caching,
resume behavior, labels, or merged Parquet columns. All alignment and merging are
permanently based on the base file's zero-based row order.

Every query owns its manifest, fragments, ledgers, and optional cache. Different jobs
can therefore process different queries without sharing a writable manifest. Before
touching a query, labeling takes a nonblocking OS lock in `.locks/`; a concurrent job
requesting the same query exits before initializing the model. The kernel releases the
lock when the job processes exit, including after termination, so rerunning resumes the
same query directory. The labels directory must be on a shared filesystem visible to
all participating jobs, and that filesystem must provide cross-node `flock` semantics.

Re-running automatically skips positions already present in fragments. With caching
enabled, completed vLLM outputs are parsed and committed to SQLite as the engine
finishes them, before the enclosing fragment is complete. Once an atomic fragment is
committed, its now-redundant cache entries are removed. The cache uses SQLite WAL mode
with `synchronous=NORMAL`: committed entries survive a killed Python job, though a
whole-machine crash can lose the most recent cache transactions.

The parser accepts only a terminal answer. When the model emits a completed thinking
block, text through the final `</think>` is ignored; when thinking is disabled, the
whole completion is treated as visible output. A final `<answer>...</answer>` block is
preferred, with an exact standalone final line such as `True`, `False`,
`Final answer: True`, or `Answer: False` accepted as a conservative fallback. After
trimming and case-folding, `true`, `1`, and `yes` map to true; `false`, `0`, and `no`
map to false. An unfinished thinking block, nested or incomplete answer markup,
nonterminal answer, or other ambiguous output becomes a completed null label.

For manual inspection, `label.py` sets `LABEL_PRINT_VLLM_SAMPLES=2` near the top of
the file. The first two newly generated, non-length vLLM requests in each Python
invocation print the exact rendered prompt and prompt token IDs returned by vLLM,
together with token count, generated completion, row position, and parsed label. Long
prompts, token lists, and responses use bounded head-and-tail previews. The log reuses
vLLM's completed request metadata and does not tokenize the prompt again. This
diagnostic does not enter query configuration, manifests, caches, or fingerprints.

For vLLM, a prompt rejected for exceeding the model context and any response whose
finish reason is `length` are deferred instead. Deferred positions are omitted from
normal fragments, recorded by row position in `deferred.parquet`, and retried
automatically on the next invocation because positional resume processes every missing
position. Increase `max_model_len` and rerun as many times as needed. Publishing remains
blocked while any positions are missing.

A fresh vLLM query first processes `canary_rows` evenly distributed positions. If more
than `canary_max_deferred_fraction` hit an input or generation length limit, the query
aborts without writing canary fragments. This catches configurations such as
`max_model_len = 20` without depending on consecutive dataset rows.

Setting `finalize_deferred_as_null = true` converts already-known deferred positions to
completed nulls without another model call. Newly encountered capacity-limited rows are
also finalized as null, but the fresh-query canary check still takes precedence. The
manifest records parser nulls and explicitly finalized capacity nulls separately. The
public dataset card reports the aggregate unparseable-output count without exposing
internal deferred-workflow details. Rendering, missing-response, model-call, storage,
schema, CUDA, and engine failures abort the run.

API concurrency applies only to independent LiteLLM requests. The vLLM backend enqueues
up to `vllm_checkpoint_every` prompts individually so a bad input can be isolated, then
lets vLLM schedule all accepted requests together with continuous batching. A small
engine-step loop observes completed outputs for incremental caching; it is the same
underlying mechanism used by vLLM's `wait_for_completion()`. 

`--queries 0 2 3` overrides the publication selection in `config.toml`, using the same
syntax as labeling.

Before writing or uploading anything, publishing verifies the base file's full SHA-256
against every selected manifest entry and verifies that each query contains every
position from `0` through `len(base)-1` exactly once. Null labels count as completed
rows. Any mismatch refuses the entire publish.-->
