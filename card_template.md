---
pretty_name: "{{DATASET_TITLE}}"
language:
- en
license: mit
annotations_creators:
- machine-generated
language_creators:
- found
multilinguality:
- monolingual
source_datasets:
- extended
task_categories:
- text-classification
tags:
- semantic-filters
- semceb
- amazon-reviews-2023
size_categories:
- {{SIZE_CATEGORY}}
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: "{{MERGED_FILENAME}}"
---

# {{DATASET_TITLE}}

## Dataset summary

This dataset extends the SemCEB {{DATASET_SUBJECT_LOWER}} table with {{N_QUERIES}}
nullable Boolean semantic-filter columns. It contains {{N_ROWS}} source rows and
{{N_JUDGMENTS}} query-row judgments generated from SemCEB's published filter
predicates. Every predicate was applied to every row; the source rows were not sampled.

The labels are **silver-standard LLM judgments**.

- **Model:** {{MODEL_SUMMARY}}
- **Base Dataset Source:** [SemCEB](https://github.com/utndatasystems/SemCEB)

## Usage

### Load as a Hugging Face Dataset

The default `train` split loads the complete dataset (`{{MERGED_FILENAME}}`), which includes all source columns and generated `label_qN` columns:

```python
from datasets import load_dataset

dataset = load_dataset("{{REPO_ID}}", split="train")
```

The split name `train` is a standard Hugging Face loading convention; this release does not define fixed training, validation, or test partitions. Each label evaluates to `true`, `false`, or `null` (if the model output could not be parsed).

### Download the Parquet file

To download the standalone Parquet file:

```bash
hf download {{REPO_ID}} {{MERGED_FILENAME}} \
  --repo-type dataset \
  --local-dir .
```

Python equivalent:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{{REPO_ID}}",
    repo_type="dataset",
    filename="{{MERGED_FILENAME}}",
    local_dir=".",
)
```

## Files

- `{{MERGED_FILENAME}}`: the recommended file, containing source data and every label.
- `{{BASE_FILENAME}}`: the source table without generated labels.
- `{{LABELS_DIRNAME}}/manifest.json`: labeling metadata and source-file identity.
- `{{LABELS_DIRNAME}}/label_qN.parquet`: one positionally aligned audit file per query.

## Label columns

{{QUERY_TABLE}}

- **Null labels:** number of unlabeled samples out of {{N_ROWS}} total samples.
- **Selectivity:** number of `true` labels divided by all samples.

## Data fields

{{DATA_FIELDS_TABLE}}

## Source and labeling method

The source table and filter predicates are sourced from [SemCEB](https://github.com/utndatasystems/SemCEB), which builds on [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

This release adds one positionally aligned label column per selected SemCEB predicate.
The exact prompt, model, effective generation settings, source-file hash, timestamps,
and per-query counts are stored in `{{LABELS_DIRNAME}}/manifest.json`.

### Model & Generation Configuration

Labeling retained [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)'s default
thinking mode and used Qwen's recommended thinking-mode sampling settings:
- **Sampling:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`
- **Penalties:** `presence_penalty=0.0`, `repetition_penalty=1.0`

The experiment additionally fixed
- **Seed:** `42`

## Quality and limitations

The labels inherit inherent biases and errors from both the underlying source data and the labeling model. These silver-standard LLM judgments should not be treated as independently verified ground truth.

Unparseable model outputs are preserved as `null`s to maintain strict 1:1 positional row alignment with the base dataset (this release contains **{{N_UNPARSEABLE}}** unparseable query-row outputs).

## License

- **Generated Labels & Code:** Released under the **MIT License**.
- **Base Dataset & Text:** Subject to the academic/non-commercial research terms of [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

## Citation

Please cite this dataset as:

```bibtex
@misc{{{CITATION_KEY}},
  title        = {{{DATASET_TITLE}}},
  author       = {{{CITATION_AUTHOR}}},
  year         = {{{CITATION_YEAR}}},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/{{REPO_ID}}}}
}
```
