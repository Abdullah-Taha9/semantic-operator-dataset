"""Validate, merge, document, and publish one completed labeled dataset.
uv run python publish.py --config confs/config-products.toml
uv run python publish.py --config confs/config-reviews.toml
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import DatasetCard

load_dotenv()


MERGED_ROW_GROUP_TARGET_BYTES = 128 * 1024 * 1024
MERGED_ROW_GROUP_MAX_BYTES = 300_000_000
MERGED_PARQUET_COMPRESSION = "zstd"
LABEL_IO_BATCH_ROWS = 100_000


def read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid label manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Label manifest {path} must contain a JSON object")
    return manifest


@contextmanager
def query_lock(labels_dir: Path, query_id: int):
    lock_dir = labels_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"query_id={query_id}.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Q{query_id}: labeling is active; refusing to publish a changing query"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def query_locks(labels_dir: Path, query_ids: list[int]):
    """Hold selected query locks in a stable order for one publication snapshot."""
    with ExitStack() as stack:
        for query_id in sorted(query_ids):
            stack.enter_context(query_lock(labels_dir, query_id))
        yield


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_query_ids(labels_dir: Path, query_ids: set[int] | None) -> list[int]:
    available: set[int] = set()
    for path in labels_dir.glob("query_id=*"):
        if path.is_dir():
            try:
                available.add(int(path.name.split("=", 1)[1]))
            except ValueError as error:
                raise ValueError(
                    f"Invalid query directory name: {path.name}"
                ) from error

    selected = available if query_ids is None else query_ids
    missing_dirs = selected - available
    if missing_dirs:
        raise ValueError(
            f"Missing fragment directories for queries: {sorted(missing_dirs)}"
        )
    if not selected:
        raise ValueError("No query fragment directories found")
    return sorted(selected)


def read_query_manifests(labels_dir: Path, query_ids: list[int]) -> dict:
    aggregate: dict[str, dict] = {}
    for query_id in query_ids:
        manifest_path = labels_dir / f"query_id={query_id}" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Q{query_id}: manifest not found: {manifest_path}")
        query_manifest = read_manifest(manifest_path)
        expected_key = str(query_id)
        if set(query_manifest) != {expected_key}:
            raise ValueError(
                f"Q{query_id}: {manifest_path} must contain exactly the key "
                f"{expected_key!r}"
            )
        entry = query_manifest[expected_key]
        if not isinstance(entry, dict):
            raise ValueError(f"Q{query_id}: manifest entry must be a JSON object")
        aggregate[expected_key] = entry
    return aggregate


def bitmap_is_set(bitmap: bytearray, position: int) -> bool:
    return bool(bitmap[position // 8] & (1 << (position % 8)))


def bitmap_set(bitmap: bytearray, position: int) -> None:
    bitmap[position // 8] |= 1 << (position % 8)


def read_complete_labels(query_dir: Path, n_rows: int) -> pa.BooleanArray:
    fragment_paths = sorted(query_dir.glob("part-*.parquet"))
    if not fragment_paths:
        raise ValueError(f"No fragments found in {query_dir}")

    bitmap_bytes = (n_rows + 7) // 8
    seen = bytearray(bitmap_bytes)
    validity = bytearray(bitmap_bytes)
    values = bytearray(bitmap_bytes)
    seen_count = 0
    valid_count = 0
    for fragment_path in fragment_paths:
        with pq.ParquetFile(fragment_path) as parquet:
            schema = parquet.schema_arrow
            if schema.names != ["row_position", "label"]:
                raise ValueError(
                    f"Unexpected schema in {fragment_path}; expected row_position and label only"
                )
            if not pa.types.is_integer(schema.field("row_position").type):
                raise ValueError(f"row_position must be an integer in {fragment_path}")
            if not pa.types.is_boolean(schema.field("label").type):
                raise ValueError(f"label must be boolean in {fragment_path}")

            for batch in parquet.iter_batches(batch_size=LABEL_IO_BATCH_ROWS):
                positions = batch.column(0).to_pylist()
                labels = batch.column(1).to_pylist()
                for position, label in zip(positions, labels):
                    if position is None:
                        raise ValueError(f"Null row positions found in {query_dir}")
                    if position < 0 or position >= n_rows:
                        raise ValueError(
                            f"Incomplete row coverage in {query_dir}: "
                            f"missing=[], out_of_range={[position]}"
                        )
                    if bitmap_is_set(seen, position):
                        raise ValueError(f"Duplicate row positions found in {query_dir}")
                    bitmap_set(seen, position)
                    seen_count += 1
                    if label is not None:
                        bitmap_set(validity, position)
                        valid_count += 1
                        if label:
                            bitmap_set(values, position)

    if seen_count != n_rows:
        missing: list[int] = []
        for position in range(n_rows):
            if not bitmap_is_set(seen, position):
                missing.append(position)
                if len(missing) == 5:
                    break
        raise ValueError(
            f"Incomplete row coverage in {query_dir}: "
            f"missing={missing}, out_of_range=[]"
        )

    return pa.Array.from_buffers(
        pa.bool_(),
        n_rows,
        [pa.py_buffer(bytes(validity)), pa.py_buffer(bytes(values))],
        null_count=n_rows - valid_count,
    )


def size_category(n_rows: int) -> str:
    if n_rows < 1_000:
        return "n<1K"
    if n_rows < 10_000:
        return "1K<n<10K"
    if n_rows < 100_000:
        return "10K<n<100K"
    if n_rows < 1_000_000:
        return "100K<n<1M"
    if n_rows < 10_000_000:
        return "1M<n<10M"
    return "n>10M"


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def read_query_reporting(path: Path) -> dict[int, dict]:
    reporting: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                query_id = raw["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid query manifest entry on line {line_number}: {error}"
                ) from error
            if not isinstance(query_id, int) or isinstance(query_id, bool):
                raise ValueError(
                    f"Query manifest line {line_number}: id must be an integer"
                )
            if query_id in reporting:
                raise ValueError(
                    f"Query manifest line {line_number}: duplicate id {query_id}"
                )

            query_type = raw.get("type")
            categories = raw.get("category")
            if query_type is not None and not isinstance(query_type, str):
                raise ValueError(
                    f"Query manifest line {line_number}: type must be a string when present"
                )
            if categories is not None and (
                not isinstance(categories, list)
                or any(not isinstance(category, str) for category in categories)
            ):
                raise ValueError(
                    f"Query manifest line {line_number}: category must be a list of strings "
                    "when present"
                )
            reporting[query_id] = {
                "type": query_type,
                "category": categories,
            }
    return reporting


def format_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    if numerator == 0:
        return "0"
    if numerator == denominator:
        return "1"
    rounded = f"{numerator / denominator:.4f}"
    if rounded == "0.0000":
        return "<0.0001"
    if rounded == "1.0000":
        return ">0.9999"
    return rounded.rstrip("0").rstrip(".")


def compute_label_statistics(
    label_columns: dict[int, list[bool | None] | pa.BooleanArray], n_rows: int
) -> dict[int, dict[str, int]]:
    statistics: dict[int, dict[str, int]] = {}
    for query_id, labels in label_columns.items():
        if len(labels) != n_rows:
            raise ValueError(
                f"Q{query_id}: label count {len(labels)} does not match "
                f"dataset row count {n_rows}"
            )
        if isinstance(labels, pa.BooleanArray):
            n_true = int(pc.sum(labels).as_py() or 0)
            n_null = labels.null_count
            n_false = n_rows - n_true - n_null
        else:
            n_true = sum(label is True for label in labels)
            n_false = sum(label is False for label in labels)
            n_null = sum(label is None for label in labels)
        statistics[query_id] = {
            "n_true": n_true,
            "n_false": n_false,
            "n_null": n_null,
            "n_rows": n_rows,
        }
    return statistics


def query_table(
    manifest: dict,
    query_ids: list[int],
    label_statistics: dict[int, dict[str, int]],
    query_reporting: dict[int, dict] | None = None,
) -> str:
    query_reporting = query_reporting or {}
    lines = [
        "| Column | Type | Category | Selectivity | Null labels | Query |",
        "|---|---|---|---:|---:|---|",
    ]
    for query_id in query_ids:
        entry = manifest[str(query_id)]
        config = entry.get("config", {})
        reporting = query_reporting.get(query_id, {})
        statistics = label_statistics[query_id]
        query_type = reporting.get("type") or "—"
        categories = reporting.get("category")
        category_text = ", ".join(categories) if categories else "—"
        lines.append(
            "| `{column}` | {query_type} | {category} | {selectivity} | "
            "{n_null:,} | {query} |".format(
                column=f"label_q{query_id}",
                query_type=markdown_cell(query_type),
                category=markdown_cell(category_text),
                selectivity=format_fraction(
                    statistics["n_true"], statistics["n_rows"]
                ),
                n_null=statistics["n_null"],
                query=markdown_cell(config.get("filter", "")),
            )
        )
    return "\n".join(lines)


def data_fields_table(schema: pa.Schema) -> str:
    lines = [
        "| Column | Type |",
        "|---|---|",
    ]
    for field in schema:
        lines.append(
            f"| `{markdown_cell(field.name)}` | `{markdown_cell(field.type)}` |"
        )
    return "\n".join(lines)


def card_run_summary(manifest: dict, query_ids: list[int]) -> dict:
    models = sorted(
        {
            str(manifest[str(query_id)].get("config", {}).get("model"))
            for query_id in query_ids
            if manifest[str(query_id)].get("config", {}).get("model")
        }
    )
    dates = sorted(
        {
            timestamp[:10]
            for query_id in query_ids
            if isinstance(
                timestamp := (
                    manifest[str(query_id)].get("completed_at")
                    or manifest[str(query_id)].get("started_at")
                ),
                str,
            )
            and len(timestamp) >= 10
        }
    )
    if not models:
        model_summary = "unknown"
    else:
        model_summary = ", ".join(models)
    if not dates:
        generation_summary = "unknown"
    elif len(dates) == 1:
        generation_summary = dates[0]
    else:
        generation_summary = f"{dates[0]} to {dates[-1]}"
    return {
        "model": model_summary,
        "generation": generation_summary,
        "n_unparseable": sum(
            int(manifest[str(query_id)].get("n_unparseable", 0))
            for query_id in query_ids
        ),
    }


def labeled_filename(dataset_path: Path) -> str:
    stem = dataset_path.stem.removesuffix("_with_embeddings")
    return f"{stem}_labeled.parquet"


def dataset_subject(base_filename: str) -> str:
    stem = Path(base_filename).stem
    for suffix in ("_with_embeddings", "_filtered"):
        stem = stem.removesuffix(suffix)
    return stem.replace("_", " ").strip().title() or "Dataset"


def citation_key(author: str, year: int, subject: str) -> str:
    surname = author.split(",", 1)[0] if "," in author else author.rsplit(" ", 1)[-1]
    raw = f"{surname}{year}SemCEB{subject}Labels"
    return "".join(character for character in raw.casefold() if character.isalnum())


def validate_card_contents(
    rendered: str,
    merged_filename: str,
    merged_schema: pa.Schema,
) -> None:
    try:
        metadata = DatasetCard(rendered).data.to_dict()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Generated dataset card metadata is invalid: {error}") from error
    expected_configs = [
        {
            "config_name": "default",
            "default": True,
            "data_files": [{"split": "train", "path": merged_filename}],
        }
    ]
    if metadata.get("configs") != expected_configs:
        raise ValueError(
            "Generated dataset card must configure only the merged Parquet as train"
        )
    if data_fields_table(merged_schema) not in rendered:
        raise ValueError("Generated dataset card is missing the complete data-fields table")


def render_card(
    template_path: Path,
    manifest: dict,
    query_ids: list[int],
    n_rows: int,
    base_filename: str,
    merged_filename: str,
    labels_dirname: str,
    merged_schema: pa.Schema,
    label_columns: dict[int, list[bool | None] | pa.BooleanArray],
    query_reporting: dict[int, dict] | None = None,
    dataset_title: str | None = None,
    repo_id: str | None = None,
    citation_author: str = "Al-Labani, Abdullah",
    citation_year: int = 2026,
) -> str:
    raw_template = template_path.read_text(encoding="utf-8")
    summary = card_run_summary(manifest, query_ids)
    label_statistics = compute_label_statistics(label_columns, n_rows)
    subject = dataset_subject(base_filename)
    title = dataset_title or f"Semantic Filter Labels for SemCEB {subject}"
    required_replacements = {
        "{{DATASET_TITLE}}": title,
        "{{DATASET_SUBJECT_LOWER}}": subject.lower(),
        "{{REPO_ID}}": repo_id or "<namespace>/<dataset-name>",
        "{{CITATION_AUTHOR}}": citation_author,
        "{{CITATION_YEAR}}": str(citation_year),
        "{{CITATION_KEY}}": citation_key(citation_author, citation_year, subject),
        "{{SIZE_CATEGORY}}": size_category(n_rows),
        "{{N_ROWS}}": f"{n_rows:,}",
        "{{N_QUERIES}}": str(len(query_ids)),
        "{{N_JUDGMENTS}}": f"{n_rows * len(query_ids):,}",
        "{{BASE_FILENAME}}": base_filename,
        "{{MERGED_FILENAME}}": merged_filename,
        "{{LABELS_DIRNAME}}": labels_dirname,
        "{{MODEL_SUMMARY}}": summary["model"],
        "{{N_UNPARSEABLE}}": str(summary["n_unparseable"]),
        "{{DATA_FIELDS_TABLE}}": data_fields_table(merged_schema),
        "{{QUERY_TABLE}}": query_table(
            manifest, query_ids, label_statistics, query_reporting
        ),
    }
    optional_replacements = {
        "{{GENERATION_SUMMARY}}": summary["generation"],
    }
    missing = [
        marker for marker in required_replacements if marker not in raw_template
    ]
    if missing:
        raise ValueError(f"Card template is missing required markers: {missing}")
    rendered = raw_template
    for replacements in (required_replacements, optional_replacements):
        for marker, value in replacements.items():
            rendered = rendered.replace(marker, value)
    unresolved = sorted(set(re.findall(r"\{\{[A-Z][A-Z_]*\}\}", rendered)))
    if unresolved:
        raise ValueError(f"Card template has unresolved markers: {unresolved}")
    validate_card_contents(rendered, merged_filename, merged_schema)
    return rendered


def merged_row_group_size(
    table: pa.Table,
    target_bytes: int = MERGED_ROW_GROUP_TARGET_BYTES,
) -> int:
    if target_bytes <= 0:
        raise ValueError("Merged Parquet row-group target must be positive")
    if table.num_rows == 0:
        return 1
    bytes_per_row = max(1, (table.nbytes + table.num_rows - 1) // table.num_rows)
    return max(1, min(table.num_rows, target_bytes // bytes_per_row))


def streamed_merged_schema(
    base_schema: pa.Schema,
    query_ids: list[int],
) -> pa.Schema:
    schema = base_schema
    for query_id in query_ids:
        column_name = f"label_q{query_id}"
        if column_name in schema.names:
            raise ValueError(f"Base dataset already contains {column_name}")
        schema = schema.append(pa.field(column_name, pa.bool_()))
    return schema


def streamed_merged_row_group_size(
    dataset_path: Path,
    label_columns: dict[int, pa.BooleanArray],
    target_bytes: int = MERGED_ROW_GROUP_TARGET_BYTES,
) -> int:
    if target_bytes <= 0:
        raise ValueError("Merged Parquet row-group target must be positive")
    with pq.ParquetFile(dataset_path) as parquet:
        n_rows = parquet.metadata.num_rows
        if n_rows == 0:
            return 1
        total_bytes = sum(
            parquet.metadata.row_group(index).total_byte_size
            for index in range(parquet.metadata.num_row_groups)
        ) + sum(labels.nbytes for labels in label_columns.values())
    bytes_per_row = max(1, (total_bytes + n_rows - 1) // n_rows)
    return max(1, min(n_rows, target_bytes // bytes_per_row))


def append_label_slices(
    base: pa.Table,
    label_columns: dict[int, pa.BooleanArray],
    row_position: int,
) -> pa.Table:
    merged = base
    for query_id, labels in label_columns.items():
        merged = merged.append_column(
            f"label_q{query_id}", labels.slice(row_position, base.num_rows)
        )
    return merged


def parquet_types_match(expected: pa.DataType, actual: pa.DataType) -> bool:
    if expected.equals(actual):
        return True
    list_kinds = (
        (pa.types.is_list, None),
        (pa.types.is_large_list, None),
        (pa.types.is_fixed_size_list, "list_size"),
    )
    for predicate, size_attribute in list_kinds:
        if predicate(expected) and predicate(actual):
            if size_attribute is not None and getattr(expected, size_attribute) != getattr(
                actual, size_attribute
            ):
                return False
            return (
                expected.value_field.nullable == actual.value_field.nullable
                and expected.value_field.metadata == actual.value_field.metadata
                and parquet_types_match(expected.value_type, actual.value_type)
            )
    return False


def parquet_metadata_matches(
    expected: dict[bytes, bytes] | None,
    actual: dict[bytes, bytes] | None,
) -> bool:
    """Treat both Arrow representations of absent metadata as equivalent."""
    return (expected or {}) == (actual or {})


def parquet_schemas_match(expected: pa.Schema, actual: pa.Schema) -> bool:
    if (
        not parquet_metadata_matches(expected.metadata, actual.metadata)
        or len(expected) != len(actual)
    ):
        return False
    return all(
        expected_field.name == actual_field.name
        and expected_field.nullable == actual_field.nullable
        and parquet_metadata_matches(
            expected_field.metadata, actual_field.metadata
        )
        and parquet_types_match(expected_field.type, actual_field.type)
        for expected_field, actual_field in zip(expected, actual)
    )


def validate_page_indexes(row_group, row_group_index: int) -> None:
    """Require the offset metadata used for page-level random access.

    A Parquet ColumnIndex is optional. PyArrow legitimately omits it when a
    non-null data page cannot provide bounded min/max statistics, including
    pages containing long strings. The OffsetIndex remains available and is
    the metadata required to locate pages by row position.
    """
    for column_index in range(row_group.num_columns):
        column = row_group.column(column_index)
        if not column.has_offset_index:
            raise ValueError(
                f"Merged Parquet row group {row_group_index}, physical column "
                f"{column_index} has no page index offset metadata"
            )


def null_masks_equal(expected: pa.Array, actual: pa.Array) -> bool:
    return pc.is_null(expected).equals(pc.is_null(actual))


def floating_arrays_equal(expected: pa.Array, actual: pa.Array) -> bool:
    if not null_masks_equal(expected, actual):
        return False
    same_value = pc.fill_null(pc.equal(expected, actual), False)
    both_nan = pc.fill_null(
        pc.and_(pc.is_nan(expected), pc.is_nan(actual)), False
    )
    matches = pc.or_(pc.or_(same_value, both_nan), pc.is_null(expected))
    result = pc.all(matches).as_py()
    return bool(result) if result is not None else len(expected) == 0


def arrow_arrays_equal(expected: pa.Array, actual: pa.Array) -> bool:
    if len(expected) != len(actual) or not expected.type.equals(actual.type):
        return False
    if expected.equals(actual):
        return True
    if isinstance(expected.type, pa.BaseExtensionType):
        return arrow_arrays_equal(expected.storage, actual.storage)
    if pa.types.is_floating(expected.type):
        return floating_arrays_equal(expected, actual)
    if pa.types.is_dictionary(expected.type):
        return arrow_arrays_equal(
            expected.dictionary_decode(), actual.dictionary_decode()
        )
    if pa.types.is_list(expected.type) or pa.types.is_large_list(expected.type):
        return (
            null_masks_equal(expected, actual)
            and expected.value_lengths().equals(actual.value_lengths())
            and arrow_arrays_equal(expected.flatten(), actual.flatten())
        )
    if pa.types.is_fixed_size_list(expected.type):
        return null_masks_equal(expected, actual) and arrow_arrays_equal(
            expected.flatten(), actual.flatten()
        )
    if pa.types.is_struct(expected.type):
        if not null_masks_equal(expected, actual):
            return False
        return all(
            arrow_arrays_equal(expected_child, actual_child)
            for expected_child, actual_child in zip(
                expected.flatten(), actual.flatten()
            )
        )
    if pa.types.is_map(expected.type):
        if not null_masks_equal(expected, actual):
            return False
        expected_offsets = expected.offsets.to_pylist()
        actual_offsets = actual.offsets.to_pylist()
        expected_start = expected_offsets[0]
        actual_start = actual_offsets[0]
        expected_normalized = [
            offset - expected_start for offset in expected_offsets
        ]
        actual_normalized = [offset - actual_start for offset in actual_offsets]
        if expected_normalized != actual_normalized:
            return False
        value_count = expected_normalized[-1]
        return arrow_arrays_equal(
            expected.keys.slice(expected_start, value_count),
            actual.keys.slice(actual_start, value_count),
        ) and arrow_arrays_equal(
            expected.items.slice(expected_start, value_count),
            actual.items.slice(actual_start, value_count),
        )
    return False


def arrow_tables_equal(expected: pa.Table, actual: pa.Table) -> bool:
    if (
        expected.num_rows != actual.num_rows
        or expected.num_columns != actual.num_columns
        or not expected.schema.equals(actual.schema, check_metadata=True)
    ):
        return False
    return all(
        arrow_arrays_equal(
            expected.column(index).combine_chunks(),
            actual.column(index).combine_chunks(),
        )
        for index in range(expected.num_columns)
    )


def validate_merged_parquet(
    path: Path,
    expected: pa.Table,
    row_group_size: int,
    max_row_group_bytes: int = MERGED_ROW_GROUP_MAX_BYTES,
) -> None:
    with pq.ParquetFile(path) as parquet:
        metadata = parquet.metadata
        if metadata.num_rows != expected.num_rows:
            raise ValueError(
                f"Merged Parquet row count changed during serialization: "
                f"expected {expected.num_rows}, found {metadata.num_rows}"
            )
        if not parquet_schemas_match(expected.schema, parquet.schema_arrow):
            raise ValueError("Merged Parquet schema changed during serialization")

        position = 0
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            if row_group.num_rows > row_group_size:
                raise ValueError(
                    f"Merged Parquet row group {row_group_index} has "
                    f"{row_group.num_rows} rows; expected at most {row_group_size}"
                )
            if row_group.total_byte_size > max_row_group_bytes:
                raise ValueError(
                    f"Merged Parquet row group {row_group_index} spans "
                    f"{row_group.total_byte_size:,} bytes, exceeding the Viewer-safe "
                    f"limit of {max_row_group_bytes:,} bytes"
                )
            validate_page_indexes(row_group, row_group_index)

            actual = parquet.read_row_group(row_group_index)
            expected_group = expected.slice(position, row_group.num_rows)
            normalized_expected = expected_group.cast(actual.schema)
            if not arrow_tables_equal(normalized_expected, actual):
                raise ValueError(
                    f"Merged Parquet values or row order changed in row group "
                    f"{row_group_index}"
                )
            position += row_group.num_rows

        if position != expected.num_rows:
            raise ValueError(
                f"Merged Parquet row groups cover {position} rows; "
                f"expected {expected.num_rows}"
            )


def write_merged_parquet(
    table: pa.Table,
    destination: Path,
    target_bytes: int = MERGED_ROW_GROUP_TARGET_BYTES,
) -> int:
    row_group_size = merged_row_group_size(table, target_bytes)
    pq.write_table(
        table,
        destination,
        row_group_size=row_group_size,
        compression=MERGED_PARQUET_COMPRESSION,
        write_page_index=True,
    )
    validate_merged_parquet(destination, table, row_group_size)
    return row_group_size


def validate_streamed_merged_parquet(
    path: Path,
    dataset_path: Path,
    label_columns: dict[int, pa.BooleanArray],
    expected_schema: pa.Schema,
    row_group_size: int,
    max_row_group_bytes: int = MERGED_ROW_GROUP_MAX_BYTES,
) -> None:
    with (
        pq.ParquetFile(dataset_path) as base_parquet,
        pq.ParquetFile(path) as merged_parquet,
    ):
        metadata = merged_parquet.metadata
        expected_rows = base_parquet.metadata.num_rows
        if metadata.num_rows != expected_rows:
            raise ValueError(
                f"Merged Parquet row count changed during serialization: "
                f"expected {expected_rows}, found {metadata.num_rows}"
            )
        if not parquet_schemas_match(expected_schema, merged_parquet.schema_arrow):
            raise ValueError("Merged Parquet schema changed during serialization")

        base_batches = base_parquet.iter_batches(batch_size=row_group_size)
        position = 0
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            if row_group.num_rows > row_group_size:
                raise ValueError(
                    f"Merged Parquet row group {row_group_index} has "
                    f"{row_group.num_rows} rows; expected at most {row_group_size}"
                )
            if row_group.total_byte_size > max_row_group_bytes:
                raise ValueError(
                    f"Merged Parquet row group {row_group_index} spans "
                    f"{row_group.total_byte_size:,} bytes, exceeding the Viewer-safe "
                    f"limit of {max_row_group_bytes:,} bytes"
                )
            validate_page_indexes(row_group, row_group_index)

            try:
                base_batch = next(base_batches)
            except StopIteration as error:
                raise ValueError(
                    "Merged Parquet has more row groups than the streamed base dataset"
                ) from error
            if base_batch.num_rows != row_group.num_rows:
                raise ValueError(
                    f"Merged Parquet row group {row_group_index} has "
                    f"{row_group.num_rows} rows but the corresponding base batch has "
                    f"{base_batch.num_rows}"
                )

            expected = append_label_slices(
                pa.Table.from_batches([base_batch]), label_columns, position
            )
            actual = merged_parquet.read_row_group(row_group_index)
            normalized_expected = expected.cast(actual.schema)
            if not arrow_tables_equal(normalized_expected, actual):
                raise ValueError(
                    f"Merged Parquet values or row order changed in row group "
                    f"{row_group_index}"
                )
            position += row_group.num_rows

        try:
            next(base_batches)
        except StopIteration:
            pass
        else:
            raise ValueError(
                "Merged Parquet has fewer row groups than the streamed base dataset"
            )
        if position != expected_rows:
            raise ValueError(
                f"Merged Parquet row groups cover {position} rows; "
                f"expected {expected_rows}"
            )


def write_streamed_merged_parquet(
    dataset_path: Path,
    destination: Path,
    label_columns: dict[int, pa.BooleanArray],
    target_bytes: int = MERGED_ROW_GROUP_TARGET_BYTES,
) -> tuple[pa.Schema, int]:
    query_ids = list(label_columns)
    with pq.ParquetFile(dataset_path) as base_parquet:
        merged_schema = streamed_merged_schema(base_parquet.schema_arrow, query_ids)
        row_group_size = streamed_merged_row_group_size(
            dataset_path, label_columns, target_bytes
        )
        print(
            f"Writing {destination.name} in streamed batches of at most "
            f"{row_group_size:,} rows...",
            flush=True,
        )
        position = 0
        with pq.ParquetWriter(
            destination,
            merged_schema,
            compression=MERGED_PARQUET_COMPRESSION,
            write_page_index=True,
        ) as writer:
            for batch in base_parquet.iter_batches(batch_size=row_group_size):
                merged = append_label_slices(
                    pa.Table.from_batches([batch]), label_columns, position
                )
                writer.write_table(merged, row_group_size=batch.num_rows)
                position += batch.num_rows

        if position != base_parquet.metadata.num_rows:
            raise ValueError(
                f"Streamed base dataset covered {position} rows; expected "
                f"{base_parquet.metadata.num_rows}"
            )

    print(f"Validating {destination.name} row by row group...", flush=True)
    validate_streamed_merged_parquet(
        destination,
        dataset_path,
        label_columns,
        merged_schema,
        row_group_size,
    )
    return merged_schema, row_group_size


def write_published_fragment(
    destination: Path,
    labels: list[bool | None] | pa.BooleanArray,
) -> None:
    schema = pa.schema(
        [pa.field("row_position", pa.int64()), pa.field("label", pa.bool_())]
    )
    label_array = labels if isinstance(labels, pa.BooleanArray) else pa.array(
        labels, type=pa.bool_()
    )
    with pq.ParquetWriter(destination, schema) as writer:
        for start in range(0, len(label_array), LABEL_IO_BATCH_ROWS):
            count = min(LABEL_IO_BATCH_ROWS, len(label_array) - start)
            table = pa.Table.from_arrays(
                [
                    pa.array(range(start, start + count), type=pa.int64()),
                    label_array.slice(start, count),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=count)


def copy_label_audit(
    labels_dir: Path,
    destination: Path,
    manifest: dict,
    query_ids: list[int],
    label_columns: dict[int, list[bool | None] | pa.BooleanArray],
) -> None:
    destination.mkdir(parents=True)
    selected_manifest = {
        str(query_id): manifest[str(query_id)] for query_id in query_ids
    }
    (destination / "manifest.json").write_text(
        json.dumps(selected_manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    for query_id in query_ids:
        source_query_dir = labels_dir / f"query_id={query_id}"
        write_published_fragment(
            destination / f"label_q{query_id}.parquet",
            label_columns[query_id],
        )
        finalized_path = source_query_dir / "finalized_deferred.parquet"
        if finalized_path.is_file():
            shutil.copy2(
                finalized_path,
                destination / f"label_q{query_id}_finalized_deferred.parquet",
            )


def _prepare_publish_locked(
    dataset_path: Path,
    labels_dir: Path,
    publish_dir: Path,
    query_ids: set[int] | None,
    template_path: Path,
    query_manifest_path: Path | None = None,
    dataset_title: str | None = None,
    repo_id: str | None = None,
    citation_author: str = "Al-Labani, Abdullah",
    citation_year: int = 2026,
) -> dict:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Base dataset not found: {dataset_path}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    dataset_resolved = dataset_path.resolve()
    labels_resolved = labels_dir.resolve()
    publish_resolved = publish_dir.resolve()
    if (
        publish_resolved == Path(publish_resolved.anchor)
        or publish_resolved == Path.cwd().resolve()
    ):
        raise ValueError(f"Refusing unsafe publish destination: {publish_dir}")
    if publish_resolved in dataset_resolved.parents:
        raise ValueError("Publish destination cannot contain the pristine base dataset")
    if publish_resolved in labels_resolved.parents:
        raise ValueError(
            "Publish destination cannot contain the source labels directory"
        )
    selected = selected_query_ids(labels_dir, query_ids)
    manifest = read_query_manifests(labels_dir, selected)
    query_reporting = (
        read_query_reporting(query_manifest_path)
        if query_manifest_path is not None
        else {}
    )
    base_hash = file_sha256(dataset_path)
    with pq.ParquetFile(dataset_path) as base_parquet:
        n_rows = base_parquet.metadata.num_rows
        base_schema = base_parquet.schema_arrow
    merged_schema = streamed_merged_schema(base_schema, selected)
    print(
        f"Preparing {n_rows:,} rows with {len(selected)} label columns...",
        flush=True,
    )

    label_columns: dict[int, pa.BooleanArray] = {}
    for query_id in selected:
        entry = manifest[str(query_id)]
        query_dir = labels_dir / f"query_id={query_id}"
        if not entry.get("config_fingerprint"):
            raise ValueError(f"Q{query_id}: manifest has no trusted config fingerprint")
        if entry.get("n_deferred", 0) or (query_dir / "deferred.parquet").exists():
            raise ValueError(
                f"Q{query_id}: deferred rows remain; rerun labeling or explicitly finalize them"
            )
        if entry.get("n_rows") != n_rows:
            raise ValueError(
                f"Q{query_id}: manifest row count {entry.get('n_rows')} "
                f"does not match base row count {n_rows}"
            )
        labelled_hash = entry.get("config", {}).get("dataset", {}).get("sha256")
        if not labelled_hash:
            raise ValueError(f"Q{query_id}: manifest has no base dataset SHA-256")
        if labelled_hash != base_hash:
            raise ValueError(
                f"Q{query_id}: base dataset SHA-256 does not match the file used for labeling"
            )
        label_columns[query_id] = read_complete_labels(query_dir, n_rows)

    publish_parent = publish_dir.parent
    publish_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{publish_dir.name}-", dir=publish_parent)
    )
    base_filename = dataset_path.name
    merged_filename = labeled_filename(dataset_path)
    labels_dirname = labels_dir.name
    try:
        shutil.copy2(dataset_path, temporary / base_filename)
        if file_sha256(temporary / base_filename) != base_hash:
            raise ValueError("Base dataset copy changed during publish staging")
        written_schema, _ = write_streamed_merged_parquet(
            dataset_path,
            temporary / merged_filename,
            label_columns,
        )
        if not parquet_schemas_match(merged_schema, written_schema):
            raise ValueError("Merged Parquet schema changed before serialization")
        copy_label_audit(
            labels_dir,
            temporary / labels_dirname,
            manifest,
            selected,
            label_columns,
        )
        card = render_card(
            template_path,
            manifest,
            selected,
            n_rows,
            base_filename,
            merged_filename,
            labels_dirname,
            merged_schema,
            label_columns,
            query_reporting,
            dataset_title,
            repo_id,
            citation_author,
            citation_year,
        )
        (temporary / "README.md").write_text(card, encoding="utf-8")
        install_publish_directory(temporary, publish_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "publish_dir": publish_dir,
        "base_path": publish_dir / base_filename,
        "merged_path": publish_dir / merged_filename,
        "card_path": publish_dir / "README.md",
        "query_ids": selected,
    }


def prepare_publish(
    dataset_path: Path,
    labels_dir: Path,
    publish_dir: Path,
    query_ids: set[int] | None,
    template_path: Path,
    query_manifest_path: Path | None = None,
    dataset_title: str | None = None,
    repo_id: str | None = None,
    citation_author: str = "Al-Labani, Abdullah",
    citation_year: int = 2026,
) -> dict:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    selected = selected_query_ids(labels_dir, query_ids)
    with query_locks(labels_dir, selected):
        return _prepare_publish_locked(
            dataset_path,
            labels_dir,
            publish_dir,
            set(selected),
            template_path,
            query_manifest_path,
            dataset_title,
            repo_id,
            citation_author,
            citation_year,
        )


def install_publish_directory(temporary: Path, publish_dir: Path) -> None:
    old = publish_dir.with_name(f".{publish_dir.name}.old")
    if old.exists():
        if not old.is_dir():
            raise ValueError(f"Publish backup path is not a directory: {old}")
        if publish_dir.exists():
            shutil.rmtree(old)
        else:
            os.replace(old, publish_dir)

    if not publish_dir.exists():
        os.replace(temporary, publish_dir)
        return
    if not publish_dir.is_dir():
        raise ValueError(f"Publish destination is not a directory: {publish_dir}")

    os.replace(publish_dir, old)
    try:
        os.replace(temporary, publish_dir)
    except BaseException:
        if not publish_dir.exists() and old.exists():
            os.replace(old, publish_dir)
        raise
    if old.exists():
        shutil.rmtree(old)


def upload_to_hub(repo_id: str, publish_dir: Path) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(publish_dir),
        commit_message=f"Publish semantic filter labels {datetime.now(timezone.utc):%Y-%m-%d}",
    )
    commit_hash = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
    if not commit_hash:
        commit_hash = api.list_repo_commits(repo_id, repo_type="dataset")[0].commit_id
    return commit_hash


def load_publish_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    try:
        shared = config["shared"]
        section = config["publish"]
        configured_queries = section["queries"]
        dataset_path = Path(shared["dataset"])
        dataset_root = dataset_path.with_suffix("")
        return {
            "dataset_path": dataset_path,
            "labels_dir": dataset_root / "labels",
            "publish_dir": dataset_root / "publish",
            "repo_id": section["repo_id"],
            "dataset_title": section.get("title"),
            "citation_author": section.get("citation_author", "Al-Labani, Abdullah"),
            "citation_year": section.get("citation_year", 2026),
            "template_path": Path(section["card_template"]),
            "query_manifest_path": Path(shared["manifest"]),
            "query_ids": set(configured_queries) or None,
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid publish configuration in {path}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        metavar="ID",
        help="Publish these query IDs instead of the query selection in config.toml",
    )
    args = parser.parse_args()
    if args.queries is not None and len(set(args.queries)) != len(args.queries):
        parser.error("--queries must not contain duplicate IDs")
    settings = load_publish_config(args.config)
    if args.queries is not None:
        settings["query_ids"] = set(args.queries)
    repo_id = settings.pop("repo_id")
    artifacts = prepare_publish(repo_id=repo_id, **settings)
    commit_hash = upload_to_hub(repo_id, artifacts["publish_dir"])
    print(f"Published {repo_id} at commit {commit_hash}")


if __name__ == "__main__":
    main()
