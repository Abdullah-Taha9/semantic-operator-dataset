"""Validate, merge, document, and publish one completed labeled dataset."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()


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


def read_complete_labels(query_dir: Path, n_rows: int) -> list[bool | None]:
    fragment_paths = sorted(query_dir.glob("part-*.parquet"))
    if not fragment_paths:
        raise ValueError(f"No fragments found in {query_dir}")

    positions: list[int] = []
    labels: list[bool | None] = []
    for fragment_path in fragment_paths:
        table = pq.read_table(fragment_path)
        if table.column_names != ["row_position", "label"]:
            raise ValueError(
                f"Unexpected schema in {fragment_path}; expected row_position and label only"
            )
        if not pa.types.is_integer(table.schema.field("row_position").type):
            raise ValueError(f"row_position must be an integer in {fragment_path}")
        if not pa.types.is_boolean(table.schema.field("label").type):
            raise ValueError(f"label must be boolean in {fragment_path}")
        positions.extend(table.column("row_position").to_pylist())
        labels.extend(table.column("label").to_pylist())

    if any(position is None for position in positions):
        raise ValueError(f"Null row positions found in {query_dir}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"Duplicate row positions found in {query_dir}")
    expected = set(range(n_rows))
    actual = set(positions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Incomplete row coverage in {query_dir}: "
            f"missing={missing[:5]}, out_of_range={extra[:5]}"
        )

    aligned: list[bool | None] = [None] * n_rows
    for position, label in zip(positions, labels):
        aligned[position] = label
    return aligned


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


def query_table(
    manifest: dict, query_ids: list[int], query_reporting: dict[int, dict] | None = None
) -> str:
    query_reporting = query_reporting or {}
    lines = [
        "| Column | Type | Category | Query | Model | Unparseable | Finalized deferred | Generated |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for query_id in query_ids:
        entry = manifest[str(query_id)]
        config = entry.get("config", {})
        reporting = query_reporting.get(query_id, {})
        query_type = reporting.get("type") or "—"
        categories = reporting.get("category")
        category_text = ", ".join(categories) if categories else "—"
        generated = entry.get("completed_at") or entry.get("started_at") or "unknown"
        finalized_deferred = entry.get("n_finalized_input_too_long", 0) + entry.get(
            "n_finalized_generation_length", 0
        )
        if generated != "unknown":
            generated = generated[:10]
        lines.append(
            "| `{column}` | {query_type} | {category} | {query} | {model} | "
            "{unparseable} | {finalized_deferred} | {generated} |".format(
                column=f"label_q{query_id}",
                query_type=markdown_cell(query_type),
                category=markdown_cell(category_text),
                query=markdown_cell(config.get("filter", "")),
                model=markdown_cell(config.get("model", "unknown")),
                unparseable=entry.get("n_unparseable", 0),
                finalized_deferred=finalized_deferred,
                generated=markdown_cell(generated),
            )
        )
    return "\n".join(lines)


def render_card(
    template_path: Path,
    manifest: dict,
    query_ids: list[int],
    n_rows: int,
    base_filename: str,
    merged_filename: str,
    labels_dirname: str,
    query_reporting: dict[int, dict] | None = None,
) -> str:
    raw_template = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{SIZE_CATEGORY}}": size_category(n_rows),
        "{{N_ROWS}}": str(n_rows),
        "{{BASE_FILENAME}}": base_filename,
        "{{MERGED_FILENAME}}": merged_filename,
        "{{LABELS_DIRNAME}}": labels_dirname,
        "{{QUERY_TABLE}}": query_table(manifest, query_ids, query_reporting),
    }
    missing = [marker for marker in replacements if marker not in raw_template]
    if missing:
        raise ValueError(f"Card template is missing required markers: {missing}")
    rendered = raw_template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    return rendered


def copy_label_audit(
    labels_dir: Path, destination: Path, manifest: dict, query_ids: list[int]
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
        destination_query_dir = destination / source_query_dir.name
        destination_query_dir.mkdir()
        for fragment_path in sorted(source_query_dir.glob("part-*.parquet")):
            shutil.copy2(fragment_path, destination_query_dir / fragment_path.name)
        for audit_name in ("deferred.parquet", "finalized_deferred.parquet"):
            audit_path = source_query_dir / audit_name
            if audit_path.is_file():
                shutil.copy2(audit_path, destination_query_dir / audit_name)


def _prepare_publish_locked(
    dataset_path: Path,
    labels_dir: Path,
    publish_dir: Path,
    query_ids: set[int] | None,
    template_path: Path,
    query_manifest_path: Path | None = None,
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
    base_hash = file_sha256(dataset_path)
    base = pq.read_table(dataset_path)
    selected = selected_query_ids(labels_dir, query_ids)
    manifest = read_query_manifests(labels_dir, selected)
    query_reporting = (
        read_query_reporting(query_manifest_path)
        if query_manifest_path is not None
        else {}
    )

    label_columns: dict[int, list[bool | None]] = {}
    for query_id in selected:
        entry = manifest[str(query_id)]
        query_dir = labels_dir / f"query_id={query_id}"
        if not entry.get("config_fingerprint"):
            raise ValueError(f"Q{query_id}: manifest has no trusted config fingerprint")
        if entry.get("n_deferred", 0) or (query_dir / "deferred.parquet").exists():
            raise ValueError(
                f"Q{query_id}: deferred rows remain; rerun labeling or explicitly finalize them"
            )
        if entry.get("n_rows") != base.num_rows:
            raise ValueError(
                f"Q{query_id}: manifest row count {entry.get('n_rows')} "
                f"does not match base row count {base.num_rows}"
            )
        labelled_hash = entry.get("config", {}).get("dataset", {}).get("sha256")
        if not labelled_hash:
            raise ValueError(f"Q{query_id}: manifest has no base dataset SHA-256")
        if labelled_hash != base_hash:
            raise ValueError(
                f"Q{query_id}: base dataset SHA-256 does not match the file used for labeling"
            )
        label_columns[query_id] = read_complete_labels(query_dir, base.num_rows)

    merged = base
    for query_id in selected:
        column_name = f"label_q{query_id}"
        if column_name in merged.column_names:
            raise ValueError(f"Base dataset already contains {column_name}")
        merged = merged.append_column(
            f"label_q{query_id}", pa.array(label_columns[query_id], pa.bool_())
        )

    publish_parent = publish_dir.parent
    publish_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{publish_dir.name}-", dir=publish_parent)
    )
    base_filename = dataset_path.name
    merged_filename = f"{dataset_path.stem}.merged.parquet"
    labels_dirname = labels_dir.name
    try:
        shutil.copy2(dataset_path, temporary / base_filename)
        pq.write_table(merged, temporary / merged_filename)
        copy_label_audit(labels_dir, temporary / labels_dirname, manifest, selected)
        card = render_card(
            template_path,
            manifest,
            selected,
            base.num_rows,
            base_filename,
            merged_filename,
            labels_dirname,
            query_reporting,
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
        commit_message=f"Publish ground-truth labels {datetime.now(timezone.utc):%Y-%m-%d}",
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
    artifacts = prepare_publish(**settings)
    commit_hash = upload_to_hub(repo_id, artifacts["publish_dir"])
    print(f"Published {repo_id} at commit {commit_hash}")


if __name__ == "__main__":
    main()
