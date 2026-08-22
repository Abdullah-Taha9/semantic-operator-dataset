"""Generate checkpointed boolean labels for every row of one Parquet dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import tomllib
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Callable, TypeVar
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv


load_dotenv()

_VLLM_ENGINE = None
_ANSWER_PATTERN = re.compile(r"<answer>([^<>]*?)</answer>", re.IGNORECASE | re.DOTALL)
_TRUE_ANSWERS = frozenset({"true", "1", "yes"})
_FALSE_ANSWERS = frozenset({"false", "0", "no"})
_T = TypeVar("_T")
Response = tuple[str, str | None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_manifest(path: Path, default_system_prompt: str) -> list[dict]:
    """Read only the query fields used by the labeler."""
    queries: list[dict] = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                query_id = raw["id"]
                filter_text = raw["filter"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid manifest entry on line {line_number}: {error}") from error

            if not isinstance(query_id, int) or isinstance(query_id, bool):
                raise ValueError(f"Manifest line {line_number}: id must be an integer")
            if query_id in seen_ids:
                raise ValueError(f"Manifest line {line_number}: duplicate query id {query_id}")
            if not isinstance(filter_text, str):
                raise ValueError(f"Manifest line {line_number}: filter must be a string")
            system_prompt = raw.get("system_prompt", default_system_prompt)
            if not isinstance(system_prompt, str):
                raise ValueError(f"Manifest line {line_number}: system_prompt must be a string")

            seen_ids.add(query_id)
            queries.append(
                {
                    "id": query_id,
                    "filter": filter_text,
                    "system_prompt": system_prompt,
                }
            )

    if not queries:
        raise ValueError("Manifest contains no queries")
    return queries


def select_queries(queries: list[dict], query_ids: set[int] | None) -> list[dict]:
    if query_ids is None:
        selected = queries
    else:
        selected = [query for query in queries if query["id"] in query_ids]
        missing = query_ids - {query["id"] for query in selected}
        if missing:
            raise ValueError(f"Unknown query ids requested: {sorted(missing)}")
    if not selected:
        raise ValueError("No queries selected")

    return selected


def referenced_columns(template: str) -> list[str]:
    columns: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name and field_name not in columns:
            columns.append(field_name)
    return columns


def dataset_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    metadata = pq.read_metadata(path)
    return {
        "sha256": digest.hexdigest(),
        "n_rows": metadata.num_rows,
        "size_bytes": path.stat().st_size,
    }


def query_config(
    query: dict,
    backend: str,
    model: str,
    generation_parameters: dict,
    engine_parameters: dict,
    identity: dict,
) -> dict:
    return {
        "backend": backend,
        "model": model,
        "filter": query["filter"],
        "system_prompt": query["system_prompt"],
        "output_protocol": "last-answer-tag-v1",
        "generation_parameters": generation_parameters,
        "engine_parameters": engine_parameters,
        "dataset": identity,
    }


def config_fingerprint(config: dict) -> str:
    fingerprinted_config = dict(config)
    engine_parameters = dict(fingerprinted_config.get("engine_parameters", {}))
    engine_parameters.pop("tensor_parallel_size", None)
    engine_parameters.pop("max_model_len", None)
    fingerprinted_config["engine_parameters"] = engine_parameters
    encoded = json.dumps(
        fingerprinted_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid output manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Output manifest {path} must contain a JSON object")
    return value


def write_manifest_atomic(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid4().hex}.json")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fragment_state(output_dir: Path, n_rows: int) -> tuple[set[int], int]:
    positions: list[int] = []
    n_unparseable = 0
    for fragment_path in sorted(output_dir.glob("part-*.parquet")):
        schema = pq.read_schema(fragment_path)
        if schema.names != ["row_position", "label"]:
            raise ValueError(
                f"Unexpected schema in {fragment_path}; expected row_position and label only"
            )
        if not pa.types.is_integer(schema.field("row_position").type):
            raise ValueError(f"row_position must be an integer in {fragment_path}")
        if not pa.types.is_boolean(schema.field("label").type):
            raise ValueError(f"label must be boolean in {fragment_path}")
        table = pq.read_table(fragment_path, columns=["row_position", "label"])
        fragment_positions = table.column("row_position").to_pylist()
        if any(position is None or not isinstance(position, int) for position in fragment_positions):
            raise ValueError(f"Invalid row_position in {fragment_path}")
        positions.extend(fragment_positions)
        n_unparseable += table.column("label").null_count

    unique = set(positions)
    if len(unique) != len(positions):
        raise ValueError(f"Duplicate row positions found in {output_dir}")
    invalid = sorted(position for position in unique if position < 0 or position >= n_rows)
    if invalid:
        raise ValueError(f"Out-of-range row positions found in {output_dir}: {invalid[:5]}")
    return unique, n_unparseable


def write_fragment_atomic(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".tmp-{uuid4().hex}.parquet"
    destination = output_dir / f"part-{uuid4().hex}.parquet"
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema([pa.field("row_position", pa.int64()), pa.field("label", pa.bool_())]),
    )
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def parse_bool(response: str) -> bool | None:
    matches = list(_ANSWER_PATTERN.finditer(response))
    if not matches:
        return None
    normalized = matches[-1].group(1).strip().casefold()
    if normalized in _TRUE_ANSWERS:
        return True
    if normalized in _FALSE_ANSWERS:
        return False
    return None


def render_rows(query: dict, columns: dict[str, list], positions: list[int]) -> list[dict]:
    rendered: list[dict] = []
    for position in positions:
        values = {name: values[position] for name, values in columns.items()}
        # Rendering errors are genuine failures and deliberately abort the run.
        prompt = query["filter"].format(**values)
        rendered.append(
            {
                "position": position,
                "prompt": prompt,
                "system_prompt": query["system_prompt"],
            }
        )
    return rendered


def retry(call: Callable[[], _T], attempts: int, delay_seconds: float) -> _T:
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            if attempt == attempts:
                raise
            print(
                f"Model call failed (attempt {attempt}/{attempts}): {error}; "
                f"retrying in {delay_seconds}s"
            )
            time.sleep(delay_seconds)
    raise AssertionError("unreachable")


def call_api_batch(
    rows: list[dict],
    model: str,
    concurrency: int,
    attempts: int,
    delay_seconds: float,
    generation_parameters: dict,
    on_response: Callable[[int, str, str | None], None],
) -> dict[int, Response]:
    import litellm

    def one(row: dict) -> Response:
        def invoke() -> Response:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["prompt"]},
                ],
                **generation_parameters,
            )
            text = response.choices[0].message.content
            if text is None:
                raise RuntimeError(f"API returned no model response for row {row['position']}")
            return text, response.choices[0].finish_reason

        return retry(invoke, attempts, delay_seconds)

    responses: dict[int, Response] = {}
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, row): row["position"] for row in rows}
        for future in as_completed(futures):
            position = futures[future]
            try:
                text, finish_reason = future.result()
            except CancelledError:
                continue
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    for pending in futures:
                        pending.cancel()
            else:
                on_response(position, text, finish_reason)
                responses[position] = (text, finish_reason)

    if first_error is not None:
        raise first_error
    return responses


def vllm_engine(model: str, engine_parameters: dict):
    global _VLLM_ENGINE
    if _VLLM_ENGINE is None:
        from vllm import LLM

        _VLLM_ENGINE = LLM(model=model, **engine_parameters)
    return _VLLM_ENGINE


def call_vllm_batch(
    rows: list[dict],
    model: str,
    attempts: int,
    delay_seconds: float,
    generation_parameters: dict,
    engine_parameters: dict,
    on_response: Callable[[int, str, str | None], None],
) -> dict[int, Response]:
    from vllm import SamplingParams

    def invoke() -> list:
        return vllm_engine(model, engine_parameters).chat(
            [
                [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["prompt"]},
                ]
                for row in rows
            ],
            SamplingParams(**generation_parameters),
            use_tqdm=True,
        )

    outputs = retry(invoke, attempts, delay_seconds)
    if len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(rows)} rows")

    responses: dict[int, Response] = {}
    for row, output in zip(rows, outputs):
        if not output.outputs:
            raise RuntimeError(f"vLLM returned no model response for row {row['position']}")
        completion = output.outputs[0]
        text = completion.text
        if text is None:
            raise RuntimeError(f"vLLM returned no model response for row {row['position']}")
        finish_reason = completion.finish_reason
        on_response(row["position"], text, finish_reason)
        responses[row["position"]] = (text, finish_reason)
    return responses


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            config_fingerprint TEXT NOT NULL,
            row_position INTEGER NOT NULL,
            response TEXT NOT NULL,
            finish_reason TEXT,
            PRIMARY KEY (config_fingerprint, row_position)
        )
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(responses)")}
    if "finish_reason" not in columns:
        connection.execute("ALTER TABLE responses ADD COLUMN finish_reason TEXT")
    connection.commit()
    return connection


def cached_responses(
    connection: sqlite3.Connection | None, fingerprint: str, positions: list[int]
) -> dict[int, Response]:
    if connection is None or not positions:
        return {}
    wanted = set(positions)
    rows = connection.execute(
        "SELECT row_position, response, finish_reason FROM responses "
        "WHERE config_fingerprint = ?",
        (fingerprint,),
    )
    return {
        position: (response, finish_reason)
        for position, response, finish_reason in rows
        if position in wanted
    }


def store_cached_response(
    connection: sqlite3.Connection | None,
    fingerprint: str,
    position: int,
    response: str,
    finish_reason: str | None,
) -> None:
    if connection is None:
        return
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO responses "
            "(config_fingerprint, row_position, response, finish_reason) VALUES (?, ?, ?, ?)",
            (fingerprint, position, response, finish_reason),
        )


def delete_cached_positions(
    connection: sqlite3.Connection | None, fingerprint: str, positions: set[int] | list[int]
) -> None:
    if connection is None or not positions:
        return
    with connection:
        connection.executemany(
            "DELETE FROM responses WHERE config_fingerprint = ? AND row_position = ?",
            ((fingerprint, position) for position in positions),
        )


def delete_cache_files(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def update_progress(
    manifest_path: Path,
    query_id: int,
    entry: dict,
    n_done: int,
    n_unparseable: int,
    completed: bool,
) -> None:
    entry["n_labelled"] = n_done - n_unparseable
    entry["n_unparseable"] = n_unparseable
    entry["completed_at"] = utc_now() if completed else None
    manifest = read_manifest(manifest_path)
    manifest[str(query_id)] = entry
    write_manifest_atomic(manifest_path, manifest)


def label_query(
    query: dict,
    dataset_path: Path,
    output_root: Path,
    backend: str,
    model: str,
    generation_parameters: dict,
    engine_parameters: dict,
    identity: dict,
    checkpoint_every: int,
    concurrency: int,
    attempts: int,
    delay_seconds: float,
    enable_cache: bool,
    force: bool,
) -> None:
    query_id = query["id"]
    output_dir = output_root / f"query_id={query_id}"
    manifest_path = output_root / "manifest.json"
    cache_path = output_root / ".cache" / f"query_id={query_id}.sqlite3"
    config = query_config(
        query, backend, model, generation_parameters, engine_parameters, identity
    )
    fingerprint = config_fingerprint(config)

    manifest = read_manifest(manifest_path)
    existing_entry = manifest.get(str(query_id))
    result_dir_exists = output_dir.exists()

    if result_dir_exists:
        if not existing_entry or not existing_entry.get("config_fingerprint"):
            raise RuntimeError(
                f"Q{query_id}: a result directory exists without a trusted manifest fingerprint; "
                "delete the query directory manually to start over"
            )
        if existing_entry["config_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Q{query_id}: configuration does not match existing fragments; "
                "delete the query directory manually to change methodology"
            )

    if force:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        delete_cache_files(cache_path)
        manifest = read_manifest(manifest_path)
        manifest.pop(str(query_id), None)
        write_manifest_atomic(manifest_path, manifest)
        existing_entry = None

    output_dir.mkdir(parents=True, exist_ok=True)
    done, n_unparseable = fragment_state(output_dir, identity["n_rows"])

    # If the result directory was deliberately removed, a stale entry may be replaced.
    if not result_dir_exists or force:
        existing_entry = None

    entry = {
        "config": config,
        "config_fingerprint": fingerprint,
        "dataset_path": str(dataset_path),
        "n_rows": identity["n_rows"],
        "n_labelled": len(done) - n_unparseable,
        "n_unparseable": n_unparseable,
        "started_at": (existing_entry or {}).get("started_at", utc_now()),
        "completed_at": None,
    }
    manifest = read_manifest(manifest_path)
    manifest[str(query_id)] = entry
    write_manifest_atomic(manifest_path, manifest)

    if len(done) == identity["n_rows"]:
        update_progress(manifest_path, query_id, entry, len(done), n_unparseable, True)
        print(f"Q{query_id}: already fully labeled ({len(done)} rows)")
        return

    columns_needed = referenced_columns(query["filter"])
    table = pq.read_table(dataset_path, columns=columns_needed)
    columns = {name: table.column(name).to_pylist() for name in columns_needed}
    remaining = [position for position in range(identity["n_rows"]) if position not in done]

    cache = open_cache(cache_path) if enable_cache else None
    try:
        delete_cached_positions(cache, fingerprint, done)
        for start in range(0, len(remaining), checkpoint_every):
            positions = remaining[start : start + checkpoint_every]
            rows = render_rows(query, columns, positions)
            responses = cached_responses(cache, fingerprint, positions)
            missing_rows = [row for row in rows if row["position"] not in responses]

            def on_response(
                position: int, response: str, finish_reason: str | None
            ) -> None:
                store_cached_response(
                    cache, fingerprint, position, response, finish_reason
                )

            if missing_rows:
                if backend == "api":
                    responses.update(
                        call_api_batch(
                            missing_rows,
                            model,
                            concurrency,
                            attempts,
                            delay_seconds,
                            generation_parameters,
                            on_response,
                        )
                    )
                else:
                    responses.update(
                        call_vllm_batch(
                            missing_rows,
                            model,
                            attempts,
                            delay_seconds,
                            generation_parameters,
                            engine_parameters,
                            on_response,
                        )
                    )

            if set(responses) != set(positions):
                missing = sorted(set(positions) - set(responses))
                raise RuntimeError(f"Q{query_id}: missing model responses for rows {missing[:5]}")

            labels = [
                None
                if responses[position][1] == "length"
                else parse_bool(responses[position][0])
                for position in positions
            ]
            write_fragment_atomic(
                output_dir,
                [
                    {"row_position": position, "label": label}
                    for position, label in zip(positions, labels)
                ],
            )
            done.update(positions)
            n_unparseable += sum(label is None for label in labels)
            update_progress(
                manifest_path,
                query_id,
                entry,
                len(done),
                n_unparseable,
                len(done) == identity["n_rows"],
            )
            delete_cached_positions(cache, fingerprint, positions)
            print(f"Q{query_id}: checkpointed {len(done)}/{identity['n_rows']}")
    finally:
        if cache is not None:
            cache.close()


def run_labeling(
    manifest_path: Path,
    dataset_path: Path,
    output_dir: Path,
    default_system_prompt: str,
    backend: str,
    model: str,
    generation_parameters: dict,
    engine_parameters: dict,
    query_ids: set[int] | None,
    concurrency: int,
    api_checkpoint_every: int,
    vllm_checkpoint_every: int,
    attempts: int,
    delay_seconds: float,
    enable_cache: bool,
    force: bool = False,
) -> None:
    if backend not in {"api", "vllm"}:
        raise ValueError("label.backend must be 'api' or 'vllm'")
    if concurrency < 1 or attempts < 1:
        raise ValueError("Concurrency and retry attempts must be positive")

    queries = select_queries(parse_manifest(manifest_path, default_system_prompt), query_ids)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Base dataset not found: {dataset_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = dataset_identity(dataset_path)
    checkpoint_every = api_checkpoint_every if backend == "api" else vllm_checkpoint_every
    if checkpoint_every < 1:
        raise ValueError("Checkpoint size must be positive")

    for query in queries:
        label_query(
            query,
            dataset_path,
            output_dir,
            backend,
            model,
            generation_parameters,
            engine_parameters,
            identity,
            checkpoint_every,
            concurrency,
            attempts,
            delay_seconds,
            enable_cache,
            force,
        )


def load_label_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    try:
        shared = config["shared"]
        section = config["label"]
        configured_queries = section["queries"]
        dataset_path = Path(shared["dataset"])
        backend = section["backend"]
        parameter_key = f"{backend}_parameters"
        generation_parameters = dict(section.get(parameter_key, {}))
        if backend == "vllm":
            # vLLM 0.25.1 otherwise defaults to an output-only limit of 16 tokens.
            generation_parameters.setdefault("max_tokens", None)
        engine_parameters = (
            dict(section.get("vllm_engine_parameters", {})) if backend == "vllm" else {}
        )
        return {
            "manifest_path": Path(shared["manifest"]),
            "dataset_path": dataset_path,
            "output_dir": dataset_path.with_suffix("") / "labels",
            "default_system_prompt": section["system_prompt"],
            "backend": backend,
            "model": section["model"],
            "generation_parameters": generation_parameters,
            "engine_parameters": engine_parameters,
            "query_ids": set(configured_queries) or None,
            "concurrency": section["concurrency"],
            "api_checkpoint_every": section["api_checkpoint_every"],
            "vllm_checkpoint_every": section["vllm_checkpoint_every"],
            "attempts": section["attempts"],
            "delay_seconds": section["retry_delay_seconds"],
            "enable_cache": section["enable_cache"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid label configuration in {path}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard and regenerate selected queries only when their fingerprints match",
    )
    args = parser.parse_args()
    run_labeling(**load_label_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
