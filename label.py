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
Outcome = tuple[bool | None, str]
_DEFERRED_REASONS = frozenset({"input_too_long", "generation_length"})


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
                raise ValueError(
                    f"Invalid manifest entry on line {line_number}: {error}"
                ) from error

            if not isinstance(query_id, int) or isinstance(query_id, bool):
                raise ValueError(f"Manifest line {line_number}: id must be an integer")
            if query_id in seen_ids:
                raise ValueError(
                    f"Manifest line {line_number}: duplicate query id {query_id}"
                )
            if not isinstance(filter_text, str):
                raise ValueError(
                    f"Manifest line {line_number}: filter must be a string"
                )
            system_prompt = raw.get("system_prompt", default_system_prompt)
            if not isinstance(system_prompt, str):
                raise ValueError(
                    f"Manifest line {line_number}: system_prompt must be a string"
                )

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


def fragment_state(output_dir: Path, n_rows: int) -> tuple[set[int], set[int]]:
    positions: list[int] = []
    null_positions: set[int] = set()
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
        if any(
            position is None or not isinstance(position, int)
            for position in fragment_positions
        ):
            raise ValueError(f"Invalid row_position in {fragment_path}")
        positions.extend(fragment_positions)
        labels = table.column("label").to_pylist()
        null_positions.update(
            position
            for position, label in zip(fragment_positions, labels)
            if label is None
        )

    unique = set(positions)
    if len(unique) != len(positions):
        raise ValueError(f"Duplicate row positions found in {output_dir}")
    invalid = sorted(
        position for position in unique if position < 0 or position >= n_rows
    )
    if invalid:
        raise ValueError(
            f"Out-of-range row positions found in {output_dir}: {invalid[:5]}"
        )
    return unique, null_positions


def write_fragment_atomic(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".tmp-{uuid4().hex}.parquet"
    destination = output_dir / f"part-{uuid4().hex}.parquet"
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [pa.field("row_position", pa.int64()), pa.field("label", pa.bool_())]
        ),
    )
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_reason_ledger(path: Path, n_rows: int) -> dict[int, str]:
    if not path.exists():
        return {}
    table = pq.read_table(path)
    if table.column_names != ["row_position", "reason"]:
        raise ValueError(
            f"Unexpected schema in {path}; expected row_position and reason only"
        )
    if not pa.types.is_integer(table.schema.field("row_position").type):
        raise ValueError(f"row_position must be an integer in {path}")
    if not pa.types.is_string(table.schema.field("reason").type):
        raise ValueError(f"reason must be a string in {path}")
    positions = table.column("row_position").to_pylist()
    reasons = table.column("reason").to_pylist()
    if any(
        position is None
        or not isinstance(position, int)
        or position < 0
        or position >= n_rows
        for position in positions
    ):
        raise ValueError(f"Invalid row_position in {path}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"Duplicate row positions found in {path}")
    if any(reason not in _DEFERRED_REASONS for reason in reasons):
        raise ValueError(f"Invalid deferred reason in {path}")
    return dict(zip(positions, reasons))


def write_reason_ledger_atomic(path: Path, entries: dict[int, str]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid4().hex}.parquet")
    table = pa.Table.from_pylist(
        [
            {"row_position": position, "reason": entries[position]}
            for position in sorted(entries)
        ],
        schema=pa.schema(
            [pa.field("row_position", pa.int64()), pa.field("reason", pa.string())]
        ),
    )
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def render_rows(
    query: dict, columns: dict[str, list], positions: list[int]
) -> list[dict]:
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
                raise RuntimeError(
                    f"API returned no model response for row {row['position']}"
                )
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
    on_outcomes: Callable[[dict[int, Outcome]], None],
    on_deferred: Callable[[dict[int, str]], None],
) -> tuple[dict[int, Outcome], dict[int, str]]:
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError

    engine = vllm_engine(model, engine_parameters)
    sampling_params = SamplingParams(**generation_parameters)
    request_positions: dict[str, int] = {}
    deferred: dict[int, str] = {}

    def is_input_too_long(error: BaseException) -> bool:
        if isinstance(error, VLLMValidationError):
            return error.parameter in {"input_text", "input_tokens"}
        # vLLM 0.25.1 raises plain ValueError when a prompt exactly fills
        # max_model_len and therefore leaves no token for generation.
        message = str(error)
        return isinstance(error, ValueError) and (
            message.startswith("The decoder prompt (length ")
            and "plus the number of requested output tokens (at least 1)" in message
        )

    for row in rows:
        request_ids: list[str] | None = None
        for attempt in range(1, attempts + 1):
            try:
                request_ids = engine.enqueue_chat(
                    [
                        {"role": "system", "content": row["system_prompt"]},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    sampling_params,
                    use_tqdm=False,
                )
                break
            except Exception as error:
                if is_input_too_long(error):
                    deferred[row["position"]] = "input_too_long"
                    break
                if attempt == attempts:
                    raise
                print(
                    f"Model call failed (attempt {attempt}/{attempts}): {error}; "
                    f"retrying in {delay_seconds}s"
                )
                time.sleep(delay_seconds)

        if request_ids is None:
            continue
        if len(request_ids) != 1:
            raise RuntimeError(
                f"vLLM returned {len(request_ids)} request ids for row {row['position']}"
            )
        request_id = str(request_ids[0])
        if request_id in request_positions:
            raise RuntimeError(f"vLLM returned duplicate request id {request_id}")
        request_positions[request_id] = row["position"]

    if deferred:
        on_deferred(dict(deferred))

    outcomes: dict[int, Outcome] = {}
    while engine.llm_engine.has_unfinished_requests():
        step_outputs = retry(engine.llm_engine.step, attempts, delay_seconds)
        finished_in_step: dict[int, Outcome] = {}
        deferred_in_step: dict[int, str] = {}
        for output in step_outputs:
            if not output.finished:
                continue
            request_id = str(output.request_id)
            if request_id not in request_positions:
                raise RuntimeError(f"vLLM returned unknown request id {request_id}")
            position = request_positions.pop(request_id)
            if not output.outputs:
                raise RuntimeError(
                    f"vLLM returned no model response for row {position}"
                )
            completion = output.outputs[0]
            if completion.finish_reason == "length":
                deferred_in_step[position] = "generation_length"
                continue
            if completion.text is None:
                raise RuntimeError(
                    f"vLLM returned no model response for row {position}"
                )
            label = parse_bool(completion.text)
            finished_in_step[position] = (
                label,
                "labelled" if label is not None else "unparseable",
            )

        if finished_in_step:
            on_outcomes(finished_in_step)
            outcomes.update(finished_in_step)
        if deferred_in_step:
            on_deferred(deferred_in_step)
            deferred.update(deferred_in_step)

    if request_positions:
        missing = sorted(request_positions.values())
        raise RuntimeError(f"vLLM returned no final output for rows {missing[:5]}")
    if set(outcomes) & set(deferred):
        raise RuntimeError("vLLM marked the same rows completed and deferred")
    expected = {row["position"] for row in rows}
    accounted = set(outcomes) | set(deferred)
    if accounted != expected:
        missing = sorted(expected - accounted)
        raise RuntimeError(f"vLLM did not account for rows {missing[:5]}")
    return outcomes, deferred


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    # NORMAL remains durable across a killed Python process in WAL mode while
    # avoiding one filesystem sync for every group of completed responses.
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            config_fingerprint TEXT NOT NULL,
            row_position INTEGER NOT NULL,
            label INTEGER,
            status TEXT NOT NULL,
            PRIMARY KEY (config_fingerprint, row_position)
        )
        """
    )
    connection.commit()
    return connection


def cached_outcomes(
    connection: sqlite3.Connection | None, fingerprint: str, positions: list[int]
) -> dict[int, Outcome]:
    if connection is None or not positions:
        return {}
    wanted = set(positions)
    rows = connection.execute(
        "SELECT row_position, label, status FROM outcomes WHERE config_fingerprint = ?",
        (fingerprint,),
    )
    outcomes: dict[int, Outcome] = {}
    for position, label, status in rows:
        if position not in wanted:
            continue
        if status not in {"labelled", "unparseable"}:
            raise ValueError(
                f"Invalid cached outcome status {status!r} for row {position}"
            )
        parsed_label = None if label is None else bool(label)
        if (status == "labelled") != (parsed_label is not None):
            raise ValueError(
                f"Invalid cached label/status combination for row {position}"
            )
        outcomes[position] = (parsed_label, status)
    return outcomes


def store_cached_outcomes(
    connection: sqlite3.Connection | None,
    fingerprint: str,
    outcomes: dict[int, Outcome],
) -> None:
    if connection is None or not outcomes:
        return
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO outcomes "
            "(config_fingerprint, row_position, label, status) VALUES (?, ?, ?, ?)",
            (
                (
                    fingerprint,
                    position,
                    None if label is None else int(label),
                    status,
                )
                for position, (label, status) in outcomes.items()
            ),
        )


def delete_cached_positions(
    connection: sqlite3.Connection | None,
    fingerprint: str,
    positions: set[int] | list[int],
) -> None:
    if connection is None or not positions:
        return
    with connection:
        connection.executemany(
            "DELETE FROM outcomes WHERE config_fingerprint = ? AND row_position = ?",
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
    done: set[int],
    null_positions: set[int],
    deferred: dict[int, str],
    finalized: dict[int, str],
) -> None:
    finalized_positions = set(finalized)
    n_unparseable = len(null_positions - finalized_positions)
    entry["n_labelled"] = len(done) - len(null_positions)
    entry["n_unparseable"] = n_unparseable
    entry["n_deferred"] = len(deferred)
    entry["n_finalized_input_too_long"] = sum(
        reason == "input_too_long" for reason in finalized.values()
    )
    entry["n_finalized_generation_length"] = sum(
        reason == "generation_length" for reason in finalized.values()
    )
    entry["completed_at"] = utc_now() if len(done) == entry["n_rows"] else None
    manifest = read_manifest(manifest_path)
    manifest[str(query_id)] = entry
    write_manifest_atomic(manifest_path, manifest)


def evenly_spaced_positions(n_rows: int, sample_size: int) -> list[int]:
    if sample_size >= n_rows:
        return list(range(n_rows))
    if sample_size == 1:
        return [n_rows // 2]
    return [
        round(index * (n_rows - 1) / (sample_size - 1)) for index in range(sample_size)
    ]


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
    finalize_deferred_as_null: bool,
    canary_rows: int,
    canary_max_deferred_fraction: float,
    force: bool,
) -> None:
    query_id = query["id"]
    output_dir = output_root / f"query_id={query_id}"
    manifest_path = output_root / "manifest.json"
    cache_path = output_root / ".cache" / f"query_id={query_id}.sqlite3"
    deferred_path = output_dir / "deferred.parquet"
    finalized_path = output_dir / "finalized_deferred.parquet"
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
    done, null_positions = fragment_state(output_dir, identity["n_rows"])
    deferred = read_reason_ledger(deferred_path, identity["n_rows"])
    finalized = read_reason_ledger(finalized_path, identity["n_rows"])

    if set(finalized) - null_positions:
        invalid = sorted(set(finalized) - null_positions)
        raise ValueError(
            f"Q{query_id}: finalized deferred rows are not null: {invalid[:5]}"
        )

    # A crash may happen after a null fragment is committed but before the
    # deferred ledgers are updated. The fragment wins and the ledgers converge.
    ledgers_changed = False
    for position in set(deferred) & done:
        reason = deferred.pop(position)
        if position in null_positions:
            finalized[position] = reason
        ledgers_changed = True
    if ledgers_changed:
        write_reason_ledger_atomic(deferred_path, deferred)
        write_reason_ledger_atomic(finalized_path, finalized)

    # If the result directory was deliberately removed, a stale entry may be replaced.
    if not result_dir_exists or force:
        existing_entry = None

    entry = {
        "config": config,
        "config_fingerprint": fingerprint,
        "dataset_path": str(dataset_path),
        "n_rows": identity["n_rows"],
        "n_labelled": len(done) - len(null_positions),
        "n_unparseable": len(null_positions - set(finalized)),
        "n_deferred": len(deferred),
        "n_finalized_input_too_long": sum(
            reason == "input_too_long" for reason in finalized.values()
        ),
        "n_finalized_generation_length": sum(
            reason == "generation_length" for reason in finalized.values()
        ),
        "canary": (existing_entry or {}).get("canary"),
        "started_at": (existing_entry or {}).get("started_at", utc_now()),
        "completed_at": None,
    }
    manifest = read_manifest(manifest_path)
    manifest[str(query_id)] = entry
    write_manifest_atomic(manifest_path, manifest)

    if len(done) == identity["n_rows"]:
        update_progress(
            manifest_path, query_id, entry, done, null_positions, deferred, finalized
        )
        print(f"Q{query_id}: already fully labeled ({len(done)} rows)")
        return

    columns_needed = referenced_columns(query["filter"])
    table = pq.read_table(dataset_path, columns=columns_needed)
    columns = {name: table.column(name).to_pylist() for name in columns_needed}
    cache = open_cache(cache_path) if enable_cache else None
    cache_write_seconds = 0.0
    try:
        delete_cached_positions(cache, fingerprint, done)

        def persist_outcomes(outcomes: dict[int, Outcome]) -> None:
            nonlocal cache_write_seconds
            started = time.perf_counter()
            store_cached_outcomes(cache, fingerprint, outcomes)
            cache_write_seconds += time.perf_counter() - started

        def process_positions(
            positions: list[int], *, persist_cache: bool, persist_deferred: bool
        ) -> tuple[dict[int, Outcome], dict[int, str]]:
            rows = render_rows(query, columns, positions)
            outcomes = cached_outcomes(cache, fingerprint, positions)
            missing_rows = [row for row in rows if row["position"] not in outcomes]
            newly_deferred: dict[int, str] = {}

            def on_response(
                position: int, response: str, finish_reason: str | None
            ) -> None:
                if finish_reason == "length":
                    newly_deferred[position] = "generation_length"
                    if persist_deferred:
                        deferred[position] = "generation_length"
                        write_reason_ledger_atomic(deferred_path, deferred)
                    return
                label = parse_bool(response)
                outcome = (label, "labelled" if label is not None else "unparseable")
                outcomes[position] = outcome
                if persist_cache:
                    persist_outcomes({position: outcome})

            def on_vllm_outcomes(completed: dict[int, Outcome]) -> None:
                outcomes.update(completed)
                if persist_cache:
                    persist_outcomes(completed)

            def on_vllm_deferred(rejected: dict[int, str]) -> None:
                newly_deferred.update(rejected)
                if persist_deferred:
                    deferred.update(rejected)
                    write_reason_ledger_atomic(deferred_path, deferred)

            if missing_rows:
                if backend == "api":
                    call_api_batch(
                        missing_rows,
                        model,
                        concurrency,
                        attempts,
                        delay_seconds,
                        generation_parameters,
                        on_response,
                    )
                else:
                    call_vllm_batch(
                        missing_rows,
                        model,
                        attempts,
                        delay_seconds,
                        generation_parameters,
                        engine_parameters,
                        on_vllm_outcomes,
                        on_vllm_deferred,
                    )

            accounted = set(outcomes) | set(newly_deferred)
            if accounted != set(positions):
                missing = sorted(set(positions) - accounted)
                raise RuntimeError(
                    f"Q{query_id}: missing model outcomes for rows {missing[:5]}"
                )
            return outcomes, newly_deferred

        def commit_results(
            outcomes: dict[int, Outcome], newly_deferred: dict[int, str]
        ) -> None:
            fragment_rows = [
                {"row_position": position, "label": outcome[0]}
                for position, outcome in sorted(outcomes.items())
            ]
            if finalize_deferred_as_null:
                fragment_rows.extend(
                    {"row_position": position, "label": None}
                    for position in sorted(newly_deferred)
                )
                fragment_rows.sort(key=lambda row: row["row_position"])

            if fragment_rows:
                write_fragment_atomic(output_dir, fragment_rows)
                committed = {row["row_position"] for row in fragment_rows}
                done.update(committed)
                null_positions.update(
                    row["row_position"] for row in fragment_rows if row["label"] is None
                )

            for position in outcomes:
                deferred.pop(position, None)
            if finalize_deferred_as_null:
                finalized.update(newly_deferred)
                for position in newly_deferred:
                    deferred.pop(position, None)
            else:
                deferred.update(newly_deferred)

            write_reason_ledger_atomic(deferred_path, deferred)
            write_reason_ledger_atomic(finalized_path, finalized)
            delete_cached_positions(cache, fingerprint, list(outcomes))
            update_progress(
                manifest_path,
                query_id,
                entry,
                done,
                null_positions,
                deferred,
                finalized,
            )

        canary_passed = bool((entry.get("canary") or {}).get("passed"))
        attempted_this_run: set[int] = set()
        if backend == "vllm" and not canary_passed and not done:
            sample_positions = evenly_spaced_positions(identity["n_rows"], canary_rows)
            canary_outcomes, canary_deferred = process_positions(
                sample_positions, persist_cache=False, persist_deferred=False
            )
            deferred_fraction = (
                len(canary_deferred) / len(sample_positions)
                if sample_positions
                else 0.0
            )
            canary_report = {
                "passed": deferred_fraction <= canary_max_deferred_fraction,
                "checked": len(sample_positions),
                "n_input_too_long": sum(
                    reason == "input_too_long" for reason in canary_deferred.values()
                ),
                "n_generation_length": sum(
                    reason == "generation_length" for reason in canary_deferred.values()
                ),
                "deferred_fraction": deferred_fraction,
                "max_deferred_fraction": canary_max_deferred_fraction,
            }
            entry["canary"] = canary_report
            manifest = read_manifest(manifest_path)
            manifest[str(query_id)] = entry
            write_manifest_atomic(manifest_path, manifest)
            if not canary_report["passed"]:
                raise RuntimeError(
                    f"Q{query_id}: vLLM canary deferred {len(canary_deferred)}/"
                    f"{len(sample_positions)} rows ({deferred_fraction:.1%}), exceeding "
                    f"the allowed {canary_max_deferred_fraction:.1%}; check max_model_len "
                    "and any configured max_tokens; no canary fragments were written"
                )
            commit_results(canary_outcomes, canary_deferred)
            attempted_this_run.update(sample_positions)
            print(
                f"Q{query_id}: canary passed with {len(canary_deferred)}/"
                f"{len(sample_positions)} deferred rows"
            )

        if finalize_deferred_as_null:
            known_deferred = {
                position: reason
                for position, reason in deferred.items()
                if position not in done
            }
            cached_resolutions = cached_outcomes(
                cache, fingerprint, list(known_deferred)
            )
            if cached_resolutions:
                commit_results(cached_resolutions, {})
                known_deferred = {
                    position: reason
                    for position, reason in deferred.items()
                    if position not in done
                }
            if known_deferred:
                commit_results({}, known_deferred)
                print(
                    f"Q{query_id}: finalized {len(known_deferred)} deferred rows as null"
                )

        remaining = [
            position
            for position in range(identity["n_rows"])
            if position not in done and position not in attempted_this_run
        ]
        for start in range(0, len(remaining), checkpoint_every):
            positions = remaining[start : start + checkpoint_every]
            outcomes, newly_deferred = process_positions(
                positions,
                persist_cache=True,
                persist_deferred=not finalize_deferred_as_null,
            )
            commit_results(outcomes, newly_deferred)
            print(
                f"Q{query_id}: checkpointed {len(done)}/{identity['n_rows']} "
                f"({len(deferred)} deferred)"
            )

        update_progress(
            manifest_path, query_id, entry, done, null_positions, deferred, finalized
        )
        if deferred:
            print(
                f"Q{query_id}: pass finished with {len(deferred)} deferred rows; "
                "increase capacity and rerun, or set finalize_deferred_as_null=true"
            )
        if enable_cache:
            print(f"Q{query_id}: SQLite cache writes took {cache_write_seconds:.3f}s")
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
    finalize_deferred_as_null: bool,
    canary_rows: int,
    canary_max_deferred_fraction: float,
    force: bool = False,
) -> None:
    if backend not in {"api", "vllm"}:
        raise ValueError("label.backend must be 'api' or 'vllm'")
    if concurrency < 1 or attempts < 1:
        raise ValueError("Concurrency and retry attempts must be positive")
    if canary_rows < 1:
        raise ValueError("Canary row count must be positive")
    if not 0.0 <= canary_max_deferred_fraction <= 1.0:
        raise ValueError("Canary deferred fraction must be between 0 and 1")

    queries = select_queries(
        parse_manifest(manifest_path, default_system_prompt), query_ids
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Base dataset not found: {dataset_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = dataset_identity(dataset_path)
    checkpoint_every = (
        api_checkpoint_every if backend == "api" else vllm_checkpoint_every
    )
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
            finalize_deferred_as_null,
            canary_rows,
            canary_max_deferred_fraction,
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
            "finalize_deferred_as_null": section["finalize_deferred_as_null"],
            "canary_rows": section["canary_rows"],
            "canary_max_deferred_fraction": section["canary_max_deferred_fraction"],
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
