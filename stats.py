#!/usr/bin/env python3
"""stats.py - Compute sequence length and token distribution statistics for Parquet datasets."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.compute as pc
import pyarrow.dataset as ds


def compute_length_stats(
    file_path: str,
    cols: List[str],
    dataset_name: str,
    char_per_tok: float = 3.8,
    quantiles: List[float] = None,
) -> Dict[str, Any]:
    """Compute UTF-8 character and estimated token statistics across specified columns."""
    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    # Load only the specified columns (ignoring heavy embedding vectors)
    table = ds.dataset(str(path)).to_table(columns=cols)

    # Sum UTF-8 character length across all selected columns
    total_len = None
    for col in cols:
        col_str = pc.cast(table[col], "string")
        col_len = pc.utf8_length(pc.fill_null(col_str, ""))
        total_len = col_len if total_len is None else pc.add(total_len, col_len)

    total_rows = len(table)
    min_chars = int(pc.min(total_len).as_py())
    max_chars = int(pc.max(total_len).as_py())
    mean_chars = float(pc.mean(total_len).as_py())
    std_chars = float(pc.stddev(total_len).as_py())

    # Calculate quantiles
    q_vals = pc.quantile(total_len, q=quantiles).to_pylist()
    percentiles = {}
    for q, val in zip(quantiles, q_vals):
        pct_label = f"p{int(q*100) if q*100 == int(q*100) else q*100}"
        chars = int(val)
        percentiles[pct_label] = {
            "quantile": q,
            "characters": chars,
            "est_tokens": int(chars / char_per_tok),
        }

    return {
        "dataset_name": dataset_name,
        "file_path": str(file_path),
        "analyzed_columns": cols,
        "total_rows": total_rows,
        "char_per_token_ratio": char_per_tok,
        "metrics": {
            "min_characters": min_chars,
            "min_est_tokens": int(min_chars / char_per_tok),
            "max_characters": max_chars,
            "max_est_tokens": int(max_chars / char_per_tok),
            "mean_characters": round(mean_chars, 2),
            "mean_est_tokens": int(mean_chars / char_per_tok),
            "std_characters": round(std_chars, 2),
        },
        "percentiles": percentiles,
    }


def format_stats_report(stats: Dict[str, Any]) -> str:
    """Format dictionary stats into a readable CLI and report string."""
    name = stats["dataset_name"]
    metrics = stats["metrics"]
    ratio = stats["char_per_token_ratio"]

    lines = [
        f"================== {name} LENGTH DISTRIBUTION ==================",
        f"File:            {stats['file_path']}",
        f"Columns:         {', '.join(stats['analyzed_columns'])}",
        f"Total Rows:      {stats['total_rows']:,}",
        f"Min Length:      {metrics['min_characters']:,} chars (~{metrics['min_est_tokens']:,} tokens)",
        f"Mean ± Std:      {metrics['mean_characters']:.1f} ± {metrics['std_characters']:.1f} chars",
        "-----------------------------------------------------------------",
        f"Percentile       Characters      ~Est. Tokens (@ ~{ratio} char/tok)",
        "-----------------------------------------------------------------",
    ]

    for label, item in stats["percentiles"].items():
        chars = item["characters"]
        tokens = item["est_tokens"]
        lines.append(f"{label:<15}  {chars:>8,} chars       ~{tokens:>6,} tokens")

    lines.append(
        f"p100 (Max)       {metrics['max_characters']:>8,} chars       ~{metrics['max_est_tokens']:>6,} tokens"
    )
    lines.append("=================================================================\n")
    return "\n".join(lines)


def main():
    output_dir = Path("stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "name": "PRODUCTS",
            "file": "datasets/products_filtered_with_embeddings.parquet",
            "cols": [
                "product_title",
                "features_json",
                "description_json",
                "details_json",
            ],
        },
        {
            "name": "REVIEWS",
            "file": "datasets/reviews_filtered_with_embeddings.parquet",
            "cols": ["review_title", "review_text"],
        },
    ]

    all_stats = []
    text_reports = []

    for task in tasks:
        stats = compute_length_stats(
            file_path=task["file"],
            cols=task["cols"],
            dataset_name=task["name"],
        )
        all_stats.append(stats)

        formatted_report = format_stats_report(stats)
        print(formatted_report)
        text_reports.append(formatted_report)

    # 1. Save full JSON report
    json_path = output_dir / "length_distribution_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "datasets": all_stats,
            },
            f,
            indent=2,
        )

    # 2. Save text summary report
    txt_path = output_dir / "summary_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(text_reports))

    print(f" Saved structured stats -> {json_path}")
    print(f" Saved text summary      -> {txt_path}")


if __name__ == "__main__":
    main()