"""
ingest.py
=========
Bulk-index a RegularizedFunctions JSONL file into Elasticsearch.

Each record is already in CLAP-normalized format, so the pipeline is:

    JSONL record  →  encode (CLAP-asm model)  →  bulk insert

Expected JSONL record format
----------------------------
    {
        "id":           "HAL_GPIO_Init|stm32f1xx_hal_gpio|O2",
        "name":         "HAL_GPIO_Init",
        "source":       "stm32f1xx_hal_gpio",
        "opt":          "O2",
        "instructions": ["push {r4 r5}", "bl EXTFUNC", "bx lr"],
        "split":        "train"
    }

CLI usage (run from project root)
----------------------------------
    python app/ingest.py                                          # uses DEFAULT_FILE
    python app/ingest.py data/RegularizedFunctions.jsonl
    python app/ingest.py data/RegularizedFunctions.jsonl --batch 64

Programmatic usage
------------------
    from app.ingest import ingest
    total = ingest(Path("data/RegularizedFunctions.jsonl"), batch_size=32)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.encoder import encode
from elastic.index import create_index
from elastic.insert import bulk_insert

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_FILE  = "data/RegularizedFunctions.jsonl"  # default JSONL to ingest
DEFAULT_BATCH = 64                                 # records encoded before each bulk insert
# ─────────────────────────────────────────────────────────────────────────────


def iter_jsonl(path: Path):
    """Yield parsed dicts from a JSONL file, skipping blank lines.

    Parameters
    ----------
    path : Path
        Path to the .jsonl file.

    Yields
    ------
    dict
        One parsed JSON object per non-empty line.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ingest(path: Path, batch_size: int = DEFAULT_BATCH) -> int:
    """Encode and index all records in a JSONL file.

    Reads records one at a time, encodes each function, accumulates them
    into batches, and flushes each batch to Elasticsearch via bulk insert.

    Parameters
    ----------
    path : Path
        Path to the RegularizedFunctions JSONL file.
    batch_size : int
        Number of functions to encode before each bulk insert.
        Larger values use more memory but fewer round-trips.

    Returns
    -------
    int
        Total number of functions successfully indexed.
    """
    total = 0
    batch = []

    for record in iter_jsonl(path):
        instructions = record.get("instructions", [])
        if not instructions:
            continue

        embedding = encode(instructions).tolist()
        batch.append({
            "function_id": record["id"],
            "name":        record["name"],
            "source":      record.get("source", ""),
            "opt":         record.get("opt", ""),
            "offset":      0,
            "size_bytes":  0,
            "instructions": instructions,
            "embedding":   embedding,
        })

        if len(batch) >= batch_size:
            bulk_insert(batch)
            total += len(batch)
            print(f"  inserted {total} so far...")
            batch = []

    if batch:
        bulk_insert(batch)
        total += len(batch)

    return total


def main(args: list[str]) -> None:
    """Parse CLI arguments and run the ingest pipeline."""
    path       = Path(DEFAULT_FILE)
    batch_size = DEFAULT_BATCH

    i = 0
    while i < len(args):
        if args[i] == "--batch" and i + 1 < len(args):
            batch_size = int(args[i + 1])
            i += 2
        else:
            path = Path(args[i])
            i += 1

    create_index()
    print(f"Ingesting : {path}")
    print(f"Batch size: {batch_size}\n")
    total = ingest(path, batch_size)
    print(f"\nDone — {total} functions indexed.")


if __name__ == "__main__":
    main(sys.argv[1:])
