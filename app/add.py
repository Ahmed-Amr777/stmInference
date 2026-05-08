"""
add.py
======
Add raw (un-normalized) functions to Elasticsearch through the full pipeline.

Pipeline:

    raw JSON  →  batch normalize  →  encode  →  bulk insert

Use this when you have objdump-extracted JSON that has NOT been through the
normalization step yet.  For already-normalized JSONL use ingest.py instead.

Expected input JSON format
--------------------------
    {
        "source_object": "stm32f1xx_hal_gpio.o",
        "functions": [
            {
                "name":        "HAL_GPIO_Init",
                "offset":      0,
                "size_bytes":  528,
                "instructions": [
                    {"address": 0,  "instruction": "stmdb\\tsp!, {r4, r5, lr}"},
                    {"address": 4,  "instruction": "ldr\\tr4, [r1, #0]"},
                    ...
                ]
            }
        ]
    }

CLI usage (run from project root)
----------------------------------
    python app/add.py data/json/stm32f1xx_hal_gpio.json
    python app/add.py data/json/               # all *.json in directory
    python app/add.py data/json/ --batch 8

    With no arguments → runs the built-in demo on two hardcoded functions.

Programmatic usage
------------------
    from app.add import add_file
    n = add_file(Path("data/json/stm32f1xx_hal_gpio.json"), batch_size=8)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.normaliztion import normalize_batch
from embeddings.encoder import encode
from elastic.index import create_index
from elastic.insert import bulk_insert

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_BATCH = 16   # functions normalized + encoded per bulk insert round-trip
# ─────────────────────────────────────────────────────────────────────────────


def add_file(path: Path, batch_size: int = DEFAULT_BATCH) -> int:
    """Normalize, encode, and index all functions from one raw extraction JSON.

    Processes functions in chunks of `batch_size` so memory stays bounded for
    large files.

    Parameters
    ----------
    path : Path
        Path to the raw extraction JSON file.
    batch_size : int
        Number of functions to normalize and encode per bulk insert round-trip.

    Returns
    -------
    int
        Number of functions successfully indexed.
    """
    data   = json.loads(path.read_text(encoding="utf-8"))
    source = data["source_object"]
    functions = data["functions"]

    if not functions:
        return 0

    total = 0
    for i in range(0, len(functions), batch_size):
        chunk = functions[i: i + batch_size]

        raw_lists  = [fn["instructions"] for fn in chunk]
        normalized = normalize_batch(raw_lists)

        docs = []
        for fn, norm_instrs in zip(chunk, normalized):
            if not norm_instrs:
                continue
            embedding = encode(norm_instrs).tolist()
            docs.append({
                "function_id": f"{source}|{fn['name']}|{fn['offset']}",
                "name":        fn["name"],
                "offset":      fn["offset"],
                "size_bytes":  fn["size_bytes"],
                "source":      source,
                "instructions": norm_instrs,
                "embedding":   embedding,
            })

        if docs:
            bulk_insert(docs)
            total += len(docs)
            print(f"  {path.name}: {total} inserted so far...")

    return total


def main(args: list[str]) -> None:
    """Parse CLI arguments and run add_file on each target."""
    batch_size = DEFAULT_BATCH
    targets    = []

    i = 0
    while i < len(args):
        if args[i] == "--batch" and i + 1 < len(args):
            batch_size = int(args[i + 1])
            i += 2
        else:
            targets.append(args[i])
            i += 1

    if not targets:
        print("Usage: python app/add.py <file.json|directory> ... [--batch N]")
        sys.exit(1)

    paths: list[Path] = []
    for t in targets:
        p = Path(t)
        paths.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])

    create_index()
    grand_total = 0
    for p in paths:
        n = add_file(p, batch_size)
        print(f"  {p.name}: {n} functions added")
        grand_total += n

    print(f"\nDone — {grand_total} functions added across {len(paths)} file(s).")


def demo() -> None:
    """Built-in demo: normalize two raw GPIO functions, insert, then search.

    Uses hardcoded objdump-style instructions so no external file is needed.
    Demonstrates the full pipeline: normalize → encode → insert → verify with search.
    """
    from elastic.connect import get_client
    from elastic.index import INDEX_NAME

    raw_functions = [
        {
            "name": "DEMO_GPIO_WritePin",
            "source_object": "demo",
            "offset": 0,
            "size_bytes": 16,
            "instructions": [
                {"address": 0x00, "instruction": "push\t{r7, lr}"},
                {"address": 0x02, "instruction": "add\tr7, sp, #0"},
                {"address": 0x04, "instruction": "str\tr0, [r7, #4]"},
                {"address": 0x06, "instruction": "str\tr1, [r7, #0]"},
                {"address": 0x08, "instruction": "bl\t0x200 <HAL_GPIO_WritePin>"},
                {"address": 0x0c, "instruction": "pop\t{r7, pc}"},
            ],
        },
        {
            "name": "DEMO_GPIO_ReadPin",
            "source_object": "demo",
            "offset": 16,
            "size_bytes": 12,
            "instructions": [
                {"address": 0x10, "instruction": "push\t{r7, lr}"},
                {"address": 0x12, "instruction": "add\tr7, sp, #0"},
                {"address": 0x14, "instruction": "bl\t0x300 <HAL_GPIO_ReadPin>"},
                {"address": 0x18, "instruction": "cbz\tr0, 0x1e <DEMO_GPIO_ReadPin+0xe>"},
                {"address": 0x1a, "instruction": "movs\tr0, #1"},
                {"address": 0x1e, "instruction": "pop\t{r7, pc}"},
            ],
        },
    ]

    # Step 1 — normalize
    print("Step 1 — Normalize")
    raw_lists  = [fn["instructions"] for fn in raw_functions]
    normalized = normalize_batch(raw_lists)
    for fn, norm in zip(raw_functions, normalized):
        print(f"  {fn['name']}: {norm}")

    # Step 2 — encode + insert
    print("\nStep 2 — Encode & insert")
    create_index()
    docs = []
    for fn, norm_instrs in zip(raw_functions, normalized):
        if not norm_instrs:
            continue
        embedding = encode(norm_instrs).tolist()
        docs.append({
            "function_id": f"demo|{fn['name']}|{fn['offset']}",
            "name":        fn["name"],
            "offset":      fn["offset"],
            "size_bytes":  fn["size_bytes"],
            "source":      "demo",
            "instructions": norm_instrs,
            "embedding":   embedding,
        })
        print(f"  {fn['name']}: encoded dim={len(embedding)}")

    bulk_insert(docs)
    get_client().indices.refresh(index=INDEX_NAME)
    print(f"  Inserted {len(docs)} docs")

    # Step 3 — search to verify
    print("\nStep 3 — Search (verify round-trip)")
    from app.search import search_from_instructions
    for fn, norm_instrs in zip(raw_functions, normalized):
        if not norm_instrs:
            continue
        results = search_from_instructions(norm_instrs, top_k=3)
        print(f"\n  Query: {fn['name']}")
        for rank, r in enumerate(results, 1):
            pct = r["score"] * 100
            print(f"    [{rank}] {r['name']:<35}  {pct:6.2f}%  id={r['function_id']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        main(sys.argv[1:])
