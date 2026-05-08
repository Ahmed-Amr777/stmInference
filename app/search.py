"""
search.py
=========
Search the Elasticsearch index using normalized (CLAP) assembly instructions.

Pipeline:

    normalized instructions  →  encode (CLAP-asm model)  →  kNN search

The input must already be in CLAP format (normalized strings, not raw objdump).
To search with un-normalized functions, normalize them first via
``preprocessing.normaliztion.normalize_function``.

Query file formats
------------------
Single function (JSON):
    {"instructions": ["push {r4 r5}", "bl EXTFUNC", "bx lr"]}

Bulk functions (JSONL, one function per line):
    {"id": "fn1", "name": "HAL_GPIO_Init",   "instructions": [...]}
    {"id": "fn2", "name": "HAL_GPIO_DeInit", "instructions": [...]}

CLI usage (run from project root)
----------------------------------
    python app/search.py query.json              # single search
    python app/search.py queries.jsonl           # bulk search
    python app/search.py queries.jsonl --top_k 10
    python app/search.py                         # runs built-in GPIO demo

Programmatic usage
------------------
    from app.search import search_from_instructions, bulk_search

    # single
    results = search_from_instructions(["push {r4}", "bl EXTFUNC", "bx lr"], top_k=5)

    # bulk
    queries = [{"id": "q1", "name": "fn", "instructions": [...]}]
    results = bulk_search(queries, top_k=5)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.encoder import encode
from elastic.connect import get_client
from elastic.index import INDEX_NAME

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K      = 5    # number of nearest neighbours to return
NUM_CANDIDATES     = 100  # ES kNN num_candidates (higher = more accurate, slower)
DEMO_GPIO_FILE     = "data/stm32f1xx_hal_gpio_O2_clap.json"
DEMO_FUNCTIONS     = 7   # how many GPIO functions to use in the demo
# ─────────────────────────────────────────────────────────────────────────────


def search_from_instructions(instructions: list[str],
                              top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Encode a single normalized function and run a kNN search.

    Parameters
    ----------
    instructions : list[str]
        CLAP-normalized instruction strings, e.g. ``["push {r4}", "bl EXTFUNC"]``.
    top_k : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each result has keys: ``function_id``, ``name``, ``score``, ``instructions``.
        ``score`` is a cosine-similarity value in [0, 1]; multiply by 100 for %.
    """
    if not instructions:
        return []
    vector = encode(instructions).tolist()
    res = get_client().search(
        index=INDEX_NAME,
        size=top_k,
        knn={
            "field":        "embedding",
            "query_vector": vector,
            "k":            top_k,
            "num_candidates": NUM_CANDIDATES,
        },
    )
    return [
        {
            "function_id":  h["_source"]["function_id"],
            "name":         h["_source"]["name"],
            "score":        h["_score"],
            "instructions": h["_source"]["instructions"],
        }
        for h in res["hits"]["hits"]
    ]


def bulk_search(queries: list[dict],
                top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Encode multiple functions and search for all of them in one msearch request.

    All queries are encoded first, then sent together as a single msearch call,
    which is significantly more efficient than calling ``search_from_instructions``
    in a loop.

    Parameters
    ----------
    queries : list[dict]
        Each dict must have ``"instructions": list[str]``.
        Optional keys ``"id"`` and ``"name"`` are echoed back in the result.
    top_k : int
        Number of nearest neighbours per query.

    Returns
    -------
    list[dict]
        One entry per valid query (queries with empty instructions are skipped).
        Each entry has: ``query_id``, ``query_name``, ``hits`` (list of result dicts).
        Each hit has: ``function_id``, ``name``, ``score``, ``instructions``.
    """
    if not queries:
        return []

    es   = get_client()
    body = []
    valid_queries = []

    for q in queries:
        instrs = q.get("instructions", [])
        if not instrs:
            continue
        vector = encode(instrs).tolist()
        body.append({})
        body.append({
            "size": top_k,
            "knn": {
                "field":        "embedding",
                "query_vector": vector,
                "k":            top_k,
                "num_candidates": NUM_CANDIDATES,
            },
        })
        valid_queries.append(q)

    if not body:
        return []

    res = es.msearch(index=INDEX_NAME, searches=body)

    return [
        {
            "query_id":   q.get("id",   "-"),
            "query_name": q.get("name", "-"),
            "hits": [
                {
                    "function_id":  h["_source"]["function_id"],
                    "name":         h["_source"]["name"],
                    "score":        h["_score"],
                    "instructions": h["_source"]["instructions"],
                }
                for h in resp["hits"]["hits"]
            ],
        }
        for q, resp in zip(valid_queries, res["responses"])
    ]


def _print_bulk_results(results: list[dict]) -> None:
    for r in results:
        print(f"Query: {r['query_name']}  (id: {r['query_id']})")
        print(f"  {'Rank':<5} {'Match name':<35} {'Score':>7}  Function ID")
        print(f"  {'-' * 78}")
        if not r["hits"]:
            print("  No results.")
        for rank, h in enumerate(r["hits"], 1):
            pct = h["score"] * 100
            print(f"  [{rank}]   {h['name']:<35} {pct:6.2f}%  {h['function_id']}")
        print()


def demo_gpio(top_k: int = DEFAULT_TOP_K) -> None:
    """Built-in demo: bulk-search the first DEMO_FUNCTIONS GPIO functions.

    Loads DEMO_GPIO_FILE, takes the first DEMO_FUNCTIONS entries, encodes them,
    and runs a single msearch call against the index.

    Parameters
    ----------
    top_k : int
        Number of nearest neighbours to retrieve per query.
    """
    data = json.loads(
        (Path(__file__).parent.parent / DEMO_GPIO_FILE).read_text(encoding="utf-8")
    )
    queries = [
        {"id": fn["name"], "name": fn["name"], "instructions": fn["instructions"]}
        for fn in data["functions"][:DEMO_FUNCTIONS]
    ]
    print(f"Demo — STM32 HAL GPIO  |  {len(queries)} queries  |  top_k={top_k}\n")
    _print_bulk_results(bulk_search(queries, top_k=top_k))


def main(args: list[str]) -> None:
    """Parse CLI arguments and dispatch to single or bulk search."""
    top_k = DEFAULT_TOP_K
    path  = None

    i = 0
    while i < len(args):
        if args[i] == "--top_k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        else:
            path = args[i]
            i += 1

    if path is None:
        print("Usage: python app/search.py <query.json|queries.jsonl> [--top_k N]")
        sys.exit(1)

    p    = Path(path)
    text = p.read_text(encoding="utf-8").strip()

    # JSONL → bulk
    if p.suffix == ".jsonl":
        queries = [json.loads(line) for line in text.splitlines() if line.strip()]
        print(f"Bulk search: {len(queries)} queries, top_k={top_k}\n")
        _print_bulk_results(bulk_search(queries, top_k=top_k))

    # JSON → single
    else:
        data    = json.loads(text)
        results = search_from_instructions(data["instructions"], top_k=top_k)
        if not results:
            print("No results.")
            return
        print(f"Top {top_k} matches:\n")
        print(f"  {'Rank':<5} {'Match name':<35} {'Score':>7}  Function ID")
        print(f"  {'-' * 78}")
        for rank, r in enumerate(results, 1):
            pct = r["score"] * 100
            print(f"  [{rank}]   {r['name']:<35} {pct:6.2f}%  {r['function_id']}")
            print(f"         instrs: {r['instructions'][:80]}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo_gpio()
    else:
        main(sys.argv[1:])
