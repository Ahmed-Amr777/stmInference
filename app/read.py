"""
read.py — List all indexed functions in Elasticsearch.

Usage (run from project root):
  python app/read.py
  python app/read.py --limit 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from elastic.connect import get_client
from elastic.index import INDEX_NAME

def list_functions(limit: int = 20) -> list[dict]:
    es = get_client()
    res = es.search(
        index=INDEX_NAME,
        size=limit,
        query={"match_all": {}},
    )
    return res["hits"]["hits"]


def main(args: list[str]) -> None:
    limit = 10
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    es = get_client()
    total = es.count(index=INDEX_NAME)["count"]
    hits = list_functions(limit)

    print(f"Index : {INDEX_NAME}")
    print(f"Total : {total} docs   |   Showing: {len(hits)}")
    print("=" * 80)

    for i, hit in enumerate(hits, 1):
        s = hit["_source"]
        print(f"[{i}]")
        print(f"  function_id     : {s.get('function_id', '-')}")
        print(f"  name            : {s.get('name', '-')}")
        print(f"  source          : {s.get('source', '-')}")
        print(f"  opt             : {s.get('opt', '-')}")
        print(f"  offset          : {s.get('offset', '-')}")
        print(f"  size_bytes      : {s.get('size_bytes', '-')}")
        print(f"  num_instructions: {s.get('num_instructions', '-')}")
        instrs = s.get("instructions", "")
        if isinstance(instrs, list):
            instrs = " | ".join(instrs)
        print(f"  instructions    : {instrs}")
        emb = s.get("embedding", [])
        print(f"  embedding       : dim={len(emb)}  [{', '.join(f'{v:.6f}' for v in emb[:8])} ...]")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
