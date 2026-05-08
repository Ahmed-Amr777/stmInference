from connect import get_client
from index import INDEX_NAME


def insert_function(function_id: str, name: str, offset: int, size_bytes: int,
                    instructions: list[str], embedding: list[float]) -> None:
    es = get_client()
    doc = {
        "function_id": function_id,
        "name": name,
        "offset": offset,
        "size_bytes": size_bytes,
        "num_instructions": len(instructions),
        "instructions": " ".join(instructions),
        "embedding": embedding,
    }
    es.index(index=INDEX_NAME, document=doc)


def bulk_insert(functions: list[dict]) -> None:
    """Insert a list of function dicts in one bulk request.

    Required keys: function_id, name, offset, size_bytes, instructions, embedding.
    Any extra keys (e.g. source, opt) are stored as-is.
    """
    from elasticsearch.helpers import bulk
    es = get_client()
    actions = []
    for fn in functions:
        instrs = fn["instructions"]
        source = {
            "function_id": fn["function_id"],
            "name": fn["name"],
            "offset": fn["offset"],
            "size_bytes": fn["size_bytes"],
            "num_instructions": len(instrs),
            "instructions": " ".join(instrs) if isinstance(instrs, list) else instrs,
            "embedding": fn["embedding"],
        }
        for key, val in fn.items():
            if key not in source:
                source[key] = val
        actions.append({"_index": INDEX_NAME, "_source": source})
    bulk(es, actions)
