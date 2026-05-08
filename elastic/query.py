from connect import get_client
from index import INDEX_NAME

es = get_client()


def search(vector: list[float], top_k: int = 5) -> list[dict]:
    res = es.search(
        index=INDEX_NAME,
        size=top_k,
        knn={
            "field": "embedding",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": 100,
        },
    )
    return [
        {
            "function_id": hit["_source"]["function_id"],
            "name": hit["_source"]["name"],
            "score": hit["_score"],
            "instructions": hit["_source"]["instructions"],
        }
        for hit in res["hits"]["hits"]
    ]
