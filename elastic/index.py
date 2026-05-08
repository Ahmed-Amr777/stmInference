from connect import get_client

INDEX_NAME = "functions_index"
VECTOR_DIM = 768

def create_index():
    es = get_client()

    if es.indices.exists(index=INDEX_NAME):
        print("Index already exists")
        return

    mapping = {
        "mappings": {
            "properties": {

                # identity
                "function_id": {"type": "keyword"},
                "name": {"type": "keyword"},

                # metadata
                "offset": {"type": "integer"},
                "size_bytes": {"type": "integer"},
                "num_instructions": {"type": "integer"},

                # raw assembly
                "instructions": {"type": "text"},

                # embedding (CLAP vector)
                "embedding": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, mappings=mapping["mappings"])
    print("Index created")