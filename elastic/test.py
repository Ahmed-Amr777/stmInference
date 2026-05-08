from index import create_index, INDEX_NAME
from insert import insert_function
from query import search
from connect import get_client

def fake_vector(seed):
    return [float(seed % 10) * 0.01] * 768


def run_test():
    print("Creating index...")
    create_index()

    print("\nInserting data...")

    insert_function(
        function_id="f1",
        name="func_a",
        offset=0,
        size_bytes=12,
        instructions=["mov r0 r1", "add r0 #1"],
        embedding=fake_vector(1),
    )

    insert_function(
        function_id="f2",
        name="func_b",
        offset=16,
        size_bytes=8,
        instructions=["push {lr}", "bl EXTFUNC", "pop {pc}"],
        embedding=fake_vector(2),
    )

    insert_function(
        function_id="f3",
        name="func_c",
        offset=32,
        size_bytes=4,
        instructions=["mov r0 r1", "bx lr"],
        embedding=fake_vector(1),  # similar to f1
    )

    get_client().indices.refresh(index=INDEX_NAME)
    print("\nSearching...")
    results = search(fake_vector(1), top_k=3)
    for r in results:
        print(r)


if __name__ == "__main__":
    run_test()
