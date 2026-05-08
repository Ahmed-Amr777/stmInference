import json
from encoder import encode

FILE = "data/RegularizedFunctions.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    count = 0

    for item in load_jsonl(FILE):
        vec = encode(item["instructions"])

        print("\n========================")
        print("ID:", item["id"])
        print("Name:", item["name"])
        print("Vector shape:", vec.shape)
        print("First 10 values:", vec[:10])

        count += 1
        if count == 3:   # test only first 3
            break


if __name__ == "__main__":
    main()