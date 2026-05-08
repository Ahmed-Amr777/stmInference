# stmInference

Vector similarity search for STM32 HAL assembly functions using **CLAP-asm** embeddings and **Elasticsearch**.

Given a binary function (raw or normalized assembly), the system finds the most similar known functions in the index — across compilers, optimization levels, and minor code variations.

---

## Architecture

```
objdump extraction
       │
       ▼
preprocessing/normaliztion.py     CLAP normalization
  - strip nop, .word, comments      (branch targets → INSTR<N>,
  - resolve branch targets            calls → EXTFUNC, .w/.n stripped)
       │
       ▼
embeddings/encoder.py             CLAP-asm fine-tuned model
  - tokenize function               → 768-d L2-normalized vector
  - forward pass (LoRA adapters)
       │
       ▼
elastic/                          Elasticsearch 8.12 kNN index
  - cosine similarity               (dense_vector, HNSW)
  - msearch for bulk queries
```

---

## Project Structure

```
├── app/
│   ├── ingest.py       Bulk-index RegularizedFunctions.jsonl
│   ├── add.py          Add raw (un-normalized) functions through full pipeline
│   ├── search.py       Single and bulk kNN search
│   └── read.py         Inspect what is stored in the index
│
├── preprocessing/
│   └── normaliztion.py CLAP normalization rules + batch support
│
├── embeddings/
│   ├── model_loader.py Loads CLAP-asm base model + LoRA adapters
│   └── encoder.py      encode(instructions) → numpy vector
│
├── elastic/
│   ├── connect.py      Elasticsearch client (localhost:9200)
│   ├── index.py        Index creation with dense_vector mapping
│   ├── insert.py       insert_function / bulk_insert
│   └── query.py        kNN search
│
├── data/
│   └── RegularizedFunctions.jsonl   Pre-normalized dataset
│
├── models/
│   └── finetuned/lora_adapters/     Fine-tuned LoRA weights (not tracked)
│
└── requirements.txt
```

---

## Setup

### Option A — Docker (recommended)

#### What Docker does here

Docker packages the app and Elasticsearch into isolated containers so you don't install anything locally beyond Docker itself. `docker compose` starts both services together, wires them on a private network (`app` talks to `elasticsearch:9200`), and keeps Elasticsearch data in a named volume so it survives restarts. The `app` container is built from the `Dockerfile` — it installs all Python deps and copies the source — then you override its command to run whichever script you need.

#### Prerequisites

[Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

#### 1. Place LoRA adapters

The model weights are not tracked in git. Put them here before starting:

```
models/finetuned/lora_adapters/
```

#### 2. Build and start

```powershell
docker compose up --build
```

This builds the app image and starts Elasticsearch. Wait until you see Elasticsearch log `"started"`.

#### 3. Ingest data (run once to populate the index)

```powershell
docker compose run app python app/ingest.py
```

#### 4. Run a search

```powershell
docker compose run app python app/search.py data/gpio_query.jsonl
```

#### Stop everything

```powershell
docker compose down          # keep ES data
docker compose down -v       # also wipe the ES data volume
```

---

### Option B — Local (no Docker)

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start Elasticsearch manually

```bash
docker run -d -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

#### 3. Place LoRA adapters

```
models/finetuned/lora_adapters/
```

---

## Usage

All commands are run from the **project root**.

### Index the dataset

```bash
# uses data/RegularizedFunctions.jsonl by default
python app/ingest.py

# custom file or batch size
python app/ingest.py data/RegularizedFunctions.jsonl --batch 64
```

### Add raw (un-normalized) functions

Use this when you have objdump-extracted JSON that has not been normalized yet.

```bash
python app/add.py data/json/stm32f1xx_hal_gpio.json
python app/add.py data/json/          # whole directory
python app/add.py                     # built-in demo
```

Input format:

```json
{
  "source_object": "stm32f1xx_hal_gpio.o",
  "functions": [
    {
      "name": "HAL_GPIO_Init",
      "offset": 0,
      "size_bytes": 528,
      "instructions": [
        {"address": 0, "instruction": "stmdb\tsp!, {r4, r5, lr}"},
        {"address": 4, "instruction": "ldr\tr4, [r1, #0]"}
      ]
    }
  ]
}
```

### Search

```bash
# single function (JSON)
python app/search.py query.json
python app/search.py query.json --top_k 10

# bulk (JSONL, one function per line)
python app/search.py queries.jsonl --top_k 5

# built-in GPIO demo (no arguments)
python app/search.py
```

Single query format (`query.json`):

```json
{"instructions": ["push {r4 r5}", "bl EXTFUNC", "bx lr"]}
```

Bulk query format (`queries.jsonl`):

```jsonl
{"id": "q1", "name": "HAL_GPIO_Init",   "instructions": ["push {r4 r5}", "..."]}
{"id": "q2", "name": "HAL_GPIO_DeInit", "instructions": ["push {r4}",    "..."]}
```

### Inspect the index

```bash
python app/read.py
python app/read.py --limit 50
```

---

## Programmatic API

```python
# Single search
from app.search import search_from_instructions

results = search_from_instructions(["push {r4}", "bl EXTFUNC", "bx lr"], top_k=5)
for r in results:
    print(r["name"], f"{r['score']*100:.1f}%")

# Bulk search (one msearch round-trip)
from app.search import bulk_search

queries = [
    {"id": "q1", "name": "HAL_GPIO_Init",   "instructions": [...]},
    {"id": "q2", "name": "HAL_GPIO_DeInit", "instructions": [...]},
]
results = bulk_search(queries, top_k=5)

# Add raw functions
from app.add import add_file
from pathlib import Path
n = add_file(Path("data/json/stm32f1xx_hal_gpio.json"), batch_size=8)

# Normalize only
from preprocessing.normaliztion import normalize_function, normalize_batch
norm = normalize_function([{"address": 0, "instruction": "bl\t0x200 <fn>"}])
```

---

## Example Results

Searching for 5 STM32 HAL GPIO functions against the indexed dataset:

```
Query: HAL_GPIO_Init
  [1] HAL_GPIO_Init      100.00%  HAL_GPIO_Init|stm32f1xx_hal_gpio|O2
  [2] HAL_GPIO_Init      100.00%  HAL_GPIO_Init|stm32f1xx_hal_gpio|O2
  [3] HAL_GPIO_Init       96.09%  HAL_GPIO_Init|stm32f1xx_hal_gpio|O3
  [4] HAL_GPIO_Init       96.09%  HAL_GPIO_Init|stm32f1xx_hal_gpio|O3
  [5] HAL_GPIO_Init       94.02%  HAL_GPIO_Init|stm32f1xx_hal_gpio|Os

Query: HAL_GPIO_WritePin
  [1] HAL_GPIO_WritePin  100.00%  HAL_GPIO_WritePin|stm32f1xx_hal_gpio|O2
  [2] HAL_GPIO_WritePin  100.00%  HAL_GPIO_WritePin|stm32f1xx_hal_gpio|O3
  [3] HAL_GPIO_WritePin  100.00%  HAL_GPIO_WritePin|stm32f1xx_hal_gpio|O2
  [4] HAL_GPIO_WritePin  100.00%  HAL_GPIO_WritePin|stm32f1xx_hal_gpio|O3
  [5] HAL_GPIO_WritePin   94.25%  HAL_GPIO_WritePin|stm32f1xx_hal_gpio|O1
```

The model correctly finds the same function across optimization levels (O0–Os) with 94–100% similarity.

---

## Requirements

- Python 3.10+
- Elasticsearch 8.12
- CUDA GPU recommended (CPU inference supported)

See `requirements.txt` for Python packages.
