"""Extract a random sample of 1000 embeddings from a train.parquet file."""

import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
SAMPLE_SIZE = 1000

input_path = sys.argv[1] if len(sys.argv) > 1 else "files/sift-128-euclidean/train.parquet"
output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace("train.parquet", "sample.parquet")

table = pq.read_table(input_path)
n = table.num_rows

rng = np.random.default_rng(SEED)
indices = rng.choice(n, size=SAMPLE_SIZE, replace=False)
indices.sort()

sampled = table.take(indices)

new_ids = pa.array(range(SAMPLE_SIZE), type=pa.int64())
original_ids = sampled.column("id")
emb_arrays = sampled.column("emb")

# Cast inner list values from double to float
emb_float = pa.compute.cast(emb_arrays, pa.list_(pa.float32()))

out_table = pa.table({
    "id": new_ids,
    "emb": emb_float,
    "original_id": original_ids,
})

pq.write_table(out_table, output_path)
print(f"Wrote {SAMPLE_SIZE} sampled embeddings to {output_path}")
