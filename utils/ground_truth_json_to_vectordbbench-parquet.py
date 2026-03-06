import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def convert_json_to_parquet(input_file, output_file):
    """
    Convert ground truth JSON file to parquet format for VectorDBBench.

    The JSON file should have query IDs as keys and arrays of neighbor IDs as values.
    The output parquet file will have:
    - id: query vector IDs (int64)
    - neighbors_id: array of neighbor IDs (list of int64)
    """
    print(f"Loading JSON file: {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)

    total_queries = len(data)
    print(f"Processing {total_queries} query vectors")

    # Schema based on VectorDBBench ground truth format
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("neighbors_id", pa.list_(pa.int64())),
        ]
    )

    # Extract IDs and neighbors
    ids = []
    neighbors = []

    # Sort keys to ensure consistent ordering (keys are strings like "0", "1", etc.)
    sorted_keys = sorted(data.keys(), key=lambda x: int(x))

    for key in tqdm(sorted_keys, desc="Processing queries", unit=" queries"):
        query_id = int(key)
        neighbor_ids = [int(nid) for nid in data[key]]

        ids.append(query_id)
        neighbors.append(neighbor_ids)

    # Create record batch and write to parquet
    print(f"Writing to {output_file}")
    batch = pa.record_batch([ids, neighbors], schema=schema)

    with pq.ParquetWriter(output_file, schema) as writer:
        writer.write_batch(batch)

    print(f"Finished writing {total_queries} query vectors to {output_file}")


def __main__():
    convert_json_to_parquet(
        "files/arxiv-for-fanns-ground-truth-r.json",
        "files/arxiv-for-fanns-1024-euclidean/neighbors_r.parquet",
    )


if __name__ == "__main__":
    __main__()
