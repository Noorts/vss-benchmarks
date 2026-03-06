import argparse
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROW_GROUP_SIZE = 122_880
NUM_ROW_GROUPS = 2
DEFAULT_NUM_ROWS = ROW_GROUP_SIZE * NUM_ROW_GROUPS


def convert_hdf5_slice_to_parquet(
    input_file: str,
    output_file: str,
    dataset_name: str | None = None,
    num_rows: int = DEFAULT_NUM_ROWS,
    chunk_size: int = 50_000,
) -> None:
    """
    Extract the first `num_rows` rows from an HDF5 embeddings dataset and write them
    to a parquet file with columns:
      - id  (int64, 0-based, contiguous)
      - emb (list<float32>, one list per embedding)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {input_path}")

    with h5py.File(input_path, "r") as f:
        # If no dataset name is provided, pick the first dataset at the root.
        if dataset_name is None:
            try:
                dataset_name = next(iter(f.keys()))
            except StopIteration:
                raise ValueError(
                    f"No datasets found in HDF5 file: {input_path}"
                ) from None

        if dataset_name not in f:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in HDF5 file {input_path}. "
                f"Available datasets: {list(f.keys())}"
            )

        dataset = f[dataset_name]
        total_embeddings, dims = dataset.shape

        rows_to_take = min(num_rows, total_embeddings)

        print(
            f"Reading {rows_to_take} / {total_embeddings} embeddings "
            f"from dataset '{dataset_name}' with {dims} dimensions"
        )

        # Schema is compatible with what VectorDBBench expects.
        schema = pa.schema(
            [
                ("id", pa.int64()),
                ("emb", pa.list_(pa.float32())),
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pq.ParquetWriter(output_path, schema) as writer:
            written = 0
            while written < rows_to_take:
                start = written
                end = min(start + chunk_size, rows_to_take)

                chunk_embeddings = dataset[start:end]
                ids = np.arange(start, end, dtype=np.int64)

                # Ensure emb is stored as list<float32>
                embeddings = [row.astype(np.float32) for row in chunk_embeddings]

                batch = pa.record_batch([ids, embeddings], schema=schema)
                writer.write_batch(batch)

                written = end

        print(
            f"Finished writing {rows_to_take} vectors "
            f"({NUM_ROW_GROUPS} row groups of {ROW_GROUP_SIZE} rows conceptually) "
            f"to {output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 4×122,880 rows (or fewer if the file is smaller) from an HDF5 "
            "embeddings file and write them to a parquet file with 'id' and 'emb' columns."
        )
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the source HDF5 file (default: None).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Name of the dataset within the HDF5 file. "
            "If omitted, the first dataset at the root is used."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="files/test.parquet",
        help="Path to the output parquet file (default: files/test.parquet).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=DEFAULT_NUM_ROWS,
        help=(
            "Number of rows to extract (default: 4×122,880). "
            "If the dataset has fewer rows, all available rows are used."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=122_880,
        help="Number of rows to process per chunk when writing parquet.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_hdf5_slice_to_parquet(
        input_file=args.input_file,
        output_file=args.output_file,
        dataset_name=args.dataset_name,
        num_rows=args.num_rows,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
