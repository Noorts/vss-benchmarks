import h5py
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

ROW_GROUP_SIZE = 122880
MAX_FILE_SIZE_GB = 4
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 2**30


def convert_hdf5_to_parquet(input_file, dataset_name, output_file, chunk_size=50_000):
    with h5py.File(input_file, "r") as f:
        dataset = f[dataset_name]
        total_embeddings, dims = dataset.shape
        print(f"Processing {total_embeddings} embeddings with {dims} dimensions")

        # Schema is based on what VectorDBBench expects.
        schema = pa.schema(
            [
                ("id", pa.int64()),
                ("emb", pa.list_(pa.float32())),
            ]
        )

        with pq.ParquetWriter(output_file, schema) as writer:
            with tqdm(
                total=total_embeddings, desc="Writing", unit=" embeddings"
            ) as tqdm_progress_bar:
                for start in range(0, total_embeddings, chunk_size):
                    end = min(start + chunk_size, total_embeddings)
                    rows_in_chunk = end - start

                    chunk_embeddings = dataset[start:end]

                    ids = np.arange(start, end, dtype=np.int64)
                    embeddings = [row.astype(np.float32) for row in chunk_embeddings]

                    batch = pa.record_batch([ids, embeddings], schema=schema)
                    writer.write_batch(batch)

                    tqdm_progress_bar.update(rows_in_chunk)

    print(f"Finished writing {total_embeddings} vectors to {output_file}")

    # Check if file needs to be split
    file_size = os.path.getsize(output_file)
    if file_size > MAX_FILE_SIZE_BYTES:
        print(
            f"File size ({file_size / (1024**3):.2f} GB) exceeds {MAX_FILE_SIZE_GB} GB, splitting..."
        )
        split_parquet_file(output_file, total_embeddings)
    else:
        print(f"File size ({file_size / (1024**3):.2f} GB) is within limit")


def split_parquet_file(parquet_file, total_rows):
    """Split a parquet file into multiple files if it exceeds 4GB.
    Uses multiples of ROW_GROUP_SIZE for splitting.
    """
    # Read the parquet file
    table = pq.read_table(parquet_file)
    schema = table.schema

    # Get actual file size to estimate bytes per row more accurately
    file_size = os.path.getsize(parquet_file)
    bytes_per_row = file_size / total_rows

    # Calculate how many rows per file to stay under 4GB
    max_rows_per_file = int(MAX_FILE_SIZE_BYTES / bytes_per_row)

    # Round down to nearest multiple of ROW_GROUP_SIZE
    rows_per_file = (max_rows_per_file // ROW_GROUP_SIZE) * ROW_GROUP_SIZE

    if rows_per_file < ROW_GROUP_SIZE:
        rows_per_file = ROW_GROUP_SIZE

    # Calculate number of files needed
    num_files = (total_rows + rows_per_file - 1) // rows_per_file

    print(f"Splitting into {num_files} files with {rows_per_file} rows per file")

    # Get the base path and filename
    parquet_path = Path(parquet_file)
    base_dir = parquet_path.parent
    base_name = parquet_path.stem  # e.g., "train" or "test"
    extension = parquet_path.suffix  # ".parquet"

    # Determine padding width (e.g., 01, 02, 10, 100)
    # Use at least 2 digits to match README format (train-01-of-10.parquet)
    padding_width = max(2, len(str(num_files - 1)))

    # Split and write files
    for file_idx in range(num_files):
        start_row = file_idx * rows_per_file
        end_row = min(start_row + rows_per_file, total_rows)

        # Create output filename: train-00-of-10.parquet
        output_filename = f"{base_name}-{str(file_idx).zfill(padding_width)}-of-{num_files}{extension}"
        output_path = base_dir / output_filename

        # Slice the table for this file
        file_table = table.slice(start_row, end_row - start_row)

        # Write the split file
        with pq.ParquetWriter(output_path, schema) as writer:
            writer.write_table(file_table)

        file_size = os.path.getsize(output_path)
        print(
            f"Created {output_filename} with {end_row - start_row} rows ({file_size / (1024**3):.2f} GB)"
        )

    # Remove the original file
    os.remove(parquet_file)
    print(f"Removed original file {parquet_file}")


def __main__():
    convert_hdf5_to_parquet(
        "files/input/openai-1536-angular.hdf5", "train", "files/train.parquet"
    )
    # convert_hdf5_to_parquet(
    #     "files/openai-1536-angular.hdf5", "test", "files/test.parquet"
    # )


if __name__ == "__main__":
    __main__()
