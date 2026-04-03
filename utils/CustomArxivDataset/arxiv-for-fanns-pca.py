"""
Reduce the dimensionality of arxiv-for-fanns (http://arxiv.org/abs/2507.21989) embeddings using PCA (via Faiss),
then output sampled dataset vectors and all query vectors as Parquet files.

Only NUM_EMBEDDINGS database vectors are read (via random-access seeks into the
fvecs file) so the full ~45 GB dataset never needs to fit in memory.
"""

import json

import numpy as np
import faiss
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Known dataset properties
DATABASE_COUNT = 2_735_264
DATABASE_DIMS = 4096
QUERY_COUNT = 10_000

# --- Configuration -----------------------------------------------------------
NUM_EMBEDDINGS = 1_200_000
TARGET_DIMS = 1024
RANDOM_SEED = 42

# Source files. From https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large.
DATABASE_FVECS = SCRIPT_DIR / "arxiv-for-fanns-large-database_vectors.fvecs"
QUERY_FVECS = SCRIPT_DIR / "arxiv-for-fanns-large-query_vectors.fvecs"
DATABASE_ATTRIBUTES = SCRIPT_DIR / "arxiv-for-fanns-large-database_attributes.jsonl"

# Output files
OUTPUT_DIR = SCRIPT_DIR / "files"
QUERY_OUTPUT = OUTPUT_DIR / "arxiv-for-fanns-query.parquet"
DATASET_OUTPUT = OUTPUT_DIR / "arxiv-for-fanns-train.parquet"


def read_fvecs(path: Path, expected_count: int, expected_dims: int) -> np.ndarray:
    """Read an entire .fvecs file into a float32 numpy array of shape (n, d)."""
    print(f"Reading {path.name} …")
    data = np.fromfile(path, dtype=np.float32)
    record_len = expected_dims + 1
    if data.size % record_len != 0:
        raise ValueError(
            f"File size mismatch: {data.size} floats is not divisible by "
            f"record length {record_len} (dims={expected_dims})"
        )
    n = data.size // record_len
    if n != expected_count:
        raise ValueError(f"Expected {expected_count} vectors, got {n}")
    data = data.reshape(n, record_len)
    dims_col = data[:, 0].view(np.int32)
    if not np.all(dims_col == expected_dims):
        raise ValueError("Dimension field mismatch inside fvecs file")
    vectors = data[:, 1:].copy()
    print(f"  → loaded {n:,} vectors, {expected_dims} dims")
    return vectors


def read_fvecs_subset(
    path: Path, indices: np.ndarray, total_count: int, dims: int
) -> np.ndarray:
    """Read specific vectors from an .fvecs file by seeking to each record.

    *indices* must be sorted. Returns vectors in the same order as *indices*.
    """
    n = len(indices)
    record_bytes = (1 + dims) * 4
    buf = np.empty((n, dims), dtype=np.float32)
    print(f"Reading {n:,} / {total_count:,} vectors from {path.name} …")

    with open(path, "rb") as f:
        for out_idx, vec_idx in enumerate(indices):
            f.seek(int(vec_idx) * record_bytes)
            raw = np.frombuffer(f.read(record_bytes), dtype=np.float32)
            d = raw[0].view(np.int32)
            if d != dims:
                raise ValueError(f"Vector {vec_idx}: expected dim={dims}, got {d}")
            buf[out_idx] = raw[1:]

    print(f"  → loaded {n:,} vectors, {dims} dims")
    return buf


def run_pca(combined: np.ndarray, d_out: int) -> np.ndarray:
    """Train a PCA on *combined* and return the transformed matrix."""
    d_in = combined.shape[1]
    print(f"Training PCA: {d_in} → {d_out} on {combined.shape[0]:,} vectors …")

    pca = faiss.PCAMatrix(d_in, d_out)
    pca.train(combined)

    eigenvalues = faiss.vector_to_array(pca.eigenvalues)
    if eigenvalues.size > 0:
        # Clamp to non-negative: the eigendecomposition can produce tiny
        # negative values for the bottom components due to FP imprecision.
        eigenvalues_nn = np.maximum(eigenvalues, 0.0)
        total_variance = eigenvalues_nn.sum()
        retained_variance = eigenvalues_nn[:d_out].sum()
        ratio = (
            retained_variance / total_variance if total_variance > 0 else float("nan")
        )
        print(
            f"  Eigenvalues (first {d_out}): min={eigenvalues[:d_out].min():.4f}, "
            f"max={eigenvalues[:d_out].max():.4f}"
        )
        print(
            f"  Cumulative explained variance ratio "
            f"(sum of top {d_out} / all {d_in}): {ratio:.6f} "
            f"({retained_variance:.2f} / {total_variance:.2f})"
        )
    else:
        print("  (Faiss did not expose eigenvalues for this build)")

    print("Applying PCA transform …")
    transformed = pca.apply(combined)
    return transformed


def load_attributes_subset(
    path: Path, indices: np.ndarray, total_count: int
) -> list[dict]:
    """Load a JSONL file and return only the rows at *indices* (must be sorted)."""
    print(f"Loading attributes from {path.name} …")
    all_attrs: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            all_attrs.append(json.loads(line))
    if len(all_attrs) != total_count:
        raise ValueError(f"Expected {total_count} attribute rows, got {len(all_attrs)}")
    selected = [all_attrs[i] for i in indices]
    print(f"  → selected {len(selected):,} / {total_count:,} attribute rows")
    return selected


def save_parquet(
    ids: np.ndarray,
    vectors: np.ndarray,
    path: Path,
    attributes: list[dict] | None = None,
    original_ids: np.ndarray | None = None,
) -> None:
    """Write (id, emb, …attribute columns) to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, pa.Array] = {
        "id": pa.array(ids.astype(np.int64)),
    }

    if original_ids is not None:
        columns["original_id"] = pa.array(original_ids.astype(np.int64))

    columns["emb"] = pa.array(
        [row for row in vectors.astype(np.float32)],
        type=pa.list_(pa.float32()),
    )

    if attributes:
        keys = attributes[0].keys()
        for key in keys:
            columns[key] = pa.array([row[key] for row in attributes])

    table = pa.table(columns)
    pq.write_table(table, path)
    col_names = list(columns.keys())
    print(f"  Wrote {len(ids):,} vectors ({vectors.shape[1]} dims) → {path}")
    print(f"  Columns: {col_names}")


def main() -> None:
    # Pick which database vectors to use (sorted for sequential-ish I/O).
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_indices = rng.choice(DATABASE_COUNT, size=NUM_EMBEDDINGS, replace=False)
    sampled_indices.sort()
    print(
        f"Selected {NUM_EMBEDDINGS:,} / {DATABASE_COUNT:,} database vectors "
        f"(id range {sampled_indices[0]}–{sampled_indices[-1]})"
    )

    db_vectors = read_fvecs_subset(
        DATABASE_FVECS, sampled_indices, DATABASE_COUNT, DATABASE_DIMS
    )
    query_vectors = read_fvecs(QUERY_FVECS, QUERY_COUNT, DATABASE_DIMS)

    # Combine: database first, then queries (so we know where each lives).
    combined = np.vstack([db_vectors, query_vectors])
    print(f"Combined matrix: {combined.shape[0]:,} × {combined.shape[1]}")
    del db_vectors, query_vectors

    transformed = run_pca(combined, TARGET_DIMS)
    del combined

    # Shuffle dimension order so high-variance components aren't clustered
    # at the front.  Applied to the full matrix before splitting, so dataset
    # and query vectors get the same permutation.
    dim_perm = rng.permutation(TARGET_DIMS)
    transformed = transformed[:, dim_perm]
    print(f"Shuffled dimension order (first 10 of permutation: {dim_perm[:10]})")

    # Split back into dataset and query portions.
    dataset_transformed = transformed[:NUM_EMBEDDINGS]
    query_transformed = transformed[NUM_EMBEDDINGS:]
    assert query_transformed.shape[0] == QUERY_COUNT
    del transformed

    # --- Save query vectors ---------------------------------------------------
    print("Saving query vectors …")
    query_ids = np.arange(QUERY_COUNT, dtype=np.int64)
    save_parquet(query_ids, query_transformed, QUERY_OUTPUT)
    del query_transformed

    # --- Load attributes for sampled database vectors -----------------------
    attributes = load_attributes_subset(
        DATABASE_ATTRIBUTES, sampled_indices, DATABASE_COUNT
    )

    # --- Save dataset vectors -------------------------------------------------
    print("Saving dataset vectors …")
    dataset_ids = np.arange(NUM_EMBEDDINGS, dtype=np.int64)
    save_parquet(
        dataset_ids, dataset_transformed, DATASET_OUTPUT, attributes, sampled_indices
    )

    print("Done.")


if __name__ == "__main__":
    main()
