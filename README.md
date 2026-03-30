# VSS Benchmarks

Vector similarity search (VSS) benchmarks mostly focused on DuckDB's vector
search capabilities with and without VSS indexes. Includes plain DuckDB (no
index), the [VSS extension](https://github.com/duckdb/duckdb-vss), and the new [PDXearch extension](https://github.com/noorts/pdxearch).

This repository contains:

1. Python scripts to orchestrate [a fork of VectorDBBench](https://github.com/noorts/vectordbbench) in `experiments/`. These scripts make it easy to sweep across the parameter space.
2. Measurements / results from those experiments in `experiments/results/`. The result JSON files contain a nested `db_label` property that stores the git commit hashes of the software versions used to produce the result.
3. Plotting notebooks to gain insights into the results in `analysis/`.
4. Utility scripts used to create VectorDBBench datasets in `utils/`.

The results and insights are mainly used for an MSc thesis.

## Running Benchmarks

> [!WARNING]
> The instructions below are not polished. Attempt to use these
> scripts at your own risk. It might be easier to use the
> [fork of VectorDBBench](https://github.com/noorts/vectordbbench) directly instead.

Ensure you have the prerequisites. Check the PDXearch [README](https://github.com/noorts/pdxearch) and [DEVELOPMENT](https://github.com/Noorts/PDXearch/blob/main/DEVELOPMENT.md) documents, and the VectorDBBench [README](https://github.com/noorts/vectordbbench).

1. Clone the repository

    ```sh
    git clone --recurse-submodules https://github.com/Noorts/vss-benchmarks.git
    ```

2. Change directory to the `vectordbbench` submodule and follow the instructions there to set it up (e.g., set up a `.env` file).

3. Change directory to the `PDXearch` submodule and build the extension.

    ```sh
    GEN=ninja DISABLE_SANITIZER=1 CC=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang CXX=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang++ EXTRA_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" make
    ```

4. (Optional) If you want to run pgvector, then change directory to `experiments/`, make sure to have Docker Desktop and `psql` installed, manually copy the `postgresql.conf` configuration (e.g., to `/opt/homebrew/var/postgresql@18/postgresql.conf`), run `docker compose up -d`, then initialize the pgvector database by executing `psql -h localhost -p 5432 -U postgres vectordb -c "CREATE EXTENSION vector;"`.

5. Change directory to `experiments/` and run the desired experiment using `uv run index_search.py`. Execute the notebook to plot the results.
