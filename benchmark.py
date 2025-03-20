import gc
import time

import pandas as pd
import polars as pl
from memory_profiler import profile

from chunk_df import chunk_df_generator, chunk_df_optimized, chunk_df_polars


def create_large_df(size=10_000_000):
    """Create a large DataFrame for benchmarking."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=1000, freq="s")
    df = pd.DataFrame({"dt": dates.repeat(size // 1000)})
    return df


@profile
def measure_generator_version(df, min_chunk_size):
    """Measure memory usage for chunk_df_generator."""
    return list(chunk_df_generator(df, min_chunk_size))


@profile
def measure_optimized_version(df, min_chunk_size):
    """Measure memory usage for chunk_df_optimized."""
    chunk_count = 0
    total_rows = 0
    for chunk in chunk_df_optimized(df, min_chunk_size):
        chunk_count += 1
        total_rows += len(chunk)
    return chunk_count, total_rows


@profile
def measure_polars_version(df, min_chunk_size):
    """Measure memory usage for chunk_df_polars."""
    polars_df = pl.from_pandas(df)
    chunk_count = 0
    total_rows = 0
    for chunk in chunk_df_polars(polars_df, min_chunk_size):
        chunk_count += 1
        total_rows += len(chunk)
    return chunk_count, total_rows


def benchmark(min_chunk_size=10000):
    """Run benchmark for all chunking methods."""
    df = create_large_df()
    print(
        f"Benchmarking with {len(df)} rows and min_chunk_size={min_chunk_size}\n")

    gc.collect()

    # Generator version (pandas)
    start_time = time.time()
    chunks_gen = measure_generator_version(df, min_chunk_size)
    gen_time = time.time() - start_time
    print(f"Generator version: {len(chunks_gen)} chunks, "
          f"Avg chunk size: {len(df) / len(chunks_gen):.0f}, "
          f"Time: {gen_time:.4f} seconds")
    del chunks_gen
    gc.collect()

    # Optimized version (pandas + numpy)
    start_time = time.time()
    chunk_count, total_rows = measure_optimized_version(df, min_chunk_size)
    opt_time = time.time() - start_time
    avg_chunk_size = total_rows / chunk_count if chunk_count > 0 else 0
    print(f"Optimized version: {chunk_count} chunks, "
          f"Avg chunk size: {avg_chunk_size:.0f}, "
          f"Time: {opt_time:.4f} seconds")
    gc.collect()

    # Polars version
    start_time = time.time()
    chunk_count, total_rows = measure_polars_version(df, min_chunk_size)
    polars_time = time.time() - start_time
    avg_chunk_size = total_rows / chunk_count if chunk_count > 0 else 0
    print(f"Polars version: {chunk_count} chunks, "
          f"Avg chunk size: {avg_chunk_size:.0f}, "
          f"Time: {polars_time:.4f} seconds")
    gc.collect()


if __name__ == "__main__":
    print("Benchmark with min_chunk_size=10000")
    benchmark(min_chunk_size=10000)
    print("\n" + "=" * 100 + "\n")
    print("\n\nBenchmark with min_chunk_size=100")
    benchmark(min_chunk_size=100)
