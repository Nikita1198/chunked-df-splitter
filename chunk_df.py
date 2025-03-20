import numpy as np
import pandas as pd
import polars as pl


def chunk_df_generator(df: pd.DataFrame, min_chunk_size: int):
    """
    Generator that splits a DataFrame into chunks based on the 'dt' column, ensuring
    each chunk has at least the specified minimum size. Each date is fully contained within a single chunk.

    Args:
        df: DataFrame with a 'dt' column (datetime or sortable type)
        min_chunk_size: Minimum chunk size (>=1)

    Yields:
        pd.DataFrame: Chunks of the DataFrame with unique dates

    Raises:
        ValueError: If the 'dt' column is missing or min_chunk_size is less than 1
    """
    if 'dt' not in df.columns:
        raise ValueError("DataFrame must contain a 'dt' column")
    if min_chunk_size < 1:
        raise ValueError("min_chunk_size must be a positive integer")
    if df.empty:
        return

    if not df['dt'].is_monotonic_increasing:
        df = df.sort_values('dt')

    group_starts = df['dt'].ne(df['dt'].shift()).to_numpy().nonzero()[0]
    group_starts = list(group_starts) + [len(df)]

    chunk_start = 0
    current_size = 0

    for i in range(1, len(group_starts)):
        group_size = group_starts[i] - group_starts[i - 1]
        current_size += group_size

        if current_size >= min_chunk_size:
            yield df.iloc[chunk_start:group_starts[i]]
            chunk_start = group_starts[i]
            current_size = 0

    if chunk_start < len(df):
        yield df.iloc[chunk_start:len(df)]


def chunk_df_optimized(df: pd.DataFrame, min_chunk_size: int):
    """
    Splits a DataFrame into chunks with a minimum size and unique dates.
    Each date (dt) is fully contained within a single chunk. Once the cumulative
    chunk size reaches or exceeds min_chunk_size, the chunk is yielded. Remaining
    data (less than min_chunk_size) is included in the final chunk.

    Args:
        df: DataFrame with a 'dt' column (datetime or sortable type)
        min_chunk_size: Minimum chunk size (>=1)

    Yields:
        pd.DataFrame: Chunks of the DataFrame with unique dates

    Raises:
        ValueError: If the 'dt' column is missing or min_chunk_size is less than 1
    """
    if 'dt' not in df.columns:
        raise ValueError("DataFrame must contain 'dt' column")
    if min_chunk_size < 1:
        raise ValueError("min_chunk_size must be positive")
    if df.empty:
        return

    if not df['dt'].is_monotonic_increasing:
        df = df.sort_values('dt')

    dt_array = df['dt'].to_numpy()
    boundary = np.empty(len(dt_array), dtype=bool)
    boundary[0] = True
    boundary[1:] = dt_array[1:] != dt_array[:-1]
    group_starts = np.flatnonzero(boundary)
    group_starts = np.append(group_starts, len(dt_array))

    chunk_start_idx = 0
    chunk_size = 0

    for i in range(len(group_starts) - 1):
        group_len = group_starts[i + 1] - group_starts[i]
        chunk_size += group_len
        if chunk_size >= min_chunk_size:
            yield df.iloc[chunk_start_idx:group_starts[i + 1]]
            chunk_start_idx = group_starts[i + 1]
            chunk_size = 0

    if chunk_start_idx < len(df):
        yield df.iloc[chunk_start_idx:]


def chunk_df_polars(df: pl.DataFrame, min_chunk_size: int):
    """
    A generator that splits a Polars DataFrame into chunks based on the 'dt' column,
    ensuring each chunk has at least the specified minimum size.
    Each date is fully contained within a single chunk.

    Args:
        df: Polars DataFrame with a 'dt' column (datetime or sortable type)
        min_chunk_size: Minimum chunk size (>=1)

    Yields:
        pl.DataFrame: DataFrame chunks with unique dates

    Raises:
        ValueError: If 'dt' column is missing or min_chunk_size is less than 1
    """
    if 'dt' not in df.columns:
        raise ValueError("DataFrame must contain 'dt' column")
    if min_chunk_size < 1:
        raise ValueError("min_chunk_size must be a positive integer")
    if df.is_empty():
        return

    df = df.sort('dt')
    boundary = (df['dt'] != df['dt'].shift()).fill_null(True)
    group_starts = df.with_row_index().filter(boundary)['index'].to_list()
    group_starts.append(len(df))

    chunk_start = 0
    current_size = 0

    for i in range(len(group_starts) - 1):
        group_len = group_starts[i + 1] - group_starts[i]
        current_size += group_len
        if current_size >= min_chunk_size:
            yield df.slice(chunk_start, group_starts[i + 1] - chunk_start)
            chunk_start = group_starts[i + 1]
            current_size = 0

    if chunk_start < len(df):
        yield df.slice(chunk_start, len(df) - chunk_start)
