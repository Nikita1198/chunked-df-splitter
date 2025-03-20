import types
from datetime import datetime

import polars as pl
import pytest

from chunk_df import chunk_df_polars


@pytest.fixture
def basic_df():
    """Fixture providing a basic DataFrame with repeated datetime values."""
    df = pl.DataFrame({
        "dt": [
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:03"
        ]
    }).with_columns(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    return df


@pytest.fixture
def empty_df():
    """Fixture providing an empty DataFrame with 'dt' column."""
    return pl.DataFrame({"dt": []}, schema={"dt": pl.Datetime})


@pytest.fixture
def single_group_df():
    """Fixture providing a DataFrame with a single datetime group."""
    df = pl.DataFrame({
        "dt": ["2023-01-01 00:00:01"] * 5
    }).with_columns(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    return df


@pytest.fixture
def unsorted_df():
    """Fixture providing an unsorted DataFrame with datetime values."""
    df = pl.DataFrame({
        "dt": [
            "2023-01-01 00:00:03",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:03"
        ]
    }).with_columns(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    return df


@pytest.fixture
def nulls_df():
    """Fixture providing a DataFrame with null values in 'dt'."""
    df = pl.DataFrame({
        "dt": ["2023-01-01 00:00:01", None, "2023-01-01 00:00:02", None]
    }).with_columns(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S",
                                              strict=False))
    return df


@pytest.fixture
def multi_column_df():
    """Fixture providing a DataFrame with additional columns beyond 'dt'."""
    df = pl.DataFrame({
        "dt": ["2023-01-01 00:00:01", "2023-01-01 00:00:01",
               "2023-01-01 00:00:02"],
        "value": [10, 20, 30]
    }).with_columns(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    return df


@pytest.fixture
def large_df():
    """Fixture providing a large DataFrame for performance testing."""
    unique_dates = pl.datetime_range(
        start=datetime(2023, 1, 1, 0, 0, 0),
        end=datetime(2023, 1, 1, 0, 0, 9),
        interval="1s",
        eager=True
    )
    dt_list = [dt for dt in unique_dates for _ in range(10)]
    return pl.DataFrame({"dt": dt_list})


def assert_no_date_overlap(chunks):
    """Helper to verify no date overlaps between chunks."""
    all_dates = set()
    for chunk in chunks:
        chunk_dates = set(chunk["dt"].unique().to_list())
        assert all_dates.isdisjoint(
            chunk_dates), "Chunks contain overlapping dates"
        all_dates.update(chunk_dates)


def test_basic_chunking(basic_df):
    """Test basic chunking with min_chunk_size=1, ensuring each unique 'dt' forms a chunk."""
    chunks = list(chunk_df_polars(basic_df, 1))
    assert len(chunks) == 3, "Expected 3 chunks for unique 'dt' values"
    assert [len(chunk) for chunk in chunks] == [2, 3,
                                                1], "Chunk sizes should match group sizes"
    assert all(chunk["dt"].n_unique() == 1 for chunk in
               chunks), "Each chunk should have one unique 'dt'"


def test_min_chunk_size(basic_df):
    """Test chunking with min_chunk_size=3, combining groups to meet the size."""
    chunks = list(chunk_df_polars(basic_df, 3))
    assert len(chunks) == 2, "Expected 2 chunks when combining groups"
    assert [len(chunk) for chunk in chunks] == [5,
                                                1], "Chunk sizes should reflect combined and remaining rows"
    assert chunks[0][
               "dt"].n_unique() == 2, "First chunk should combine two 'dt' groups"
    assert chunks[1]["dt"].n_unique() == 1, "Last chunk should have one 'dt'"


@pytest.mark.parametrize("min_size", [6, 7])
def test_full_size_chunking(basic_df, min_size):
    """Test chunking when min_chunk_size equals or exceeds DataFrame size."""
    chunks = list(chunk_df_polars(basic_df, min_size))
    assert len(
        chunks) == 1, f"Expected 1 chunk when min_size={min_size} >= DataFrame size"
    assert len(chunks[0]) == 6, "Chunk should contain all 6 rows"


def test_empty_df(empty_df):
    """Test chunking with an empty DataFrame yields no chunks."""
    chunks = list(chunk_df_polars(empty_df, 1))
    assert len(chunks) == 0, "Empty DataFrame should yield no chunks"


def test_single_group(single_group_df):
    """Test chunking with a single datetime group yields one chunk."""
    chunks = list(chunk_df_polars(single_group_df, 3))
    assert len(chunks) == 1, "Expected 1 chunk for a single 'dt' group"
    assert len(chunks[0]) == 5, "Chunk should contain all 5 rows"
    assert chunks[0]["dt"].n_unique() == 1, "Chunk should have one unique 'dt'"


def test_no_date_overlap(basic_df):
    """Test that chunks have no overlapping dates."""
    chunks = list(chunk_df_polars(basic_df, 3))
    assert_no_date_overlap(chunks)


def test_unsorted_data(unsorted_df):
    """Test chunking with unsorted data sorts correctly and respects min_chunk_size."""
    chunks = list(chunk_df_polars(unsorted_df, 3))
    assert len(chunks) == 2, "Expected 2 chunks after sorting"
    assert [len(chunk) for chunk in chunks] == [4,
                                                2], "Chunk sizes should match sorted groups"
    assert_no_date_overlap(chunks)
    assert chunks[0]["dt"].min() < chunks[1][
        "dt"].min(), "Chunks should be in sorted order"


def test_missing_dt_column():
    """Test that a missing 'dt' column raises a ValueError."""
    df = pl.DataFrame({"wrong_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="DataFrame must contain 'dt' column"):
        list(chunk_df_polars(df, 1))


@pytest.mark.parametrize("invalid_size", [-1, 0])
def test_invalid_min_chunk_size(invalid_size):
    """Test that invalid min_chunk_size values raise a ValueError."""
    df = pl.DataFrame({"dt": ["2023-01-01"]}).with_columns(
        pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    with pytest.raises(ValueError,
                       match="min_chunk_size must be a positive integer"):
        list(chunk_df_polars(df, invalid_size))


def test_null_values(nulls_df):
    """Test chunking with null values in 'dt' handles them appropriately."""
    chunks = list(chunk_df_polars(nulls_df, 2))
    assert len(chunks) == 2, "Expected 2 chunks with nulls split"
    assert [len(chunk) for chunk in chunks] == [2,
                                                2], "Chunk sizes should split nulls and non-nulls"
    assert chunks[0][
               "dt"].null_count() == 2, "First chunk should have all nulls after sorting"
    assert chunks[1][
               "dt"].null_count() == 0, "Second chunk should have no nulls"


def test_single_row():
    """Test chunking a single-row DataFrame yields one chunk."""
    df = pl.DataFrame({"dt": ["2023-01-01 00:00:01"]}).with_columns(
        pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    chunks = list(chunk_df_polars(df, 1))
    assert len(chunks) == 1, "Expected 1 chunk for a single row"
    assert len(chunks[0]) == 1, "Chunk should contain 1 row"


def test_all_same_date():
    """Test chunking when all rows have the same 'dt' value yields one chunk."""
    df = pl.DataFrame({"dt": ["2023-01-01"] * 10}).with_columns(
        pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    chunks = list(chunk_df_polars(df, 5))
    assert len(chunks) == 1, "Expected 1 chunk for identical 'dt' values"
    assert len(chunks[0]) == 10, "Chunk should contain all 10 rows"


def test_multi_column_preservation(multi_column_df):
    """Test that additional columns are preserved in chunks."""
    chunks = list(chunk_df_polars(multi_column_df, 1))
    assert len(chunks) == 2, "Expected 2 chunks for two 'dt' groups"
    assert chunks[0].columns == ["dt",
                                 "value"], "All columns should be preserved"
    assert chunks[0]["value"].to_list() == [10,
                                            20], "Values in first chunk should match"
    assert chunks[1]["value"].to_list() == [
        30], "Values in second chunk should match"


def test_large_data(large_df):
    """Test chunking a large DataFrame ensures correct sizes and no overlaps."""
    chunks = list(chunk_df_polars(large_df, 25))
    assert all(len(chunk) >= 25 for chunk in chunks[
                                             :-1]), "All chunks except the last should meet min_chunk_size"
    assert len(chunks[
                   -1]) <= 25, "Last chunk should be smaller or equal to min_chunk_size if remaining"
    assert sum(len(chunk) for chunk in
               chunks) == 100, "Total rows should match original"
    assert_no_date_overlap(chunks)


def test_exact_chunk_boundary(basic_df):
    """Test chunking where group sizes align with min_chunk_size."""
    chunks = list(chunk_df_polars(basic_df, 2))
    assert len(chunks) == 3, "Expected 3 chunks with exact boundaries"
    assert [len(chunk) for chunk in chunks] == [2, 3,
                                                1], "Chunk sizes should match groups"
    assert_no_date_overlap(chunks)


@pytest.mark.parametrize("dt_type", [
    pl.Int32,
    pl.Utf8
])
def test_sortable_dt_types(dt_type):
    """Test chunking with non-datetime sortable 'dt' types."""
    if dt_type == pl.Int32:
        df = pl.DataFrame({"dt": [1, 1, 2, 2, 3]})
    else:
        df = pl.DataFrame({"dt": ["a", "a", "b", "b", "c"]})
    chunks = list(chunk_df_polars(df, 2))
    assert len(
        chunks) == 3, f"Expected 3 chunks for {dt_type} type with min_chunk_size=2"
    assert [len(chunk) for chunk in chunks] == [2, 2,
                                                1], f"Chunk sizes should match groups for {dt_type}"
    assert_no_date_overlap(chunks)


def test_immutability(basic_df):
    """Test that the original DataFrame remains unchanged."""
    original_df = basic_df.clone()
    chunks = list(chunk_df_polars(basic_df, 1))
    assert basic_df.equals(
        original_df), "Original DataFrame should not be modified"
    assert len(chunks) == 3, "Chunking should still work as expected"


@pytest.mark.slow
def test_memory_efficiency(large_df):
    """Test memory efficiency by ensuring generator yields chunks lazily."""
    chunks_iter = chunk_df_polars(large_df, 20)
    first_chunk = next(chunks_iter)  # Only first chunk materialized
    assert len(first_chunk) >= 20, "First chunk should meet min_chunk_size"
    assert isinstance(chunks_iter,
                      types.GeneratorType), "Should return a generator"
