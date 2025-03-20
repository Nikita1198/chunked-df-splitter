import pandas as pd
import pytest

from chunk_df import chunk_df_generator


@pytest.fixture
def sample_df():
    """Fixture providing a sample DataFrame with datetime 'dt' column."""
    df = pd.DataFrame({
        "dt": [
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:03"
        ]
    })
    df['dt'] = pd.to_datetime(df['dt'])
    return df


@pytest.fixture
def df_with_nulls():
    """Fixture providing a DataFrame with None"""
    df = pd.DataFrame({
        "dt": ["2023-01-01 00:00:01", None, "2023-01-01 00:00:02", pd.NaT]
    })
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    return df


@pytest.fixture
def df_with_extra_cols():
    """Fixture providing a DataFrame with extra columns"""
    df = pd.DataFrame({
        "dt": ["2023-01-01 00:00:01", "2023-01-01 00:00:02"],
        "value": [42, 73],
        "text": ["a", "b"]
    })
    df['dt'] = pd.to_datetime(df['dt'])
    return df


@pytest.fixture
def empty_df():
    """Fixture providing an empty DataFrame."""
    return pd.DataFrame(columns=["dt"])


@pytest.fixture
def single_group_df():
    """Fixture providing a DataFrame with a single datetime group."""
    df = pd.DataFrame({"dt": ["2023-01-01 00:00:01"] * 5})
    df['dt'] = pd.to_datetime(df['dt'])
    return df


def test_basic_chunking(sample_df):
    """Test basic chunking with min_chunk_size=1."""
    chunks = list(chunk_df_generator(sample_df, 1))
    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [2, 3, 1]
    assert all(len(chunk['dt'].unique()) == 1 for chunk in chunks)


def test_min_chunk_size(sample_df):
    """Test chunking with min_chunk_size=3."""
    chunks = list(chunk_df_generator(sample_df, 3))
    assert len(chunks) == 2
    assert [len(chunk) for chunk in chunks] == [5, 1]
    assert len(chunks[0]['dt'].unique()) == 2
    assert len(chunks[1]['dt'].unique()) == 1


def test_full_size_chunking(sample_df):
    """Test chunking with min_chunk_size equal to or exceeding DataFrame size."""
    chunks = list(chunk_df_generator(sample_df, 6))
    assert len(chunks) == 1
    assert len(chunks[0]) == 6
    chunks = list(chunk_df_generator(sample_df, 7))
    assert len(chunks) == 1
    assert len(chunks[0]) == 6


def test_empty_df(empty_df):
    """Test chunking with an empty DataFrame."""
    chunks = list(chunk_df_generator(empty_df, 1))
    assert len(chunks) == 0


def test_single_group(single_group_df):
    """Test chunking with a single datetime group."""
    chunks = list(chunk_df_generator(single_group_df, 3))
    assert len(chunks) == 1
    assert len(chunks[0]) == 5


def test_no_date_overlap(sample_df):
    """Test that chunks have no overlapping dates."""
    chunks = list(chunk_df_generator(sample_df, 3))
    all_dates = set()
    for chunk in chunks:
        chunk_dates = set(chunk['dt'].unique())
        assert all_dates.isdisjoint(chunk_dates)
        all_dates.update(chunk_dates)


def test_unsorted_data(sample_df):
    """Test chunking with unsorted data."""
    unsorted_df = sample_df.sample(frac=1, random_state=42)
    chunks = list(chunk_df_generator(unsorted_df, 3))
    assert len(chunks) == 2
    assert all(len(chunk['dt'].unique()) <= 2 for chunk in chunks)


def test_null_values(df_with_nulls):
    """Test validate null values."""
    chunks_gen = list(chunk_df_generator(df_with_nulls, 1))
    assert len(chunks_gen) > 0


def test_all_same_dates(single_group_df):
    """Test validate all same dates."""
    chunks_gen = list(chunk_df_generator(single_group_df, 10))
    assert len(chunks_gen) == 1
    assert len(chunks_gen[0]) == 5


def test_extra_columns(df_with_extra_cols):
    """Test validate extra columns."""
    chunks_gen = list(chunk_df_generator(df_with_extra_cols, 1))
    assert all("value" in chunk.columns and "text" in chunk.columns for chunk in
               chunks_gen)
    assert chunks_gen[0]["value"].iloc[0] == 42


def test_invalid_min_chunk_size(sample_df):
    """Test validate invalid chunk size."""
    with pytest.raises(ValueError,
                       match="min_chunk_size must be a positive integer"):
        list(chunk_df_generator(sample_df, 0))
