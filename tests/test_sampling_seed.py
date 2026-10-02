import json

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from madclean.components.dataprofiler.data_selection import DataSampler
from madclean.config.loader import load_default_cleaning_config
from madclean.utils.helpers import seeded_generator, seeded_random

# Small sizes so that every generated column is larger than the sample and sampling really chooses.
SMALL_SIZES = {
    "NUMERIC": {"clean_sample_size": 5, "dirty_sample_size": 3},
    "DATETIME": {"clean_sample_size": 5, "dirty_sample_size": 3},
    "DIRTY_NUMERIC": {"random_sample_size": 5, "unique_sample_size": 3},
    "STRING": {"random_sample_size": 5, "unique_sample_size": 3},
    "NLT": {"short_sample_size": 4, "long_sample_size": 2},
}
MIXED_TYPES = ["INTEGER", "FLOAT", "DATETIME", "BOOLEAN", "DIRTY_INTEGER", "DIRTY_FLOAT",
               "NAMED_ENTITY", "DISCRETE_STRING", "COLLECTION", "DELIMITED_STRING",
               "IDENTIFIER", "CATEGORICAL", "EMAIL", "URL", "MIXED"]

keys = st.lists(st.one_of(st.text(max_size=10), st.integers()), max_size=4)


@given(seed=st.integers(-(2**63), 2**63), keys=keys)
def test_same_seed_and_keys_give_the_same_random_streams(seed, keys):
    assert [seeded_random(seed, *keys).random() for _ in range(2)] == [seeded_random(seed, *keys).random()] * 2
    first = seeded_generator(seed, *keys).integers(0, 2**32, size=5)
    second = seeded_generator(seed, *keys).integers(0, 2**32, size=5)
    assert first.tolist() == second.tolist()


def test_no_seed_means_no_seeded_stream():
    assert seeded_random(None, "validation", "abv", 0) is None
    assert seeded_generator(None, "profile", "abv") is None


values = st.lists(st.one_of(st.integers(-1000, 1000), st.text(max_size=12)), min_size=1, max_size=300)
texts = st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=300)


@settings(max_examples=40, deadline=None)
@given(data=values, seed=st.integers(0, 10**9), column_type=st.sampled_from(MIXED_TYPES))
def test_same_seed_gives_the_same_column_sample(data, seed, column_type):
    series = pd.Series(data)
    first = DataSampler(SMALL_SIZES, seed=seed).sample_column(series, "c", column_type)
    second = DataSampler(SMALL_SIZES, seed=seed).sample_column(series, "c", column_type)
    assert first == second


@settings(max_examples=40, deadline=None)
@given(data=texts, seed=st.integers(0, 10**9))
def test_same_seed_gives_the_same_long_text_sample(data, seed):
    series = pd.Series(data)
    first = DataSampler(SMALL_SIZES, seed=seed).sample_column(series, "c", "NATURAL_LANGUAGE_TEXT")
    second = DataSampler(SMALL_SIZES, seed=seed).sample_column(series, "c", "NATURAL_LANGUAGE_TEXT")
    assert first == second


def test_different_seeds_pick_different_samples():
    series = pd.Series(range(1000))
    sizes = {**SMALL_SIZES, "NUMERIC": {"clean_sample_size": 50, "dirty_sample_size": 3}}

    assert DataSampler(sizes, seed=1).sample_column(series, "c", "INTEGER") != \
        DataSampler(sizes, seed=2).sample_column(series, "c", "INTEGER")


@pytest.mark.parametrize(("stored", "loaded"), [(7, 7), ("12", 12), (None, None), ("abc", None), (True, None)])
def test_sampling_seed_loads_as_a_whole_number_or_none(tmp_path, stored, loaded):
    path = tmp_path / "configurations.json"
    path.write_text(json.dumps({"sampling_seed": stored}), encoding="utf-8")

    seed = load_default_cleaning_config(path).sampling_seed

    assert seed == loaded
    assert seed is None or type(seed) is int
