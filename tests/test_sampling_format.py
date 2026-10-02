import re
import time
from collections import Counter

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from madclean.components.dataprofiler.data_selection import DataSampler, value_shape
from madclean.config.settings import CleaningConfig

SIZES = CleaningConfig().sample_sizes
COUNT_LINE = re.compile(r"^(.*) \((\d+)\)$")


def sample(values, column_type, sizes=SIZES, seed=7, col="c") -> str:
    return DataSampler(sizes, seed=seed).sample_column(pd.Series(values), col, column_type)


def counts_section(text: str) -> list[tuple[str, int]]:
    """The (value, count) lines under 'Values with counts', in order."""
    lines = text.splitlines()
    start = lines.index("Values with counts (most frequent first):") + 1
    pairs = []
    for line in lines[start:]:
        match = COUNT_LINE.match(line)
        if not match or "←" in line:
            break
        pairs.append((match.group(1), int(match.group(2))))
    return pairs


CITIES = ["Saint Louis"] * 52 + ["Saint Louis MO"] + ["Denver"] * 4 + ["Chicago"] * 3 + ["Boston"] * 2


def test_city_column_lists_counts_most_frequent_first_and_groups_the_variant():
    text = sample(CITIES, "NAMED_ENTITY", col="city")

    assert "Saint Louis (52)" in text
    first_rare = min(text.index(f"{city} (1)") for city in ["Saint Louis MO"])
    assert text.index("Saint Louis (52)") < first_rare
    assert "Saint Louis (52) ← Saint Louis MO (1)" in text
    assert "Random Sample" not in text  # every distinct value fits, so the random sample is dropped


def test_date_column_lists_its_shapes_with_counts():
    dates = ["03/14/2021", "04/02/2021", "11/30/2020", "01/01/2022", "07/04/2021", "12/25/2020", "02/28/2021",
             "09/09/2021", "2021-03-14", "2020-12-01"]

    text = sample(dates, "DATETIME", col="date")

    assert "Formats by shape (count, example):" in text
    assert "99/99/9999 (8)" in text
    assert "9999-99-99 (2)" in text
    assert text.index("99/99/9999 (8)") < text.index("9999-99-99 (2)")


def test_float_column_has_a_numeric_summary():
    text = sample([1.5, 2.25, 3.0, 10.125, "n/a"], "FLOAT", col="abv")

    assert "Numeric summary: min 1.5" in text
    assert "max 10.125" in text
    assert "decimal places:" in text
    assert "3 → 1 values" in text


def test_dirty_numeric_summary_reads_the_first_number_of_each_value():
    text = sample(["12 kg", "3.5 kg", "$7"], "DIRTY_FLOAT")

    assert "Numeric summary: min 3.5" in text
    assert "max 12" in text


def test_one_distinct_value_gives_one_line_and_no_variants():
    text = sample(["Red"] * 20, "CATEGORICAL")

    assert counts_section(text) == [("Red", 20)]
    assert "Possible variants" not in text


def test_no_near_duplicates_means_no_variants_section():
    text = sample(["Red"] * 5 + ["Blue"] * 5 + ["Green"], "CATEGORICAL")

    assert "Possible variants" not in text


def test_long_values_are_cut_at_80_characters():
    long_value = "x" * 120

    text = sample([long_value, "short"], "DISCRETE_STRING")

    assert "x" * 80 + "… (1)" in text
    assert "x" * 81 not in text


def test_values_with_spaces_at_their_ends_are_quoted():
    text = sample(["Boston", "Boston", "Boston "], "NAMED_ENTITY")

    assert '"Boston " (1)' in text


def test_every_value_distinct_stays_fast_and_skips_variants_for_lists():
    ids = [f"ID-{i:05d}" for i in range(5000)]
    lists = [f"a{i}, b{i}" for i in range(5000)]

    start = time.perf_counter()
    id_text = sample(ids, "IDENTIFIER")
    list_text = sample(lists, "DELIMITED_STRING")
    elapsed = time.perf_counter() - start

    assert elapsed < 5
    assert "ID-00000 (1)" in id_text
    assert "Formats by shape (count, example):\nA-99999 (5000)" in id_text
    assert "Possible variants" not in list_text
    assert "Random Sample" in list_text  # 5,000 distinct values do not fit in 500


def test_when_values_do_not_fit_rare_variants_are_kept_beside_the_frequent_ones():
    frequent = [f"Label {i:03d}" for i in range(600) for _ in range(3)]
    values = frequent + ["Labl 001"]
    sizes = {**SIZES, "STRING": {"random_sample_size": 10, "unique_sample_size": 200}}

    text = sample(values, "CATEGORICAL", sizes=sizes)

    listed = counts_section(text)
    assert len(listed) == 200
    assert ("Labl 001", 1) in listed
    assert "Label 001 (3) ← Labl 001 (1)" in text


def test_identifier_and_mixed_samples_carry_shapes():
    assert "Formats by shape" in sample(["01234", "02345", "12345"], "IDENTIFIER")
    assert "Formats by shape" in sample(["2021-01-01", "apple", "pear"], "MIXED")
    assert "Formats by shape" not in sample(["apple", "pear"], "NAMED_ENTITY")


def test_shapes_mark_digits_and_letter_runs():
    assert value_shape("03/14/2021") == "99/99/9999"
    assert value_shape("AB-1234") == "A-9999"
    assert value_shape("Saint louis MO") == "Aa a A"
    assert value_shape("eBay") == "Aa"


words = st.text(alphabet="abcdefXYZ019 ", min_size=1, max_size=6).map(str.strip).filter(bool)


@settings(max_examples=60, deadline=None)
@given(values=st.lists(words, min_size=1, max_size=200))
def test_counts_cover_every_row_and_never_increase_down_the_list(values):
    text = sample(values, "DISCRETE_STRING")

    listed = counts_section(text)
    counts = [count for _, count in listed]
    assert counts == sorted(counts, reverse=True)
    assert dict(listed) == dict(Counter(values))
