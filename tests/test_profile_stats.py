"""The Profile view's per-column numbers and findings, computed on this machine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GUI_DIR = Path(__file__).resolve().parents[1] / "gui"
sys.path.insert(0, str(GUI_DIR))

from gui.profile_stats import column_profile, profile_rows, value_shape  # noqa: E402


def titles(row):
    return [f["title"].split(" · ")[0] for f in row["findings"]]


def finding(row, title):
    return next(f for f in row["findings"] if f["title"].startswith(title))


def test_value_shape_exact_keeps_lengths_and_case():
    assert value_shape("35233") == r"^\d{5}$"
    assert value_shape("hf-3") == r"^[a-z]{2}\-\d$"
    assert value_shape("OR") == r"^[A-Z]{2}$"
    assert value_shape("") == "^$"


def test_value_shape_loose_folds_words_of_any_length():
    assert value_shape("fort payne", loose=True) == value_shape("Boston", loose=True) == "^[A-Za-z ]+$"
    assert value_shape("12.0", loose=True) == r"^\d+\.\d+$"
    assert value_shape("St. Paul", loose=True) != "^[A-Za-z ]+$"


def test_odd_values_listed_when_one_shape_covers_95_percent():
    zips = pd.Series(["35233"] * 97 + ["3x233", "3x233", "35x33"])
    row = column_profile(zips, "CATEGORICAL", 100)
    odd = finding(row, "Odd values")
    assert odd["title"] == "Odd values · 3"
    assert r"^\d{5}$" in odd["rule"]
    # rarest first: a value seen once is more likely a typo
    assert [v["value"] for v in odd["values"]] == ["35x33", "3x233"]
    assert [v["count"] for v in odd["values"]] == ["1", "2"]


def test_no_odd_values_when_no_shape_dominates():
    mixed = pd.Series(["12.0 oz"] * 60 + ["12.0 oz."] * 40)
    row = column_profile(mixed, "DISCRETE_STRING", 100)
    assert "Odd values" not in titles(row)
    assert [s["pct"] for s in row["shapes"]] == ["60.0", "40.0"]


def test_number_column_finds_text_and_far_values():
    abv = pd.Series(["0.05"] * 40 + ["0.06"] * 40 + ["0.055%"] * 5 + ["9.9"])
    row = column_profile(abv, "DIRTY_FLOAT", 86)
    assert row["is_numeric"] == "1"
    text = finding(row, "Not a number")
    assert text["values"] == [{"value": "0.055%", "count": "5"}]
    far = finding(row, "Far from the rest")
    assert far["title"] == "Far from the rest · 1"
    assert far["values"][0]["value"] == "9.9"
    assert row["numeric_max"] == "9.9"
    assert len(row["histogram"]) > 0


def test_identifier_has_no_far_values():
    ids = pd.Series(list(range(1, 100)) + [100000])
    row = column_profile(ids, "IDENTIFIER", 100)
    assert "Far from the rest" not in titles(row)
    assert row["is_numeric"] == "1"


def test_placeholders_spaces_and_repeats():
    names = pd.Series([f"name {i}" for i in range(97)] + ["N/A", " padded", "name 1"])
    row = column_profile(names, "NAMED_ENTITY", 100)
    assert finding(row, "Placeholder text")["values"] == [{"value": "N/A", "count": "1"}]
    assert finding(row, "Extra spaces")["values"] == [{"value": "' padded'", "count": "1"}]
    repeats = finding(row, "Repeats in a mostly unique column")
    assert repeats["values"] == [{"value": "name 1", "count": "2"}]
    assert row["repeated_count"] == "1"


def test_clean_column_has_no_findings():
    row = column_profile(pd.Series(["al"] * 50 + ["ak"] * 50), "CATEGORICAL", 100)
    assert row["findings"] == []
    assert row["has_findings"] == "0"
    assert row["length_range"] == "2"


def test_empty_column_and_missing_counts():
    row = column_profile(pd.Series([np.nan, np.nan]), "EMPTY", 2)
    assert row["missing_count"] == "2"
    assert row["missing_pct"] == "100.0"
    assert row["shapes"] == []
    assert row["findings"] == []


def test_every_shown_value_is_a_string():
    df = pd.DataFrame({"n": [1.5, 2.5, None, 99.0], "t": ["a", "b", "b", None]})
    for row in profile_rows(df, ["n", "t", "gone"], {"n": "FLOAT"}):
        for key, value in row.items():
            if isinstance(value, list):
                for item in value:
                    for leaf in item.values():
                        assert isinstance(leaf, (str, list)), (key, leaf)
            else:
                assert isinstance(value, str), (key, value)
