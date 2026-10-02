import sys
from pathlib import Path

import pandas as pd
import pytest

from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.coordinator.prompts import RECOMMENDATION_PROMPT_TEMPLATES, RECOMMENDER_TYPE_PARTS
from madclean.components.dataprofiler.semantic_mapping import SemanticTypeDetection
from madclean.utils.helpers import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark_datasets"
NEW_TYPES = ["IDENTIFIER", "CATEGORICAL", "EMAIL", "URL", "MIXED"]


@pytest.fixture(scope="module")
def detector():
    return SemanticTypeDetection()


@pytest.fixture(scope="module")
def beers_types(detector):
    return detector.detect_types(load_dataset(DATA / "beers_dirty.csv"))


@pytest.fixture(scope="module")
def hospital_types(detector):
    return detector.detect_types(load_dataset(DATA / "hospital_dirty.csv"))


def infer(detector, values, name="c"):
    return detector._infer_column_type(pd.Series(values), name)


# --------------------------------------------------------------------------------------
# The benchmark columns the ticket names
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("column", ["id", "brewery_id"])
def test_beers_ids_are_identifiers(beers_types, column):
    assert beers_types[column] == "IDENTIFIER"


@pytest.mark.parametrize("column", ["ZipCode", "PhoneNumber"])
def test_hospital_codes_are_identifiers(hospital_types, column):
    assert hospital_types[column] == "IDENTIFIER"


@pytest.mark.parametrize("column", ["state", "style"])
def test_beers_labels_are_categorical(beers_types, column):
    assert beers_types[column] == "CATEGORICAL"


def test_hospital_city_is_categorical(hospital_types):
    assert hospital_types["City"] == "CATEGORICAL"


def test_beers_city_has_too_many_names_to_be_categorical(beers_types):
    assert beers_types["city"] == "NAMED_ENTITY"


def test_movie_descriptions_with_commas_are_not_lists(detector):
    movies = pd.read_csv(DATA / "movies_dirty.csv", usecols=["Description"])

    assert detector.detect_types(movies)["Description"] != "DELIMITED_STRING"


# --------------------------------------------------------------------------------------
# Small frames
# --------------------------------------------------------------------------------------

def test_email_addresses(detector):
    assert infer(detector, ["a@b.com", "jo.smith@mail.org", "x_y@site.co.uk", "q@r.net"]) == "EMAIL"


def test_web_addresses(detector):
    assert infer(detector, ["https://x.org", "http://example.com/a", "www.site.net", "https://y.io/b?c=1"]) == "URL"


def test_half_dates_half_words_is_mixed(detector):
    values = ["2021-03-14", "2020-01-02", "1999-12-31", "2005-07-04", "apple", "banana", "cherry", "grape"]

    assert infer(detector, values) == "MIXED"


def test_gender_letters_are_not_booleans(detector):
    assert infer(detector, ["M", "F", "M", "F", "M", "X", "F", "M"]) != "BOOLEAN"


def test_t_and_f_are_booleans(detector):
    assert infer(detector, ["t", "f", "t", "t", "f", "F", "T", "f"]) == "BOOLEAN"


def test_first_names_with_month_names_among_them_are_not_dates(detector):
    names = ["May", "June", "Sam", "Alice", "Robert", "Maria", "John", "Peter", "Linda", "Karen", "Oscar", "Emma"]

    assert infer(detector, names) != "DATETIME"


@pytest.mark.parametrize("values", [["$1,234", "$2,500", "$999", "$1,000"],
                                    ["$1,234.50", "$12,000.00", "$3,100.25"]])
def test_money_with_thousands_separators_stays_a_number_with_noise(detector, values):
    assert infer(detector, values) == "DIRTY_INTEGER"


def test_digit_only_codes_with_a_leading_zero_are_identifiers(detector):
    assert infer(detector, ["01234", "02345", "12345", "54321", "00999"], name="area") == "IDENTIFIER"


def test_two_part_numbers_are_not_dates(detector):
    assert not SemanticTypeDetection._is_datetime("12-15")
    assert SemanticTypeDetection._is_datetime("2023-01-02")
    assert SemanticTypeDetection._is_datetime("1/2/23")
    assert SemanticTypeDetection._is_datetime("08:30")
    assert SemanticTypeDetection._is_datetime("Mon")


# --------------------------------------------------------------------------------------
# Every new type is wired into the recommender, the validator and the GUI
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("column_type", NEW_TYPES)
def test_new_type_has_a_recommender_part(column_type):
    assert RECOMMENDER_TYPE_PARTS[column_type].strip()
    assert f"COLUMN TYPE: {column_type}" in RECOMMENDATION_PROMPT_TEMPLATES[column_type]


@pytest.mark.parametrize("column_type", NEW_TYPES)
def test_new_type_is_validated_with_the_string_validator(column_type):
    prompt = PromptGeneration().create_prompt_validation(
        "c", pd.Series(["a", "b"]), pd.Series(["a", "c"]), column_type)

    assert "COLUMN TYPE: STRING" in prompt


@pytest.mark.parametrize("column_type", NEW_TYPES)
def test_new_type_has_its_own_gui_colour(column_type):
    sys.path.insert(0, str(ROOT / "gui"))
    from gui.state import State

    assert State._semantic_type_to_color(column_type) != "#9CA3AF"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def test_raw_text_load_keeps_leading_zeros_and_placeholder_text():
    tax = load_dataset(DATA / "tax_dirty.csv", keep_raw_text=True)

    assert tax["zip"].astype(str).str.startswith("0").any()
    assert (tax["rate"] == "NaN").sum() == 187


def test_default_load_is_unchanged():
    path = DATA / "tax_dirty.csv"

    loaded = load_dataset(path)

    pd.testing.assert_frame_equal(loaded, pd.read_csv(path, encoding="utf-8", on_bad_lines="skip"))
    assert not loaded["zip"].astype(str).str.startswith("0").any()
