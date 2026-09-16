import hashlib
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from madclean.components.coordinator import prompts
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import ColumnProfile
from madclean.utils.helpers import format_prompt_template

# SHA-256 of every template as it was at thesis-final. Prompt wording is out of
# scope for v2, so any edit to a template's text must fail here.
THESIS_TEMPLATE_HASHES = {
    "CODING_PROMPT_TEMPLATE": "19b0aac7f3d8b43f82dbbf5393ec2984839b10b48a4be466f41534e666ab9871",
    "FD_CODING_PROMPT_TEMPLATE": "baa56118831cf1dae804d46c3d9fbf57f4a06d42883630e92eab65900e07adae",
    "FD_RECOMMENDATION_PROMPT_TEMPLATE": "a20ef863a747316284d3cb3ea730f5bbb85f8f80d33c236f8fc81eaaaa3e6a12",
    "FD_VALIDATION_PROMPT_TEMPLATE": "45595a34daede4a21597240e683e179f6c2d1fd1c468c8a5b2cdf4339df382ab",
    "OUTLIER_PROMPT_TEMPLATE": "05eb6379f752e751187b836ed2b4f429e30deaf5faa46c93c81d590eadb5713d",
    "RECOMMENDATION_PROMPT_TEMPLATES.BOOLEAN": "096dbf073cdb566e61c3dd7581a177650c67a9a177fffff3d7adcbf69b1ff8b4",
    "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION": "4ca4dc9ea47dddef129808c0ab781b08da4dad8e4b5e508509a35f71968ff00f",
    "RECOMMENDATION_PROMPT_TEMPLATES.DATETIME": "c63cfa0241ad64e6dc1da38da7c1dac2fb32f3ab819ac8d21e329acf4f7714d1",
    "RECOMMENDATION_PROMPT_TEMPLATES.DELIMITED_STRING": "9877e4a5313194ed79061c917a0554c47c11ac8935b7d89274c8435331672121",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_FLOAT": "a51377718aa315a9070dc255e07b2137f29435e4234090d9dd886b9731ec6c5d",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_INTEGER": "735e4b0e5606b9158f4470b767449ce91cf05f7232b55aa7289f5c9ba81baac7",
    "RECOMMENDATION_PROMPT_TEMPLATES.DISCRETE_STRING": "38ec9cf285b7c680c903134acc397525bf051f5d72069854ffb247f51a1d4c28",
    "RECOMMENDATION_PROMPT_TEMPLATES.FLOAT": "3ee531cbca2e52fa78ef55a6dceb63090e027d9e01beddb10989f25c38660592",
    "RECOMMENDATION_PROMPT_TEMPLATES.INTEGER": "f547b5c3ae121eeb68cfc86d6db48b90b5e95b770250d68932a0c74fd58ac602",
    "RECOMMENDATION_PROMPT_TEMPLATES.NAMED_ENTITY": "ecf49c96af1640248b6cf636b0b0daeacfda4223fc650ee709f2dd7264b88b81",
    "RECOMMENDATION_PROMPT_TEMPLATES.NATURAL_LANGUAGE_TEXT": "4799e418522fde93e65283cdff7eb382fe7ed7b9a37f617ff73ea1b1069aab2e",
    "VALIDATION_PROMPT_TEMPLATES.BOOLEAN": "36a9e7f1c95b82ffb99743e0c5c4ce95d83c9a430aac59f84ef3cbfab6145068",
    "VALIDATION_PROMPT_TEMPLATES.DATETIME": "89297b66091c26af328c0dbf01b977bbca9cf1c4692d6f7205f3e030fb145f46",
    "VALIDATION_PROMPT_TEMPLATES.DIRTY_NUMERIC": "c6d9d5ecfa6fdc4e83d83896482c9d2fb6c84255629f17e59c38ba8767e0b182",
    "VALIDATION_PROMPT_TEMPLATES.NLT": "c08519b83c0d3d93cda8096293f3d6f3a1af5a190cc5860168b09194b9447e8c",
    "VALIDATION_PROMPT_TEMPLATES.NUMERIC": "36a9e7f1c95b82ffb99743e0c5c4ce95d83c9a430aac59f84ef3cbfab6145068",
    "VALIDATION_PROMPT_TEMPLATES.STRING": "a7570b01962b12cfccc5f13d1d9531ebdee359929a06d549f29b836252634c71",
}


def _current_template_hashes():
    hashes = {}
    for name in dir(prompts):
        if not name.isupper():
            continue
        value = getattr(prompts, name)
        if isinstance(value, dict):
            for key, template in value.items():
                hashes[f"{name}.{key}"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
        elif isinstance(value, str):
            hashes[name] = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return hashes


def test_templates_unchanged_since_thesis():
    assert _current_template_hashes() == THESIS_TEMPLATE_HASHES


# The COLLECTION template holds literal examples like {'k': 1} that str.format cannot
# parse, so its placeholder names are listed by hand.
COLLECTION_FIELDS = ["additional_context", "column_name", "column_sample", "labeled_examples", "user_constraints"]


def _all_templates():
    templates = {}
    for group in ("RECOMMENDATION_PROMPT_TEMPLATES", "VALIDATION_PROMPT_TEMPLATES"):
        for key, template in getattr(prompts, group).items():
            templates[f"{group}.{key}"] = template
    for name in ("OUTLIER_PROMPT_TEMPLATE", "CODING_PROMPT_TEMPLATE", "FD_RECOMMENDATION_PROMPT_TEMPLATE",
                 "FD_CODING_PROMPT_TEMPLATE", "FD_VALIDATION_PROMPT_TEMPLATE"):
        templates[name] = getattr(prompts, name)
    return templates


def _fields(name, template):
    if name == "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION":
        return COLLECTION_FIELDS
    return sorted({field for _, field, _, _ in string.Formatter().parse(template) if field})


TEMPLATES = _all_templates()

# Text that is dense in braces, colons and quotes, mixed with arbitrary characters, so
# injected data looks like placeholders, escapes and dict literals as often as possible.
injected_text = st.one_of(
    st.text(alphabet=st.sampled_from(list("{}{}:'\"k1 _abc!\n")), max_size=40),
    st.text(max_size=40),
    st.sampled_from(["{column_name}", "{{", "}}", "{'k': 1}", "{0}", "{", "}", "{x:>10}"]),
)
# The outlier block formats these with numeric specs ({median:g}, {mad:.4f}); the pipeline passes numbers.
numeric = st.one_of(st.integers(-10**9, 10**9), st.floats(allow_nan=False, allow_infinity=False))
FIELD_STRATEGIES = {"median": numeric, "mad": numeric}


def _draw_values(data, name, template):
    return {field: data.draw(FIELD_STRATEGIES.get(field, injected_text), label=field) for field in _fields(name, template)}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_filling_never_raises_and_keeps_injected_text_verbatim(name, data):
    template = TEMPLATES[name]
    values = _draw_values(data, name, template)

    filled = format_prompt_template(template, **values)

    for value in values.values():
        if isinstance(value, str):
            assert value in filled


@pytest.mark.parametrize("name", sorted(n for n in TEMPLATES if n != "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION"))
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_filling_matches_str_format_for_every_thesis_template(name, data):
    template = TEMPLATES[name]
    values = _draw_values(data, name, template)

    assert format_prompt_template(template, **values) == template.format(**values)


def test_collection_prompt_keeps_its_brace_examples_and_fills_every_placeholder():
    profile = ColumnProfile(name="tags", semantic_type="COLLECTION", sample="Column 'tags': [\"{'k': 1}\"]")

    prompt = PromptGeneration().create_prompt_recommender("tags", profile)

    assert "{'k': 1}" in prompt
    assert "Column 'tags': [\"{'k': 1}\"]" in prompt
    for field in COLLECTION_FIELDS:
        assert "{" + field + "}" not in prompt


def test_missing_placeholder_value_still_raises():
    with pytest.raises(KeyError):
        format_prompt_template("Column {column_name}", other="x")
