import hashlib
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from madclean.components.coordinator import prompts
from madclean.components.coordinator.prompt_generation import PromptGeneration
from madclean.components.domain.schema import ColumnProfile
from madclean.utils.helpers import format_prompt_template

# SHA-256 of every template, its shared cores and its type parts. A template edit must update its
# hash here in the same commit, so no prompt changes by accident. The thesis wording is kept in git
# under the v2 tag.
TEMPLATE_HASHES = {
    "CODING_PROMPT_TEMPLATE": "c647c6ff27748b0c658a02048803f268692715cb8c549ae0eb7dff0b84cf339b",
    "FD_CODING_PROMPT_TEMPLATE": "ce18e023e070dcd9bab88fe59df8f27596bb8c0ba6800ce85e5d7e6562ea4a8c",
    "FD_RECOMMENDATION_PROMPT_TEMPLATE": "235f86f178ff413df786ee9ef566c7a11ae7478fca895f6e87ce82f9c55d6ec9",
    "FD_VALIDATION_PROMPT_TEMPLATE": "6691c7b3d018e65ea9d02211b5f69a3e93e10e910bed845ef36aa0bc602dffc3",
    "OUTLIER_PROMPT_TEMPLATE": "8042f39725ca5b462b199af059e56a42a3cc732cba8b401b6dd35bbc5a5dccff",
    "RECOMMENDATION_PROMPT_TEMPLATES.BOOLEAN": "fe3a5125646344d0e7461bf94bb6cb99ba5fb22c8cb3bc18b50f3f5fcc1ef061",
    "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION": "edbfaeaaa7b198f578e04f927f17585584d8bdc5ecfbd1fef7852edfbbf6f8b1",
    "RECOMMENDATION_PROMPT_TEMPLATES.DATETIME": "98a81c4e5f0fa7e5e6351a698041a83e895b0d92cd9944d4c6f2d243426915f4",
    "RECOMMENDATION_PROMPT_TEMPLATES.DELIMITED_STRING": "8ec5ddc4732d537224f598e0fd77f61a1181617d5b3e9789156478fb76036e20",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_FLOAT": "6b257962b7fd63a5eecefeb0fa77bace818d0b5d61e51869b22c7bfee875cdd3",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_INTEGER": "32fb06c878a94905a92842e8afd3ddec7b13a88f787e444fb0d2850266544b6e",
    "RECOMMENDATION_PROMPT_TEMPLATES.DISCRETE_STRING": "d1792f67ad81081f19c70b1921d613ada47657ba8281a6c7eca7f94d1ab58d2d",
    "RECOMMENDATION_PROMPT_TEMPLATES.FLOAT": "6d944b649ffe62f60a11eb7e9ce35c4c5e22e024f3aa6cbabc3e41d24d4c51fa",
    "RECOMMENDATION_PROMPT_TEMPLATES.INTEGER": "8b8973f47ab97182bf92727d8e4d7f4372eb1423bfb0a751cbe9e86dd933910b",
    "RECOMMENDATION_PROMPT_TEMPLATES.NAMED_ENTITY": "261af36ab4a520104920aa0266bdebdab0a6a5ff7b85ee6d643358dd34c2b2d4",
    "RECOMMENDATION_PROMPT_TEMPLATES.NATURAL_LANGUAGE_TEXT": "ab0a288e0e431473aac8edf7d85be53601f0ca5e3f882f981978da797f2b0fc0",
    "RECOMMENDER_CORE": "b208cecab6df123b1e7b460119be996658de367e85de0b3eca8872f6e560001d",
    "RECOMMENDER_TYPE_PARTS.BOOLEAN": "f293488df61b19032f7afb4fe8f10b50db4c344c9e3c6533599c209ec554c157",
    "RECOMMENDER_TYPE_PARTS.COLLECTION": "9148cddc585b104d2bc1b7182c98020c5f0a2005fd79264ee01d65aeae6b5c8e",
    "RECOMMENDER_TYPE_PARTS.DATETIME": "7edb720a463d74517df6ecb1f9b905e052b493b7cfb9c42984b83a35ef1f44b7",
    "RECOMMENDER_TYPE_PARTS.DELIMITED_STRING": "e22f02625c938e79b9e9fae710b1893d0fd8b8d9b270f1e30d813f8101afd600",
    "RECOMMENDER_TYPE_PARTS.DIRTY_FLOAT": "78d340418ca1731200c096a0c89637a81423d77ab45350aa50216c6e578f119e",
    "RECOMMENDER_TYPE_PARTS.DIRTY_INTEGER": "bf451fcdd9779fffb57c2237c16d0b2081b7ba27863571dacb799873cf93c4fc",
    "RECOMMENDER_TYPE_PARTS.DISCRETE_STRING": "04c17c5f6fd3d5039b07d08b942e0c9908e935863671db99ac123ec768944607",
    "RECOMMENDER_TYPE_PARTS.FLOAT": "f97ccf397f02c3a6e1fc021ac5bbb3bab1764d8c18b54eee1bb648b5a8c3fbec",
    "RECOMMENDER_TYPE_PARTS.INTEGER": "3da7ce2a56f6da40ef5e51d2d72f4e98b591a451858406391dd15bc971ad93bb",
    "RECOMMENDER_TYPE_PARTS.NAMED_ENTITY": "9b10d5a94df02faa25e46b5271fbda5cfdf7a236b4262034e0e20796b5ef7ad5",
    "RECOMMENDER_TYPE_PARTS.NATURAL_LANGUAGE_TEXT": "3cdda4fabe0875da60288a97d982c5f287fd36881313b5da8b961d6092de3268",
    "VALIDATION_PROMPT_TEMPLATES.BOOLEAN": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATION_PROMPT_TEMPLATES.DATETIME": "fdcf7d7f5216cc4f8f9c9cce4559385a35aae1c5fced622e79b8602e6f4801e9",
    "VALIDATION_PROMPT_TEMPLATES.DIRTY_NUMERIC": "a203f4956769479d93eefeb4ad11b56c4d076d80b65d88808063719a4518f6c1",
    "VALIDATION_PROMPT_TEMPLATES.NLT": "c03c6052b215ed7bfa3de568ba54288ae59b8fe64dc1d16672ba1a524b58c3cc",
    "VALIDATION_PROMPT_TEMPLATES.NUMERIC": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATION_PROMPT_TEMPLATES.STRING": "3df94e610862d76e2af60e3290ec04cef75128d50586912e42559ec0cc65850b",
    "VALIDATOR_CORE": "1714cc7f7cd89c09644336a7aeb4ae76055583c7a1b8a5f06af13018024c6020",
    "VALIDATOR_TYPE_PARTS.BOOLEAN": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATOR_TYPE_PARTS.DATETIME": "9ed0baacb07ea28c2d0763cf9299f6415bba647cd4b44c5d007abe3943387848",
    "VALIDATOR_TYPE_PARTS.DIRTY_NUMERIC": "fd8ef3780cadf7a77dfceb3055c7afcfb7192ef5ae9516ec5205cd4e959c2d1d",
    "VALIDATOR_TYPE_PARTS.NLT": "532d347093370c0e4602365c9c25ea13767eca05a93ea631067525f584ccbb3b",
    "VALIDATOR_TYPE_PARTS.NUMERIC": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATOR_TYPE_PARTS.STRING": "df6ea0b81dc28d4e5787142bdf3afa5b6f60a9958431181bff9e2f635d587927",
    "_TYPE_SLOT": "738d3105d417aa6236087ddab7edcbd7ecb993134dc8f3ddc0f02378e28d2112",
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


def test_templates_match_their_pinned_hashes():
    assert _current_template_hashes() == TEMPLATE_HASHES


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
