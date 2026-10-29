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
    "CODING_PROMPT_TEMPLATE": "352d037a68d5c8bccf34ba6bc1fb14ac599b084c554b90e6ec2647eb5c4b502f",
    "FD_RECOMMENDATION_PROMPT_TEMPLATE": "bb13b89fa236a3e7aed5ab6d30b1f94be0f808ba1332612972ad044963bfeb25",
    "FD_VALIDATION_PROMPT_TEMPLATE": "6691c7b3d018e65ea9d02211b5f69a3e93e10e910bed845ef36aa0bc602dffc3",
    "OUTLIER_PROMPT_TEMPLATE": "8042f39725ca5b462b199af059e56a42a3cc732cba8b401b6dd35bbc5a5dccff",
    "RECOMMENDATION_PROMPT_TEMPLATES.BOOLEAN": "fadfe697b5bd0767fe47fb18caa8c1437c8939a1f3a417f984f79766cfc02e7e",
    "RECOMMENDATION_PROMPT_TEMPLATES.CATEGORICAL": "b3cda173a2c6eaefeabdcd5f2fe34fa8b3245e63a415629167636e5c5364c3fc",
    "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION": "b00127cb62ad67894c1d2b78815a250693fe7af2dfb9406388aaf434485225e4",
    "RECOMMENDATION_PROMPT_TEMPLATES.DATETIME": "01fcb41156fad87e64da801fc71c32a8cd2b3199c2497ab50bdf71f810155d0d",
    "RECOMMENDATION_PROMPT_TEMPLATES.DELIMITED_STRING": "7357c50ab837bbbca2c2e9174bb745804d3d7c5f925816e31cace90008c93288",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_FLOAT": "6112ccb10d71edc0a03cbb3dd598c439a0af0c9744335b8716c97c34631a1381",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_INTEGER": "57cee595b684bd6f2c372b05c29bae3352e56af15443f3f489cc09efb0873b8b",
    "RECOMMENDATION_PROMPT_TEMPLATES.DISCRETE_STRING": "c7d8636afffca11d52062fd4b254ebc3c9e43195c55287c45ce7b09aeeb0f6d0",
    "RECOMMENDATION_PROMPT_TEMPLATES.EMAIL": "84427133351308757244574708b421830625abb61c3e3841c715dacd6bd18d78",
    "RECOMMENDATION_PROMPT_TEMPLATES.FLOAT": "edfc32fa6279b4a13ce57ebba56d8f75e12353d48da843aae5890060a06068b2",
    "RECOMMENDATION_PROMPT_TEMPLATES.IDENTIFIER": "9517142668d3b79c8e686073333fb8226665fb247497b88d0617e317bc1d9b6c",
    "RECOMMENDATION_PROMPT_TEMPLATES.INTEGER": "3bf60a7d4e3a326a562128c9576d7d093a0620a8634e9748186c1747d445b28d",
    "RECOMMENDATION_PROMPT_TEMPLATES.MIXED": "a7a2cb1f5b2eb01c6236e343b35ce5755585215acd5320311d69fad36a44aad8",
    "RECOMMENDATION_PROMPT_TEMPLATES.NAMED_ENTITY": "820c70cc6e4a21f92683eea4aa2929a0e098e183fa9b2771d63a392185ccf5a8",
    "RECOMMENDATION_PROMPT_TEMPLATES.NATURAL_LANGUAGE_TEXT": "59d0c7934d75ef738692c04a758cd1b0838ed671a4b3a5a10a8867e2f6a634c1",
    "RECOMMENDATION_PROMPT_TEMPLATES.URL": "d10f054cd1b0950c5d7b2c90f84c4e9bde1e6f3b05a19ec7a05f4f0bccbba16f",
    "RECOMMENDER_CORE": "9b614c1e7dbf24ea989ff9c25a3ff7048857c986dad0042f000820501cdf30af",
    "RECOMMENDER_TYPE_PARTS.BOOLEAN": "c4f91ad79efa7c9428e91d173b21c63489cb43ddbcc9992c5f306e6c87b95563",
    "RECOMMENDER_TYPE_PARTS.CATEGORICAL": "9b9ca97bdcffa48166bab3b696f3957d4a56c9b1843799a2895f4ffff7be69a1",
    "RECOMMENDER_TYPE_PARTS.COLLECTION": "b9e1eaeb173f0e533cffc35c2abbf8c81c5f4cb10d51b7b46caa4d39585c3dfd",
    "RECOMMENDER_TYPE_PARTS.DATETIME": "7edb720a463d74517df6ecb1f9b905e052b493b7cfb9c42984b83a35ef1f44b7",
    "RECOMMENDER_TYPE_PARTS.DELIMITED_STRING": "798a21f99e18789a8071dc138a4d69f73c3011eaa71c8ecf436b364c012ba61d",
    "RECOMMENDER_TYPE_PARTS.DIRTY_FLOAT": "78d340418ca1731200c096a0c89637a81423d77ab45350aa50216c6e578f119e",
    "RECOMMENDER_TYPE_PARTS.DIRTY_INTEGER": "bf451fcdd9779fffb57c2237c16d0b2081b7ba27863571dacb799873cf93c4fc",
    "RECOMMENDER_TYPE_PARTS.DISCRETE_STRING": "d1e9093dfd1b30c657d2201596f5c4d8332e07f433e51d13df0c65299f560c26",
    "RECOMMENDER_TYPE_PARTS.EMAIL": "681699c08a4f9776933d4e1c416c93245fd1a42095a25990d415525e0ee84b6c",
    "RECOMMENDER_TYPE_PARTS.FLOAT": "f97ccf397f02c3a6e1fc021ac5bbb3bab1764d8c18b54eee1bb648b5a8c3fbec",
    "RECOMMENDER_TYPE_PARTS.IDENTIFIER": "4bd57a7ebde8ecf36c61e69ef94e0e323d802a2a73b29f6cd6c3a8b98274e3e9",
    "RECOMMENDER_TYPE_PARTS.INTEGER": "3da7ce2a56f6da40ef5e51d2d72f4e98b591a451858406391dd15bc971ad93bb",
    "RECOMMENDER_TYPE_PARTS.MIXED": "2fb55a576350672447e0c4cb6150c7126551622d61eac532d9f85260703f9597",
    "RECOMMENDER_TYPE_PARTS.NAMED_ENTITY": "4c8e5293b9db5f2ee556b1996103f382cc2d36fb34eb7dfa0bffbfa1040594bf",
    "RECOMMENDER_TYPE_PARTS.NATURAL_LANGUAGE_TEXT": "3cdda4fabe0875da60288a97d982c5f287fd36881313b5da8b961d6092de3268",
    "RECOMMENDER_TYPE_PARTS.URL": "486b5f4dad099621f63a01143f0dd13fde5ddddc281480dfe11695fba76b5fe1",
    "VALIDATION_PROMPT_TEMPLATES.BOOLEAN": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATION_PROMPT_TEMPLATES.DATETIME": "13d59965a38b9d2a7b0ef1c9cb4430a225b1ba6dc7f12aad39facdc1959193a8",
    "VALIDATION_PROMPT_TEMPLATES.DIRTY_NUMERIC": "b8ffa442b41edf198cd8a92759dd861a4f75c16f040492e19937f26d97951811",
    "VALIDATION_PROMPT_TEMPLATES.NLT": "e6dc41af2287ac659c9723cbef94611ec520611979486573b052d356aea690ee",
    "VALIDATION_PROMPT_TEMPLATES.NUMERIC": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATION_PROMPT_TEMPLATES.STRING": "0a2a22aa4a3b7c9d4010471f60cd8e23b6d0e06ea01d232e744daf9537aab13d",
    "VALIDATOR_CORE": "b151b2dd2f58db776320351cfcffbda7fbe48ae1a1ce13ca6bacb77e1f8ae45d",
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
                 "FD_VALIDATION_PROMPT_TEMPLATE"):
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


IS_CLEAN_CHECK = "Before you answer true, go through every distinct value in the sample"


def test_recommender_prompt_for_an_empty_only_column_holds_the_is_clean_check_and_example():
    profile = ColumnProfile(name="Address2", semantic_type="DISCRETE_STRING", sample="Column 'Address2': empty (1000)")

    prompt = PromptGeneration().create_prompt_recommender("Address2", profile)

    assert "Column 'Address2': empty (1000)" in prompt
    assert IS_CLEAN_CHECK in prompt
    assert "if any one is a missing-value placeholder from RULE 3" in prompt
    assert 'Example: a column whose only value is "empty"' in prompt
    assert '"is_clean": false, "value_mapping": [{"from_value": "empty", "to_value": null}]' in prompt


def test_recommender_prompt_for_a_shared_unit_column_holds_the_unit_rule():
    sample = "Column 'Sample': 33 patients (120), 45 patients (98), 12 patients (40)"
    profile = ColumnProfile(name="Sample", semantic_type="DISCRETE_STRING", sample=sample)

    prompt = PromptGeneration().create_prompt_recommender("Sample", profile)

    assert sample in prompt
    assert IS_CLEAN_CHECK in prompt
    assert 'if every value, or nearly every value, carries the same unit word after its number ("33 patients", "15 kg")' in prompt
    assert 'Mixed units ("5 kg", "3 lb") are not a shared unit.' in prompt
