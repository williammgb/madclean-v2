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
    "RECOMMENDATION_PROMPT_TEMPLATES.BOOLEAN": "e526a5b9ab68b8fd94886183169e44ed675025ad770944e5ca756f0392725fac",
    "RECOMMENDATION_PROMPT_TEMPLATES.CATEGORICAL": "92a9c152f0f0af76c0aea1e5423f2bbdf5dbcbe3aff7730923cdccfc795a766e",
    "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION": "1a20dbcb3dde5b0c72ad33db7024e20a17909370a18dc20c449664bed7b85252",
    "RECOMMENDATION_PROMPT_TEMPLATES.DATETIME": "4d79672cdc1b5f2cf4bb4e974e11fa35f9d5a4e1ce76afc12040ee62693fb1d7",
    "RECOMMENDATION_PROMPT_TEMPLATES.DELIMITED_STRING": "2c9df62fb7020d9679f88b7d55f995da1240415b00218b8d4819c8e0ad9c0cfb",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_FLOAT": "35323e01c9753ce7cf42c0007e2162e7cb0b5275c9946d766132e648ec439eb4",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_INTEGER": "2f5614a0564ec540294dbf9390c77e4dc2ab477adab745eb3697fc527f6382ed",
    "RECOMMENDATION_PROMPT_TEMPLATES.DISCRETE_STRING": "3bdf2eabc07c2652211ce9a79fd5d154a9d26e9cb5689d70d0bbd85b1856346b",
    "RECOMMENDATION_PROMPT_TEMPLATES.EMAIL": "b7704943a785c9af0f37181af3eb27b5a4c9d8f4d10ebdc48c5fb6b630c5ba45",
    "RECOMMENDATION_PROMPT_TEMPLATES.FLOAT": "da0d11e975c91477852c63368bd36f3945c5feff5d7c226924cc7d3bae100c95",
    "RECOMMENDATION_PROMPT_TEMPLATES.IDENTIFIER": "478ce58f24f71d59b4df3ff0a5c4832bd4f76cdb3d5a586cd5a21edaf11c4a5f",
    "RECOMMENDATION_PROMPT_TEMPLATES.INTEGER": "7e485ed6ee988888881726d200cdd4e0efbd99dd15a444456258639f1795785f",
    "RECOMMENDATION_PROMPT_TEMPLATES.MIXED": "c14eb224dc63673d2b559ce5b5ae124431f727350ca517cd4fd163b2b7398d44",
    "RECOMMENDATION_PROMPT_TEMPLATES.NAMED_ENTITY": "727c44899b6b35893ae96ca0a7f58d0c6f16ef7702ef3d6dbaa92ef7a6684f56",
    "RECOMMENDATION_PROMPT_TEMPLATES.NATURAL_LANGUAGE_TEXT": "d3ea72c181ecee8e3d3573200167335226c9b342a1965571b97d7ab384a79e99",
    "RECOMMENDATION_PROMPT_TEMPLATES.URL": "d0ff808141b9edd363432e6b5ba3e19f4110ea606e26bf2ab0fcb88c4c5d9b60",
    "RECOMMENDER_CORE": "2194ff383ce047ed78e8a8c63ed215bcce249cf0f7d22dde106e2b97f002a06f",
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
    "VALIDATION_PROMPT_TEMPLATES.DATETIME": "6cc49f4fa50fd332b3f818747229500ebc9ac9e58ddbb92e6c53b2aa3d60fb88",
    "VALIDATION_PROMPT_TEMPLATES.DIRTY_NUMERIC": "90249177ce3ad41d6202ebd24fc85a3a6b5ef108133d06ed78555d7b19c7064e",
    "VALIDATION_PROMPT_TEMPLATES.NLT": "d014aed3b1ccdd1c357877f5a84df7093267281fbd0e7e77d666d37b4a42d38a",
    "VALIDATION_PROMPT_TEMPLATES.NUMERIC": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "VALIDATION_PROMPT_TEMPLATES.STRING": "700e8baa8d91ef26b7ce1bd72fd5819eed70e93d7fcd73ed06ca0e65966c9ed3",
    "VALIDATOR_CORE": "2ddd49502e2a60cf785a41d99980964744fe93084378e2c88b6d67b1c067c251",
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
