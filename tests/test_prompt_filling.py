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
    "RECOMMENDATION_PROMPT_TEMPLATES.BOOLEAN": "1747c1f5a28d3e42d6c1709c8f527f0b3cc875fe2098101443280e6bfbdff814",
    "RECOMMENDATION_PROMPT_TEMPLATES.CATEGORICAL": "7bc9c5815939f4f0edbb76b28f659be912dcff9bcada445b61812fcc4206006d",
    "RECOMMENDATION_PROMPT_TEMPLATES.COLLECTION": "a563d156075470edc9688b83fdf5db6945981d43b2f7cb305f50be1fb38f4911",
    "RECOMMENDATION_PROMPT_TEMPLATES.DATETIME": "3921a1e47cd32d3925cba622e12a0bb8417dee41631331a615cd373bbff96956",
    "RECOMMENDATION_PROMPT_TEMPLATES.DELIMITED_STRING": "ea699de6643b5db9d75778b0899321bd14d7077268e9b72af01ab830fcea8496",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_FLOAT": "5eba2d65166c2db86187cf7561c7fefacf076083537a7fe7f93404153e2ec9fc",
    "RECOMMENDATION_PROMPT_TEMPLATES.DIRTY_INTEGER": "40e42e84b0961f2a53c4c270f36076775de0b1e8cbf3f69fd6c11e14c6234fdf",
    "RECOMMENDATION_PROMPT_TEMPLATES.DISCRETE_STRING": "f7c0b10b3ada577cfe6c4d1336a750c884e189c0c7b57f82aa42138952aae91d",
    "RECOMMENDATION_PROMPT_TEMPLATES.EMAIL": "1f9434f07ffbccc830b5e7bd33f95bb60b2edf227932d91e509ec372451e0900",
    "RECOMMENDATION_PROMPT_TEMPLATES.FLOAT": "302fa801a7ab1587c7fa1e804e5dcdc168b0d05363c0870cad216d8ff984ff26",
    "RECOMMENDATION_PROMPT_TEMPLATES.IDENTIFIER": "bbe79bd85c80a1bd1d0bd329e254d6313556c63b6ce43a2a8a0ed6e1d6e021a3",
    "RECOMMENDATION_PROMPT_TEMPLATES.INTEGER": "fcd74d01799c2730c7be7c0fa9b41e75fe8ab46f7644121d6b8518f9a35e66bc",
    "RECOMMENDATION_PROMPT_TEMPLATES.MIXED": "eb671715ad2b8a10fcddd39c2e5c48fcca6f347a1dfbe218ad806d764bd290c1",
    "RECOMMENDATION_PROMPT_TEMPLATES.NAMED_ENTITY": "d14375c54299678e79fe70dd9fac774a3548acc23a5f91cf4be1c246b4af2509",
    "RECOMMENDATION_PROMPT_TEMPLATES.NATURAL_LANGUAGE_TEXT": "0d590ed8fddc61eefd301ccafb2976a12ed7be8ae12b8e48a5851f8b818b30fe",
    "RECOMMENDATION_PROMPT_TEMPLATES.URL": "cc1ad260608c8e20139740f5e39c6710d649a3181c18741cc15d1632c3e1990b",
    "RECOMMENDER_CORE": "9ff0a52f59b333d59cbd15bcdf1ffba1ba029f94c2a446998cad9d8a0183c16e",
    "RECOMMENDER_TYPE_PARTS.BOOLEAN": "c4f91ad79efa7c9428e91d173b21c63489cb43ddbcc9992c5f306e6c87b95563",
    "RECOMMENDER_TYPE_PARTS.CATEGORICAL": "11ecbd84f7f69b894c284b4919b341eccec08a52f475b5a26da02592519c5df0",
    "RECOMMENDER_TYPE_PARTS.COLLECTION": "b9e1eaeb173f0e533cffc35c2abbf8c81c5f4cb10d51b7b46caa4d39585c3dfd",
    "RECOMMENDER_TYPE_PARTS.DATETIME": "7edb720a463d74517df6ecb1f9b905e052b493b7cfb9c42984b83a35ef1f44b7",
    "RECOMMENDER_TYPE_PARTS.DELIMITED_STRING": "798a21f99e18789a8071dc138a4d69f73c3011eaa71c8ecf436b364c012ba61d",
    "RECOMMENDER_TYPE_PARTS.DIRTY_FLOAT": "78d340418ca1731200c096a0c89637a81423d77ab45350aa50216c6e578f119e",
    "RECOMMENDER_TYPE_PARTS.DIRTY_INTEGER": "bf451fcdd9779fffb57c2237c16d0b2081b7ba27863571dacb799873cf93c4fc",
    "RECOMMENDER_TYPE_PARTS.DISCRETE_STRING": "f5763cbfbdf1440a1c9146f416cdc517fef71d4f67b302f1ed3811f2a2703089",
    "RECOMMENDER_TYPE_PARTS.EMAIL": "681699c08a4f9776933d4e1c416c93245fd1a42095a25990d415525e0ee84b6c",
    "RECOMMENDER_TYPE_PARTS.FLOAT": "f97ccf397f02c3a6e1fc021ac5bbb3bab1764d8c18b54eee1bb648b5a8c3fbec",
    "RECOMMENDER_TYPE_PARTS.IDENTIFIER": "4bd57a7ebde8ecf36c61e69ef94e0e323d802a2a73b29f6cd6c3a8b98274e3e9",
    "RECOMMENDER_TYPE_PARTS.INTEGER": "3da7ce2a56f6da40ef5e51d2d72f4e98b591a451858406391dd15bc971ad93bb",
    "RECOMMENDER_TYPE_PARTS.MIXED": "aca56d019e0aca0062ab03421a989bf28d64dba5551ba2238fbf1e52ce4786ab",
    "RECOMMENDER_TYPE_PARTS.NAMED_ENTITY": "6b63f8b5e1d27293702b4a9063f6e794c2f4629a8e37093ec3833d5737c0d2c4",
    "RECOMMENDER_TYPE_PARTS.NATURAL_LANGUAGE_TEXT": "3cdda4fabe0875da60288a97d982c5f287fd36881313b5da8b961d6092de3268",
    "RECOMMENDER_TYPE_PARTS.URL": "486b5f4dad099621f63a01143f0dd13fde5ddddc281480dfe11695fba76b5fe1",
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
