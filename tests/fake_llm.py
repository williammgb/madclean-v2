import copy
import json

from madclean.components.multi_agent_cleaner.llm_recommending import (
    CodeOutputFDRecommendation,
    CodeOutputRecommendation,
)
from madclean.components.multi_agent_cleaner.llm_validation import CodeOutputValidation
from madclean.llm.llm_clients import BaseLLMClient
from madclean.llm.llm_registry import LLMSpec

IDENTITY_CODE = "def clean_column(data):\n    return data\n"


class ServiceUnavailable(RuntimeError):
    """What a provider raises when it is overloaded; the agents map status 503 to an api_unavailable step."""

    status_code = 503


def first_user_text(messages) -> str:
    """The first user turn of a prompt, which is where the agents name the column or dependency."""
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


class FakeLLMClient(BaseLLMClient):
    """Scripted stand-in for a real model: answers by agent and records every prompt."""

    def __init__(self, coder_code: str = IDENTITY_CODE, script=None):
        super().__init__(model_name="fake")
        self.coder_code = coder_code
        self.script = script
        self.calls: list[tuple[str, list[dict]]] = []

    async def call_llm_async(self, messages, response_schema=None, temperature=None, top_p=None):
        kind = self._agent_for(response_schema)
        self.calls.append((kind, copy.deepcopy(messages)))
        answer = self.script(kind, messages) if self.script is not None else None
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            answer = self._answer(kind)
        usage = {
            "input_tokens": sum(len(str(m.get("content", ""))) for m in messages),
            "output_tokens": len(answer),
        }
        return answer, usage

    @staticmethod
    def _agent_for(response_schema) -> str:
        if response_schema == "text/plain":
            return "coder"
        if response_schema is CodeOutputRecommendation:
            return "recommender"
        if response_schema is CodeOutputFDRecommendation:
            return "fd_recommender"
        if response_schema is CodeOutputValidation:
            return "validator"
        raise AssertionError(f"FakeLLMClient got an unknown response schema: {response_schema!r}")

    def _answer(self, kind: str) -> str:
        if kind == "coder":
            return f"```python\n{self.coder_code}```"
        if kind == "recommender":
            return json.dumps({
                "is_clean": False,
                "summary": "Fake recommendation.",
                "error_types": ["formatting"],
                "examples_clean": [],
                "examples_dirty": [],
                "cleaning_instructions": ["Return the column unchanged."],
            })
        if kind == "fd_recommender":
            # No corrections and no imputation: the dependency is left as it is.
            return json.dumps({
                "analysis": "Fake dependency analysis.",
                "summary": "Fake dependency recommendation.",
                "corrections": None,
                "skipped_lhs_values": None,
                "impute_missing": False,
            })
        return json.dumps({"needs_correction": False, "feedback_target": None, "correction_instructions": ""})

    def calls_for(self, kind: str) -> list[list[dict]]:
        return [messages for call_kind, messages in self.calls if call_kind == kind]

    @property
    def coder_calls(self) -> int:
        """How many times the Coder was asked for code."""
        return len(self.calls_for("coder"))

    def llm_config(self) -> LLMSpec:
        """A registry entry, as Pipeline expects from LLM_CLIENT_MAP."""
        return LLMSpec(
            client_class=lambda model_name: self,
            default_model="fake",
            role="assistant",
            api_key_name="FAKE_API_KEY",
        )
