import copy
import json

from madclean.components.multi_agent_cleaner.llm_recommending import (
    CodeOutputFDRecommendation,
    CodeOutputRecommendation,
)
from madclean.components.multi_agent_cleaner.llm_validation import CodeOutputValidation
from madclean.llm.llm_clients import BaseLLMClient

IDENTITY_CODE = "def clean_column(data):\n    return data\n"


class FakeLLMClient(BaseLLMClient):
    """Scripted stand-in for a real model: answers by agent and records every prompt."""

    def __init__(self, coder_code: str = IDENTITY_CODE):
        super().__init__(model_name="fake")
        self.coder_code = coder_code
        self.calls: list[tuple[str, list[dict]]] = []

    async def call_llm_async(self, messages, response_schema=None, temperature=None, top_p=None):
        kind = self._agent_for(response_schema)
        self.calls.append((kind, copy.deepcopy(messages)))
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
            return json.dumps({
                "summary": "Fake dependency recommendation.",
                "violation_instructions": "Leave conflicting values unchanged.",
                "imputation_instructions": "Leave missing values unchanged.",
            })
        return json.dumps({"needs_correction": False, "feedback_target": None, "correction_instructions": ""})

    def calls_for(self, kind: str) -> list[list[dict]]:
        return [messages for call_kind, messages in self.calls if call_kind == kind]

    def llm_config(self) -> dict:
        """A registry-shaped entry, as Pipeline expects from LLM_CLIENT_MAP."""
        return {"class": lambda model_name: self, "default_model": "fake", "role": "assistant"}
