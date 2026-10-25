"""An offline model for trying the app for free: it answers slowly, without any network call.

Switched on by setting MADCLEAN_DEMO_LLM_SECONDS to how long each reply takes. It keeps every
column as it is (identity code) and approves it, so a run always finishes.
"""

import asyncio
import json
import os

from madclean.components.multi_agent_cleaner.llm_recommending import (
    CodeOutputFDRecommendation,
    CodeOutputRecommendation,
)
from madclean.components.multi_agent_cleaner.llm_validation import CodeOutputValidation
from madclean.llm.llm_clients import BaseLLMClient

DEMO_SECONDS_VARIABLE = "MADCLEAN_DEMO_LLM_SECONDS"
IDENTITY_CODE = "def clean_column(data):\n    return data\n"


def demo_seconds() -> float | None:
    """The demo model's reply time, or None when the variable is unset or not a usable number."""
    raw = os.getenv(DEMO_SECONDS_VARIABLE)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return None
    return seconds


class SlowDemoClient(BaseLLMClient):
    """Answers each agent after `seconds`, as a model would: keep the column, identity code, approve."""

    def __init__(self, model_name: str = "demo", seconds: float = 0.0):
        super().__init__(model_name)
        self.seconds = seconds

    async def call_llm_async(self, messages, response_schema=None, temperature=None, top_p=None):
        answer = self._answer(response_schema)
        call_id = self._activity_start()
        try:
            await asyncio.sleep(self.seconds)
        finally:
            self._activity_finish(call_id)
        usage = {
            "input_tokens": sum(len(str(m.get("content", ""))) for m in messages),
            "output_tokens": len(answer),
        }
        return answer, usage

    @staticmethod
    def _answer(response_schema) -> str:
        if response_schema == "text/plain":
            return f"```python\n{IDENTITY_CODE}```"
        if response_schema is CodeOutputRecommendation:
            return json.dumps({
                "is_clean": False,
                "summary": "Demo recommendation: keep the column as it is.",
                "error_types": ["formatting"],
                "examples_clean": [],
                "examples_dirty": [],
                "cleaning_instructions": ["Return the column unchanged."],
            })
        if response_schema is CodeOutputFDRecommendation:
            return json.dumps({
                "analysis": "Demo dependency analysis.",
                "summary": "Demo recommendation: leave the dependency as it is.",
                "corrections": None,
                "skipped_lhs_values": None,
                "impute_missing": False,
            })
        if response_schema is CodeOutputValidation:
            return json.dumps({"needs_correction": False, "feedback_target": None, "correction_instructions": ""})
        raise ValueError(f"The demo model has no answer for response schema {response_schema!r}")
