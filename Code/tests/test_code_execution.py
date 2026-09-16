import asyncio
import time

import pandas as pd

from madclean.components.multi_agent_cleaner.llm_coding import LLMCodingAgent

SLEEPING_CODE = "import time\n\ndef clean_column(data):\n    time.sleep(1)\n    return data\n"


def _run_with_heartbeat(code: str, df: pd.DataFrame, columns):
    """Runs generated code while a coroutine ticks every 50 ms; returns the result and the tick gaps."""

    async def scenario():
        gaps: list[float] = []
        finished = asyncio.Event()

        async def heartbeat():
            last = time.perf_counter()
            while not finished.is_set():
                await asyncio.sleep(0.05)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        result = await LLMCodingAgent._execute_code_async(code, df, columns)
        finished.set()
        await beat
        return result, gaps

    return asyncio.run(scenario())


def test_event_loop_keeps_running_while_generated_code_runs():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    result, gaps = _run_with_heartbeat(SLEEPING_CODE, df, "a")

    pd.testing.assert_series_equal(result, df["a"])
    assert len(gaps) >= 10
    assert max(gaps) < 0.5


def test_multi_column_code_gets_only_its_columns_and_returns_a_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
    code = "def clean_column(data):\n    assert list(data.columns) == ['a', 'b']\n    return data\n"

    result, _ = _run_with_heartbeat(code, df, ["a", "b"])

    pd.testing.assert_frame_equal(result, df[["a", "b"]])
