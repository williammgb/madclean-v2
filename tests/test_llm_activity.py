"""The live record of open model calls, and the offline demo model that lets it be seen for free."""

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest
from google.genai import errors as genai_errors

from fake_llm import FakeLLMClient
from madclean.config.loader import load_default_cleaning_config
from madclean.llm.activity import LLMActivity, current_agent, current_attempt, current_task
from madclean.llm.demo_client import SlowDemoClient
from madclean.llm.llm_clients import GeminiClient
from madclean.pipeline import Pipeline

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_a_started_call_is_open_with_who_made_it_and_for_how_long():
    clock = _Clock()
    activity = LLMActivity(clock=clock)

    call_id = activity.start("Coder", "price", 2, "gemini-x")
    clock.now += 7.5

    assert activity.snapshot() == {
        "open": [{
            "agent": "Coder", "task": "price", "attempt": 2, "model": "gemini-x",
            "seconds": 7.5, "try_no": 1, "tries": None, "reason": None,
        }],
        "last_reply_seconds": None,
    }
    assert isinstance(call_id, int)


def test_a_retry_shows_which_try_and_why():
    activity = LLMActivity(clock=_Clock())
    call_id = activity.start("Validator", "city", 1, "m")

    activity.retrying(call_id, 2, 3, "rate limited")

    (call,) = activity.snapshot()["open"]
    assert (call["try_no"], call["tries"], call["reason"]) == (2, 3, "rate limited")


def test_a_finished_call_is_gone_and_counts_as_the_last_reply():
    clock = _Clock()
    activity = LLMActivity(clock=clock)
    first = activity.start("Recommender", "a", 1, "m")
    second = activity.start("Coder", "b", 1, "m")

    clock.now += 3
    activity.finish(first)
    clock.now += 2

    snap = activity.snapshot()
    assert [call["task"] for call in snap["open"]] == ["b"]
    assert snap["last_reply_seconds"] == 2
    activity.finish(second)
    activity.finish(second)  # finishing twice is harmless
    activity.retrying(second, 2, 3, "timed out")  # and so is a retry report for a closed call
    assert activity.snapshot()["open"] == []


def test_every_call_gets_its_own_id_when_started_from_many_threads():
    import threading

    activity = LLMActivity()
    ids: list[int] = []

    def start_many():
        for _ in range(200):
            ids.append(activity.start("Coder", "t", 1, "m"))

    threads = [threading.Thread(target=start_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(ids)) == 1600
    assert len(activity.snapshot()["open"]) == 1600


def _gemini_with(generate):
    """A GeminiClient whose one-shot model call is replaced, as tests/test_llm_clients.py does."""
    client = GeminiClient.__new__(GeminiClient)
    client.model_name = "gemini-test"
    client.activity = LLMActivity()
    client._generate = generate
    return client


def _in_context(coro, agent="Coder", task="price", attempt=2):
    async def run():
        current_agent.set(agent)
        current_task.set(task)
        current_attempt.set(attempt)
        return await coro()

    return asyncio.run(run())


def test_a_gemini_call_is_open_while_it_runs_and_closed_after():
    seen = []

    async def generate(contents, config):
        seen.append(client.activity.snapshot()["open"])
        usage = SimpleNamespace(prompt_token_count=3, total_token_count=5)
        return SimpleNamespace(text="ok", usage_metadata=usage)

    client = _gemini_with(generate)

    content, _ = _in_context(lambda: client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert content == "ok"
    ((call,),) = seen
    assert (call["agent"], call["task"], call["attempt"], call["model"]) == ("Coder", "price", 2, "gemini-test")
    assert client.activity.snapshot()["open"] == []
    assert client.activity.snapshot()["last_reply_seconds"] is not None


def test_a_gemini_call_that_raises_leaves_no_open_call():
    seen = []

    async def generate(contents, config):
        seen.append(len(client.activity.snapshot()["open"]))
        raise RuntimeError("provider down")

    client = _gemini_with(generate)

    with pytest.raises(RuntimeError, match="provider down"):
        _in_context(lambda: client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert seen == [1]
    assert client.activity.snapshot()["open"] == []


def test_the_gemini_retry_loop_reports_each_retry_and_its_reason():
    unavailable = genai_errors.ServerError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}})
    rate_limited = genai_errors.ClientError(429, {"error": {"message": "slow", "status": "RESOURCE_EXHAUSTED"}})
    outcomes = [unavailable, rate_limited, "hang", "ok"]
    seen = []

    class Models:
        async def generate_content(self, **kwargs):
            seen.append([(c["try_no"], c["tries"], c["reason"]) for c in client.activity.snapshot()["open"]])
            outcome = outcomes.pop(0)
            if outcome == "hang":
                await asyncio.sleep(3600)
            if isinstance(outcome, Exception):
                raise outcome
            usage = SimpleNamespace(prompt_token_count=3, total_token_count=5)
            return SimpleNamespace(text=outcome, usage_metadata=usage)

    client = GeminiClient.__new__(GeminiClient)
    client.model_name = "gemini-test"
    client.timeout_seconds = 0.05
    client.retry_wait_seconds = 0
    client.attempts = 4
    client.activity = LLMActivity()
    client.async_client = SimpleNamespace(models=Models())

    content, _ = _in_context(lambda: client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert content == "ok"
    assert seen == [
        [(1, None, None)],
        [(2, 4, "server error")],
        [(3, 4, "rate limited")],
        [(4, 4, "timed out")],
    ]
    assert client.activity.snapshot()["open"] == []


def test_a_client_without_a_record_records_nothing_and_still_answers():
    async def generate(contents, config):
        usage = SimpleNamespace(prompt_token_count=3, total_token_count=5)
        return SimpleNamespace(text="ok", usage_metadata=usage)

    client = _gemini_with(generate)
    client.activity = None

    content, _ = asyncio.run(client.call_llm_async(MESSAGES, response_schema="text/plain"))

    assert content == "ok"


def _pipeline(fake=None):
    config = load_default_cleaning_config()
    config.verbose = False
    config.sampling_seed = 7
    config.enable_multi_col_cleaning = False
    llm = (fake or FakeLLMClient()).llm_config()
    return Pipeline(
        llm_config=llm,
        agent_llm_configs={"recommender": llm, "coding": llm, "validation": llm},
        config=config,
        verbose=False,
    )


def test_two_pipelines_keep_separate_records(monkeypatch):
    monkeypatch.delenv("MADCLEAN_DEMO_LLM_SECONDS", raising=False)
    first = _pipeline(FakeLLMClient())
    second = _pipeline(FakeLLMClient())

    first.activity.start("Coder", "a", 1, "m")

    assert first.activity is not second.activity
    assert len(first.activity.snapshot()["open"]) == 1
    assert second.activity.snapshot()["open"] == []
    assert first.client.activity is first.activity
    assert second.client.activity is second.activity


def _small_table(tmp_path):
    # Two text columns: a whole-number column whose code changes nothing is kept without asking the
    # Validator, so it could never show an approval.
    frame = pd.DataFrame(
        {
            "letters": ["anna ", "Bram", "cees", "Dirk ", "eva", "Femke", "gijs", "Hanna"],
            "cities": ["Utrecht", "utrecht ", "Gouda", "Delft", "delft", "Breda", "Ede", "Assen"],
        }
    )
    path = tmp_path / "small_dirty.csv"
    frame.to_csv(path, index=False)
    return path


def test_the_demo_model_runs_a_whole_table_offline_and_approves_every_column(tmp_path, monkeypatch):
    monkeypatch.setenv("MADCLEAN_DEMO_LLM_SECONDS", "0")
    pipeline = _pipeline()

    agents = pipeline.multi_agent_loop
    clients = [agents.recommender_agent.llm_client, agents.coding_agent.llm_client, agents.validation_agent.llm_client]
    assert all(isinstance(client, SlowDemoClient) for client in clients)
    assert all(client.activity is pipeline.activity for client in clients)

    _, report = pipeline.run(file_path=str(_small_table(tmp_path)))

    assert set(report.entries) == {"letters", "cities"}
    for entry in report.entries.values():
        assert entry.cleaned and entry.cleaning_validated
        assert entry.trace_steps[-1].status == "approved"
    assert pipeline.activity.snapshot()["open"] == []
    assert pipeline.activity.snapshot()["last_reply_seconds"] is not None


def test_the_demo_model_shows_its_call_as_open_while_it_waits():
    client = SlowDemoClient(seconds=0.05)
    client.activity = LLMActivity()

    async def run():
        current_agent.set("Validator")
        task = asyncio.ensure_future(client.call_llm_async(MESSAGES, response_schema="text/plain"))
        await asyncio.sleep(0.01)
        during = client.activity.snapshot()["open"]
        await task
        return during

    (call,) = asyncio.run(run())
    assert (call["agent"], call["model"]) == ("Validator", "demo")
    assert client.activity.snapshot()["open"] == []


@pytest.mark.parametrize("value", ["soon", "", "-1", "nan", "inf"])
def test_a_variable_that_is_not_a_usable_number_is_ignored(monkeypatch, value):
    monkeypatch.setenv("MADCLEAN_DEMO_LLM_SECONDS", value)
    fake = FakeLLMClient()

    pipeline = _pipeline(fake)

    assert pipeline.client is fake
    assert pipeline.multi_agent_loop.coding_agent.llm_client is fake
