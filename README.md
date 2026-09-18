# MADClean — Multi-Agent Data Cleaning

MADClean is an automated tabular data cleaning framework that combines statistical data
profiling with the semantic reasoning and code generation abilities of large language models.
The profiler extracts column types, outliers and functional dependencies; a multi-agent LLM
loop (Recommender → Coder → Validator) turns those findings into executable cleaning code and
validates the result before it is applied.

```
madclean/          the framework: profiler, coordinator, multi-agent cleaner, LLM clients
evaluation/        the benchmark runner, the scorer, baselines and their stored results
gui/               the Reflex web interface
tests/             the test suite (fast tests, plus real-size tests marked slow)
data/              the seven benchmark datasets, dirty and ground truth
docs/              screenshots used by this README
run                the single entrypoint: setup, fast, full, smoke, check, live
```

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
git clone https://github.com/williammgb/madclean-v2.git
cd madclean-v2
./run setup
```

`./run setup` installs the locked dependency set from `uv.lock`, including the spaCy model
`en_core_web_sm`. Nothing else needs to be installed by hand.

<details>
<summary>Installing with pip instead</summary>

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install .
python -m spacy download en_core_web_sm
```
</details>

## Configuring an LLM

Create a `.env` file in the repository root with the key for the provider you want to use:

```
GEMINI_API_KEY="your_api_key_here"
# or
OPENAI_API_KEY="your_api_key_here"
# or
OPENROUTER_API_KEY="your_api_key_here"
```

`.env` is git-ignored: keys never enter the repository. The default provider is the
`LLM_CLIENT_NAME` entry in `madclean/llm/llm_settings.py`; every model MADClean knows about is
listed in `madclean/llm/llm_registry.py`.

### Adding a provider

1. Add a client class in `madclean/llm/llm_clients.py`, following `BaseLLMClient`.
2. Register it in `madclean/llm/llm_registry.py` as an `LLMSpec` with its API key name.
3. Select it in the UI, per agent on the command line, or as the new `LLM_CLIENT_NAME`.

## Usage

```bash
madclean path/to/file.csv [OPTIONS]
```

| Option | Effect |
| --- | --- |
| `-v`, `--verbose` | print profiling and per-agent progress |
| `--save-cleaned` | write the cleaned dataset to `data/cleaned` |
| `--llm-recommender` | LLM client for the Recommender agent |
| `--llm-coding` | LLM client for the Coder agent |
| `--llm-validation` | LLM client for the Validator agent |
| `--help` | full usage |

An omitted `--llm-*` option falls back to `LLM_CLIENT_NAME`.

```bash
madclean data/benchmark_datasets/beers_dirty.csv --verbose --save-cleaned
madclean data/benchmark_datasets/beers_dirty.csv --llm-recommender Gemini --llm-coding OpenAI
```

## The web interface

```bash
madclean-ui
```

Reflex prints the URLs it serves on, normally a frontend on `http://localhost:3000/` and a
backend on `http://localhost:8000`.

- **Upload data** — load a CSV or Excel file and preview the table.
- **Model selection** — pick an LLM per agent, or `USER` to validate by hand.
- **Advanced configuration** — cleaning and validation settings, sample sizes, human-in-the-loop.
- **Run pipeline** — live logs and progress, with step-level detail per column.
- **Usage and report** — tokens, runtime and the generated cleaning code.
- **Evaluation** — upload a ground-truth file with the same columns and row count to score a run.
- **Edit and download** — correct cells in the cleaned table and export the result.

![The MADClean interface](docs/main_window.png)

## Development

| Command | What it does |
| --- | --- |
| `./run setup` | install Python 3.12 and the locked dependencies |
| `./run fast` | lint, then every test not marked slow (no network) |
| `./run full` | lint, then the whole suite at real size |
| `./run smoke` | start the CLI and the UI and check that both answer |
| `./run check <file.py>` | lint one file |
| `./run live <dataset>` | one real-model benchmark run — this costs API tokens |

No test in `./run fast` or `./run full` calls a real model; they all run against a scripted fake
client, so the suite is free to run and deterministic.

## Security note

MADClean executes code written by a language model. It does **not** sandbox that code. Run it
only with models and in environments you trust.

## License

MIT — see [LICENSE](LICENSE).
