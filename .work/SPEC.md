# MADClean v2

## Goal
A hardened, faster and better-structured copy of the MADClean thesis system (same cleaning behaviour), with one reproducible evaluation module and a rebuilt GUI from the approved preview.

## Out of scope
- Changing prompt wording (except escaping), the agent loop order, retry counts, FD threshold, type detection, sampling logic or LLM providers
- Changing the paper, or making code match the paper where they differ (code behaviour wins)
- New cleaning features; new GUI features beyond the approved preview and notebook export
- Real LLM calls inside any gate
- A type checker in the gates; mutation testing before the evaluation slice is done

## Environment
Measured by env-doctor on 2026-09-16.
- OS: Windows 11 Home 10.0.26200, native host (no WSL, no container)
- machine: 6 cores / 12 logical, ~15.8 GiB RAM (~8.5 GiB free), ~339 GiB free on C:
- uv 0.12.5: runs uv-managed CPython 3.12.14 and 3.13.15; ambient `py` is 3.14.3
- Node v24.18.0 on PATH; bun not installed; Docker Desktop 29.6.1 installed, daemon not running
- git 2.54.0, gh 2.98.0, git-lfs 3.7.1; git core.autocrlf=true
- ports 3000 and 8000 free
- pinned by: uv (`Code/.python-version` = 3.12, `Code/uv.lock`, `uv sync --frozen`)
- known blocker: uv's CPython 3.11 is blocked by Windows Smart App Control — worked around by using 3.12
- known blocker: Smart App Control blocks the venv's pytest.exe and hypothesis >= 6.156 — worked around by `python -m pytest` and hypothesis < 6.156
- Reflex downloads bun to %LOCALAPPDATA%\reflex\bun and it runs under Smart App Control (proved in slice 0)

## Stack
- package: `Code/madclean` — pandas 2.3.2, numpy 2.3.2, spaCy 3.8.7, openai 1.102.0, google-genai 1.39.1, pydantic 2.11.7, rapidfuzz 3.14.3, fuzzywuzzy 0.18.0 (existing pins)
- spaCy model: en_core_web_sm 3.8.0 pinned by wheel URL in the lock (no runtime download)
- GUI: Reflex 0.8.28.post1 (`Code/gui`)
- toolchain: uv + Python 3.12
- tests: pytest + hypothesis (< 6.156, see Environment)
- lint: ruff (check only, no reformatting of existing files; rules E9, F63, F7, F82 — a slice that fixes a rule's findings adds that rule)

## Decisions
- base: thesis repo msc-thesis-cosc (tag `thesis-final`) + three fixes ported from hqahtan/COSC-Thesis-William: `format_prompt_template` safe prompt filling, OpenAI structured-output handling in `llm_clients.py`, Qwen 9B model id `qwen/qwen3.5-9b`
- NOT ported from hqahtan: recommender returning `None` history for already-clean columns
- data files: `.gitattributes` marks `*.csv` as `-text` (byte-identical everywhere); tax, adult, restaurants committed with plain git
- layout: keep `Code/madclean`, `Code/gui`, `Code/evaluation`; tests in `Code/tests`; `./run` at repo root
- internal records: stdlib `@dataclass`; pydantic only for parsing model JSON; records are converted back to plain dicts at the GUI edge — `asdict` for trace events, and `CleaningReport.to_dict()` for the run report, because the GUI prints that dict and it must keep the thesis's flat shape, key order and absent-when-unset keys
- offline tests: scripted fake LLM client implementing `BaseLLMClient`
- one failing column: marked failed in the report, keeps original values, other columns continue
- evaluation: one module shared by CLI, GUI and benchmark scripts; paper mode is default and reproduces committed JSON exactly (0.0 where undefined); strict-types mode added; display shows "—" for undefined; per-run JSON plus mean and standard deviation
- GUI: split into page modules and state per area; cancel and review state per browser session, no module-level globals; thin colour-band column headers; IBM Plex Sans; existing colours; approved preview https://claude.ai/artifact/BrQSQZrshcV1G2tiiR487E
- notebook export: `.ipynb` written with the standard library
- git: one branch per slice, merged to `main` after both gates

## Overturned defaults
- none (user took every recommendation)

## Constraints
- thesis repo msc-thesis-cosc is never modified
- same cleaning behaviour as `thesis-final` unless a slice names the bug being fixed
- paper-mode scores for stored outputs must equal the committed results JSON to 4 decimals
- API keys only in `.env` (git-ignored); never in source or committed config
- ask before adding any dependency not listed in Stack

## Verification
Fast gate: ./run fast     # < 60s, no network — ruff (E9,F63,F7,F82); pytest -m "not slow": unit + property tests, fake-LLM pipeline on beers, re-score one stored beers run vs committed JSON — measured 12s warm (slice 0)
Full gate: ./run full     # background, minutes — ruff; pytest incl. slow: re-score all stored MADClean/ablation/baseline outputs vs committed JSON, profile all 7 datasets — measured 236s cold, sharing the machine with the smoke gate (slice 0)
Smoke:     ./run smoke    # `madclean --help`; start madclean-ui; wait for :3000 and :8000/ping; render in headless Edge, fail on console errors; stop and confirm nothing still answers (the GUI build is proved here) — measured 125s on first start (slice 0)
Live:      ./run live beers  # manual only, real Gemini calls, once after slice 1
Properties: scores in [0,1] or undefined; unchanged table → zero changes; cleaned == ground truth → P = R = 1; fast scorer == paper scorer; FD ordering respects dependencies; prompt filling never raises on braces in data
Mutation:  decided after slice 3, once per milestone on the evaluation module, never in a gate

## Skeleton (slice 0) — done means
- [x] `./run setup` creates the uv environment on Python 3.12 from `uv.lock` (frozen), including the spaCy model
- [x] `.gitattributes` in place; all 14 benchmark CSVs committed and byte-identical to their source repos
- [x] one test re-scores stored `beers_cleaned.csv` (run 1) and matches `detailed_results/eval_results_1.json`
- [x] `madclean --help` runs; `madclean-ui` serves a page on port 3000
- [x] both gates run, and their real durations are recorded here (fast 12s warm, full 236s, smoke 49s warm / 125s first start)

Proved by: both gates above, plus the smoke launch

## Slice 1 — done means
- [ ] a column typed COLLECTION no longer crashes prompt creation (property test with random braces)
- [ ] one column raising an exception leaves the other columns cleaned and that column marked failed
- [ ] `evaluation_pipeline.py` and `evaluate_files.py` import and run
- [ ] generated code runs off the event loop; a fake-LLM run with 10 columns is faster than the thesis code on the same fake
- [ ] `--save-cleaned` works on a fresh clone and keeps the input's file type; saving after a stopped run does not crash
- [ ] FD counts are stored under `violations_count` / `imputables_count`
- [ ] sampling takes a seed from config; two fake-LLM runs with the same seed give identical output
- [ ] the three hqahtan fixes are in, each with a test
- [ ] one live beers run with Gemini completes and its paper-mode scores are recorded in the journal

Proved by: both gates, plus the smoke launch and the live beers run

## Later slices
- slice 2: dataclasses for report, trace steps, token usage, registry, scores; duplicate report/trace blocks in `multi_agent_cleaning.py` removed; fake-LLM output identical before and after
- slice 3: evaluation module, strict mode, mean ± std, agent stats (attempts, validator rejections, tokens per column), one command re-scores all methods
- slice 4: GUI rebuild from the preview, notebook export, per-session state

## Open
- mutation testing tool on Windows — needed after slice 3
