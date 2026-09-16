# Journal

One sentence per step: what was chosen, what was tested, what the reviewers found,
and what was done about it. It is written to be learned from, so plain English
throughout — someone who never saw the code should be able to follow it.

Who writes which lines:

- **Agent lines** — a hook copies them from each agent's report.
  `designer chose: ...`, `[F3] edge-hunter found: ...`, `final-check verdict: ...`
- **Gate lines** — `./run` writes one for every full run of a gate.
  `fast gate passed (9s): 61 passed in 9.12s`
- **Everything else** — the main thread, one sentence each:
  `you decided: ...` for an answer to a question, `built: ...` for what a slice
  delivered, and an answer to every finding: `F3 fixed: ...`, `F3 rejected: ...`
  or `F3 deferred: ...`

A slice cannot be committed as finished while any `[Fn]` finding is unanswered, or
while its section has no `built:` line.

## Before the spec (2026-09-16)

- you decided: MADClean v2 is built in a new private repo, madclean-v2, so the graded thesis repo msc-thesis-cosc stays untouched.
- built: imported the thesis repo with its full git history into madclean-v2 and tagged the last thesis commit as thesis-final, so every v2 change can be diffed against it.
- you decided: keep the code's behaviour where it differs from the paper (validator skipped for integer, float and boolean columns; multi-column validation off; config sample sizes), and do not change the paper.
- you decided: the evaluation keeps a paper-compatible scoring mode as the default and adds a strict-types mode next to it.
- you decided: no real LLM calls until the bug-fix slice is done, then one Beers run with the Gemini key to confirm.
- you decided: the GUI gets a more professional layout that keeps the current colours, shown first as a static preview before any GUI code is written.
- built: published GUI preview v1 (https://claude.ai/artifact/BrQSQZrshcV1G2tiiR487E) with a top bar for the run, a left rail ordered by workflow and a pipeline inspector, using real Beers data.
- env-doctor found: uv's Python 3.11 build is blocked from starting by Windows Smart App Control, while uv's 3.12 and 3.13 builds run fine.
- env-doctor found: Reflex downloads its own JavaScript runtime on first run, and whether Smart App Control lets it start is untested.
- env-doctor found: the repo pins no Python, Node or package-manager version anywhere.
- you decided: the preview is approved; column headers use the thin colour band, the GUI font is IBM Plex Sans, and a notebook (.ipynb) export is added.
- built: downloaded tax, adult and restaurants (dirty and ground truth) from hqahtan/COSC-Thesis-William into Code/data/benchmark_datasets; their git blob hashes match that repo and the row counts match the paper (200,000, 45,222 and 28,787).
- found: hqahtan/COSC-Thesis-William holds a later copy of 9 code files: it fixes the prompt brace crash, OpenAI structured output and the Qwen 9B model id, but its recommender returns no message history for already-clean columns, which breaks the human-review rejection path.
- found: git core.autocrlf is true on this machine, so committed CSVs are checked out with Windows line endings and are not byte-identical to the stored copies.
- you decided: start from the thesis code plus three fixes ported from the hqahtan repo, and do not port its change that drops recommender history.
- you decided: commit tax, adult and restaurants with plain git, not Git LFS or a download script.
- you decided: split the GUI into page modules and state per area, with per-session stop and review state.
- you decided: build the GUI last, after the fixes, data classes and evaluation slices.
- built: wrote .work/SPEC.md and .work/slices.json with slices 0 to 4, Python 3.12 via uv, pytest with hypothesis, ruff, and fast, full, smoke and live run tasks.

## Slice 0 — skeleton

- built: .gitattributes stops line-ending conversion for CSV and JSON files; after re-checking them out, beers_dirty.csv has the same git blob hash as the source repos.
- found: hypothesis 6.156 and later ship a compiled module that Windows Smart App Control blocks on this machine, so hypothesis is capped below 6.156 (6.155.7 is pure Python and runs).
- found: ruff 0.16's default rules report 450 style findings in the thesis code, so the lint gate selects only rules for code that cannot run (E9, F63, F7, F82), which pass today.
- fast gate failed (1s):   Caused by: An Application Control policy has blocked this file. (os error 4551)
- found: Smart App Control blocks the pytest.exe launcher in the virtual environment, while madclean.exe, reflex.exe, ruff.exe and python.exe run, so ./run calls pytest as `python -m pytest`.
- found: all 16 stored thesis runs (hospital, beers, movies, rayyan, 4 each) re-score to exactly the committed overall and per-column results with the thesis evaluation code.
- fast gate passed (15s): 2 passed, 21 deselected, 1 warning in 12.36s
- smoke gate passed (125s): gui: http://localhost:3000 and backend /ping answered
- full gate passed (236s): 23 passed, 2 warnings in 234.17s (0:03:54)
- found: Reflex downloads bun into %LOCALAPPDATA%\reflex\bun and it runs under Smart App Control, so the GUI starts on this machine.
- found: killing the GUI's processes one by one leaves a backend worker serving port 8000, so the smoke gate stops the whole process tree and fails if anything still answers afterwards.
- built: the smoke gate now also renders the page in headless Edge and fails on a missing MADClean title or any console error from the page.
- fast gate passed (12s): 2 passed, 21 deselected, 1 warning in 9.48s
- smoke gate passed (49s): gui: page rendered in the browser with no console errors; backend /ping answered
- built: slice 0 skeleton — Python 3.12 environment locked with uv, ./run with setup, fast, full, smoke and check tasks, golden-score and profiler tests, and the tax, adult and restaurants datasets; the full gate was not re-run after the smoke-gate edits because they only change how the GUI is started and stopped, which the full gate does not cover.
