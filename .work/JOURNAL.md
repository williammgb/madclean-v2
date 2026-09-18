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

## Slice 1 — bug fixes and offline end-to-end test

- [F1] designer found: returned no journal lines (its reply was 3671 characters) — read the reply, or relaunch it if it died
- F1 rejected: the designer finished; its full plan and journal block arrived as a hand-back message rather than in the reply the hook read, and the plan is saved to .work/PLAN.md.
- designer chose: an offline fake beers run is written first and must pass on the unchanged thesis code, so every fix is checked against the thesis's cleaning behaviour.
- designer chose: generated cleaning code runs in a worker thread around the unchanged subprocess call, with its input columns copied on the event loop first.
- designer chose: a column or FD task that raises is marked failed with the error as its reason and keeps its original values; stop requests still pass through.
- designer chose: seeded sampling builds a separate random generator from the seed plus purpose, column and attempt, because columns run at the same time.
- designer chose: "faster than the thesis" is measured by rerunning the same 10-column fake run with the thesis's blocking code execution swapped back in, as a slow test.
- designer ruled out: asyncio's own subprocess API, because on Windows it needs one specific event-loop type that the GUI's worker thread is not guaranteed to have.
- [F2] plan-check found: returned no journal lines (its reply was 27 characters) — read the reply, or relaunch it if it died
- F2 rejected: plan-check finished; its report arrived as a hand-back message and found no contradictions across 18 claims compared between the spec and the plan.
- plan-check: 18 claims compared between .work/SPEC.md and .work/PLAN.md, none in conflict.
- you decided: prompt filling renders every existing template exactly as the thesis did (doubled braces become single); only literal braces such as {'k': 1} that crashed now pass through.
- you decided: with no sampling seed set, samples stay random like the thesis; a number in configurations.json makes runs repeatable.
- live nosuch failed (8s): live: unknown dataset 'nosuch'. Choose one of: hospital, beers, movies, rayyan
- fast gate passed (59s): 77 passed, 22 deselected, 1 warning in 56.09s
- found: the fake-model beers run passed on the unchanged thesis code in 17.3s with the table returned unchanged, which is the reference every fix was checked against.
- found: the outlier template formats median and mad with numeric specs ({median:g}, {mad:.4f}), so the prompt-filling property test passes numbers for those two fields instead of text.
- built: prompt filling now keeps literal braces such as {'k': 1}; property tests show it never raises on brace-heavy data and renders every other template exactly like str.format, and a hash test pins all 22 templates to their thesis text.
- built: a column or dependency task that raises is reported failed with its error and keeps its original values while the rest of the run continues; both tests fail on the thesis coordinator and pass now.
- built: generated cleaning code runs in a worker thread, so the fake beers run dropped from 17.3s to about 7s and 10 columns take 3.1s instead of 22.4s with the thesis's blocking execution.
- built: saving creates data/cleaned, keeps the input's file type for numbered copies, and skips saving when a stopped run returns no table; FD counts are stored under violations_count and imputables_count.
- built: an optional sampling_seed makes profiling, validator and review samples repeatable per column and attempt; two same-seed runs send identical prompts and give identical results, and without a seed sampling calls the same random functions as the thesis.
- built: ported hqahtan's OpenAI structured-output handling and the Qwen 9B model id, each with tests; the two thesis evaluation scripts import and run again; `./run live <dataset>` runs one real-model benchmark and journals its scores.
- built: to keep the fast gate under 60s, the two full same-seed beers runs moved to the full gate and a 300-row, 3-column beers slice pair stays in the fast gate (fast gate 50s warm, 77 tests).
- smoke gate passed (50s): gui: page rendered in the browser with no console errors; backend /ping answered
- [F3] plan-drift found: returned no journal lines (its reply was 27 characters) — read the reply, or relaunch it if it died
- [F4] edge-hunter found: returned no journal lines (its reply was 309 characters) — read the reply, or relaunch it if it died
- [F5] final-check found: returned no journal lines (its reply was 2534 characters) — read the reply, or relaunch it if it died
- full gate passed (319s): 100 passed, 2 warnings in 315.60s (0:05:15)
- F3, F4, F5 rejected: plan-drift, edge-hunter and final-check all finished; their reports and journal blocks arrived as hand-back messages the hook did not read, and their findings are recorded below by hand.
- plan-drift: all 33 changed files match the plan's file list and steps, including the deliberate fast-gate fallback (b); no skipped, loosened or deleted tests.
- [F6] plan-drift found: .work/slices.json is not ticked yet although the plan lists it as modified.
- F6 fixed: slices.json is ticked with gate evidence in the commit that closes the slice.
- [F7] plan-drift found: live_run.py adds an unknown-dataset check (exit 2) that the plan's step 12 did not name.
- F7 rejected: it is a two-line guard that turns a KeyError into a clear message, inside the file the plan creates, and changes nothing else.
- [F8] edge-hunter found: the OpenAI create path returns None content unchanged, so validator and recommender parsing call .strip() on None.
- F8 rejected: this is thesis behaviour and not a crash — the AttributeError is caught by validate_async's failure strategy and by generate_recommendations_async, which retries with its "not valid JSON" message.
- [F9] edge-hunter found: a duplicate column name would make single-column code execution receive a DataFrame instead of a Series.
- F9 rejected: every loader (pandas read_csv, read_excel, read_json records) renames or rejects duplicate headers, so a dataset with duplicate column names cannot reach the pipeline.
- [F10] edge-hunter found: sampling_seed strings such as "1e5" or "3.0" fall back to no seed silently.
- F10 rejected: the config loader already falls back to the default for every invalid value of every field; a seed is documented as a whole number or null.
- [F11] edge-hunter found: two failing tasks with the same report key would overwrite each other's failure entry.
- F11 rejected: report keys are column names, which are unique after loading, and dependency keys "lhs → rhs", where merging keeps one rule per right-hand column; the thesis success and failure paths already key the report the same way.
- final-check verdict: SHIP, on condition that the full gate passes, which it did (100 passed).
- [F12] final-check found: live_run.py prints a passing score line even when some columns failed inside the run.
- F12 deferred: until `./run live beers` runs; its log is then read for "FAILED cleaning" lines and a non-trivial token count before the live check is ticked.
- [F13] final-check found: a "cleaner not configured" configuration error now shows up as one failed dependency task instead of stopping the run.
- F13 rejected: the pipeline always registers the FD cleaner, so this error cannot happen through Pipeline, and if it did the failure is reported with its reason instead of discarding every cleaned column.
- [F14] designer found: returned no journal lines (its reply was 3152 characters) — read the reply, or relaunch it if it died
- F14 rejected: this is the slice 2 designer launched in the fan-out; it finished and its plan arrived as a hand-back message, saved to .work/PLAN.next.md for slice 2.

## Slice 2 — dataclasses instead of dicts (not started)

- you decided: the per-call token counts each model client returns stay a plain dict; only the totals per agent become a dataclass.
- you decided: only the thesis scorer gets typed scores in slice 2; the GUI scorer waits for slice 3, which replaces it.
- you decided: the two identical copies of the human-review sampling code are merged in slice 2; the third copy, which only dependency tasks would reach, stays as it is.

## Slice 1 — bug fixes and offline end-to-end test
- [F15] plan-check found: returned no journal lines (its reply was 38 characters) — read the reply, or relaunch it if it died
- F15 rejected: this plan-check ran against the slice 2 plan (the heading above it is the hook's, slice 1 is unchanged); it finished and its report arrived as a hand-back message with the two findings below.
- [F16] plan-check found: the spec says records convert to dicts with `asdict` at the GUI edge, but the slice 2 plan uses a hand-written `CleaningReport.to_dict()` for the report, and that deviation was not recorded in the spec.
- F16 fixed: the spec's records decision now says trace events use `asdict` and the run report uses `CleaningReport.to_dict()`, because the GUI prints that dict and it has to keep the thesis's flat shape and key order.
- [F17] plan-check found: the slice 2 done-means demands 7 equivalence scenarios in the fast gate, although the plan's own fallback can move 3 of them to the full gate.
- F17 fixed: that done-means line now requires all 8 scenarios to pass, with 7 in the fast gate, or 4 if the fallback is applied, and all 8 in the full gate.
