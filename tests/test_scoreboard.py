"""`./run score`: one offline command that re-scores everything stored in this repository."""

import json
import math

import pytest

from madclean.evaluation import BENCHMARKS, Mode
from madclean.evaluation import scoreboard


def test_it_finds_every_stored_run_in_order():
    files = scoreboard.cleaned_files("madclean", "beers")
    assert [path.name for path in files] == [
        "beers_cleaned.csv",
        "beers_cleaned_2.csv",
        "beers_cleaned_3.csv",
        "beers_cleaned_4.csv",
    ]
    # A baseline stores one run per dataset.
    assert [path.name for path in scoreboard.cleaned_files("raha_baran", "beers")] == [
        "beers_cleaned.csv"
    ]


def test_a_method_without_stored_output_is_left_out():
    assert scoreboard.cleaned_files("madclean", "tax") == []
    assert scoreboard.score_method("madclean", BENCHMARKS["beers"], Mode.PAPER) is not None


def test_a_method_that_only_detects_is_scored_from_its_mask():
    """SAGED flags cells and never repairs them, so it stores a mask, not a cleaned file."""
    assert scoreboard.cleaned_files("saged", "beers") == []
    assert scoreboard.detection_file("saged", "beers") is not None

    scored = scoreboard.score_method("saged", BENCHMARKS["beers"], Mode.PAPER)

    assert scored["detection_only"] is True
    assert 0.0 < scored["detection"]["mean"] < 1.0
    assert scored["correction"]["mean"] == 0.0
    assert 0.0 < scored["detection_recall"] <= 1.0


def test_madclean_first_then_the_baselines():
    names = scoreboard.methods()
    assert names[0] == "madclean"
    assert "raha_baran" in names
    assert names[1:] == sorted(names[1:])


def test_the_paper_scores_it_prints_are_the_committed_ones():
    """The command re-scores; it does not read the stored numbers, so they have to agree."""
    scored = scoreboard.score_method("madclean", BENCHMARKS["beers"], Mode.PAPER)
    committed = json.loads(
        (scoreboard.RESULTS_DIR / "beers" / "avg_eval_results.json").read_text(encoding="utf-8")
    )

    assert scored["runs"] == 4
    assert math.isclose(
        scored["detection"]["mean"], committed["detection_metrics"]["f1_score"], abs_tol=5e-5
    )
    assert math.isclose(
        scored["correction"]["mean"], committed["correction_metrics"]["f1_score"], abs_tol=5e-5
    )
    assert scored["detection"]["standard_deviation"] >= 0.0


@pytest.mark.slow
def test_the_whole_scoreboard_builds_and_prints(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(scoreboard, "SCOREBOARD_PATH", tmp_path / "scoreboard.json")
    assert scoreboard.main([]) == 0

    printed = capsys.readouterr().out
    assert "PAPER MODE" in printed and "STRICT MODE" in printed
    assert "madclean" in printed

    board = json.loads((tmp_path / "scoreboard.json").read_text(encoding="utf-8"))
    assert set(board) == {"paper", "strict"}
    assert set(board["paper"]) == set(BENCHMARKS)
    assert board["paper"]["beers"]["madclean"]["runs"] == 4
