from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CleaningConfig:
    verbose: bool = True

    enable_validation: bool = True
    enable_user_validation: bool = False
    enable_validation_multi: bool = False
    enable_multi_col_cleaning: bool = True

    # Human-in-the-loop (GUI): optional code review after coder and review after LLM validator.
    human_in_the_loop: bool = False
    hitl_apply_to_all_columns: bool = True
    hitl_column_list: list[str] = field(default_factory=list)
    # Per-column extra instructions merged into the recommender prompt (GUI table headers).
    recommender_column_hints: dict[str, str] = field(default_factory=dict)
    # Optional per-column few-shot / labeled examples text merged into recommender prompts (before OUTPUT FORMAT).
    recommender_column_labeled_examples: dict[str, str] = field(default_factory=dict)

    # LLM sampling (omit from API request when None to use provider defaults).
    llm_temperature: Optional[float] = 0.2
    llm_top_p: Optional[float] = None

    # sampling
    sample_sizes: dict = field(default_factory=lambda: {
        "NUMERIC": {
            "clean_sample_size": 50,
            "dirty_sample_size": 500
        },
        "DATETIME": {
            "clean_sample_size": 100,
            "dirty_sample_size": 500
        },
        "DIRTY_NUMERIC": { 
            "random_sample_size": 50,
            "unique_sample_size": 500
        },
        "STRING": { 
            "random_sample_size": 150,
            "unique_sample_size": 500
        },
        "NLT": {
            "short_sample_size": 100,
            "long_sample_size": 20
        }
    })

    # None keeps sampling random on every run; a number makes the samples each agent sees repeatable.
    sampling_seed: Optional[int] = None

    sample_size_validator: int = 150
    sample_size_validator_random: int = 60
    sample_size_validator_changed: int = 90
    validator_failure_strategy: str = "accept_cleaned"  # accept_cleaned|leave_uncleaned|ask_user

    # retries
    max_cleaning_attempts: int = 6
    max_multi_col_attempts: int = 3
    max_parse_attempts: int = 3
    max_coding_attempts: int = 3

    # code checks
    # Code that empties more than this share of the filled cells (placeholders left out) goes back to the coder.
    max_emptied_share: float = 0.5
    # Dependency violations listed in the recommender prompt, most rows first; the rest stay unchanged.
    max_fd_violations_in_prompt: int = 100

    # concurrency
    semaphore_limit: int = 15

    # additional
    include_metadata: bool = True

    # Columns that should be skipped entirely by the LLM pipeline.
    # Used by the GUI when the user marks a column as "already clean".
    skip_columns: list[str] = field(default_factory=list)


