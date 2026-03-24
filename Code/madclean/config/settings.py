from dataclasses import dataclass, field
@dataclass
class CleaningConfig:
    verbose: bool = True

    enable_validation: bool = True
    enable_validation_multi: bool = False
    enable_multi_col_cleaning: bool = True

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
            "unique_sample_size": 600
        },
        "NLT": {
            "short_sample_size": 100,
            "long_sample_size": 20
        }
    })

    sample_size_validator: int = 150

    # retries
    max_cleaning_attempts: int = 5
    max_multi_col_attempts: int = 3
    max_parse_attempts: int = 3
    max_coding_attempts: int = 3

    # concurrency
    semaphore_limit: int = 15

    # additional
    include_metadata: bool = True
    

