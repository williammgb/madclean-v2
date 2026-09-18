import os
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
### Local imports
from madclean.pipeline import Pipeline
from madclean.llm.llm_settings import LLM_CLIENT_NAME
from madclean.llm.llm_registry import LLM_CLIENT_MAP, LLMSpec

def run_ui() -> int:
    """Launch the Reflex UI from the repository GUI folder."""
    repo_code_dir = Path(__file__).resolve().parents[1]
    gui_dir = repo_code_dir / "gui"
    if not gui_dir.exists():
        raise FileNotFoundError(f"Could not find GUI directory at: {gui_dir}")
    try:
        subprocess.run(["reflex", "run"], cwd=str(gui_dir), check=True)
        return 0
    except FileNotFoundError as ex:
        raise RuntimeError("`reflex` is not installed or not on PATH.") from ex

def cli_ui() -> int:
    """Dedicated UI entry point for pyproject console script."""
    return run_ui()

def main(file_path, 
         save_cleaned: bool = False,
         verbose: bool = False,
         llm_client_name_recommender: str | None = None,
         llm_client_name_coding: str | None = None,
         llm_client_name_validation: str | None = None):
    try:
        # Backwards compatible default: if no per-agent selection is provided,
        # use the global/default LLM_CLIENT_NAME for all agents.
        recommender_name = llm_client_name_recommender or LLM_CLIENT_NAME
        coding_name = llm_client_name_coding or LLM_CLIENT_NAME
        validation_name = llm_client_name_validation or LLM_CLIENT_NAME

        llm_configs = {
            "recommender": setup_llm(recommender_name, LLM_CLIENT_MAP),
            "coding": setup_llm(coding_name, LLM_CLIENT_MAP),
            "validation": setup_llm(validation_name, LLM_CLIENT_MAP),
        }
    except (ValueError, EnvironmentError) as e:
        print(f"Configuration Error: {e}")
        return
    pipeline = Pipeline(
        llm_config=llm_configs["coding"],
        agent_llm_configs=llm_configs,
        verbose=verbose,
    )
    pipeline.run(file_path=file_path, save_cleaned=save_cleaned)

def setup_llm(llm_client_name: str, llm_clients: dict[str, LLMSpec]) -> LLMSpec:
    load_dotenv()
    if llm_client_name not in llm_clients:
        raise ValueError(
            f"Invalid LLM client '{llm_client_name}'. "
            f"Available options: {list(llm_clients.keys())} \n"
            f"Or view README to add LLM API."
        )
    llm_config = llm_clients[llm_client_name]
    api_key_name = llm_config.api_key_name
    if not api_key_name:
        raise ValueError(
            f"api_key_name not defined in registry for {llm_client_name}"
            f"First add it before running the pipeline."
        )
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise EnvironmentError(
            f"Missing API Key: {api_key_name} not found in .env file. "
            f"First add it before running the pipeline."
        )
    return llm_config

def cli(argv=None) -> int:
    """CLI wrapper. Returns a process exit code."""
    argv = list(argv or [])
    if len(argv) == 0:
        import sys
        argv = sys.argv[1:]

    if argv and argv[0].lower() == "ui":
        return run_ui()

    parser = argparse.ArgumentParser(
        prog="madclean",
        description="Run the data cleaning pipeline"
    )
    # 1. Enter filepath
    parser.add_argument(
        "file_path",
        help="Path to the input dataset file (e.g., data/benchmark_datasets/beers_dirty.csv)."
    )
    # 2. Enable verbose
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (true/false)."
    )
    # 3. Save cleaned dataset
    parser.add_argument(
        "--save-cleaned",
        action="store_true",
        help="Save the cleaned dataset output."
    )
    # Optional per-agent LLM selection (falls back to LLM_CLIENT_NAME if omitted).
    parser.add_argument(
        "--llm-recommender",
        default=None,
        help=f"LLM client key for the recommender agent. Options: {list(LLM_CLIENT_MAP.keys())}",
    )
    parser.add_argument(
        "--llm-coding",
        default=None,
        help=f"LLM client key for the coding agent. Options: {list(LLM_CLIENT_MAP.keys())}",
    )
    parser.add_argument(
        "--llm-validation",
        default=None,
        help=f"LLM client key for the validation agent. Options: {list(LLM_CLIENT_MAP.keys())}",
    )
    args = parser.parse_args(argv)
    # 4. Run main function
    main(
        file_path=args.file_path,
        save_cleaned=args.save_cleaned,
        verbose=args.verbose,
        llm_client_name_recommender=args.llm_recommender,
        llm_client_name_coding=args.llm_coding,
        llm_client_name_validation=args.llm_validation,
    )
    return 0

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset = "beers"
    file_path = BASE_DIR / "data" / "benchmark_datasets" / f"{dataset}_dirty.csv"
    main(file_path=file_path, save_cleaned=True, verbose=True)

    # raise SystemExit(cli())

    # pip install -e .
    # madclean data\benchmark_datasets\beers_dirty.csv --verbose --save-cleaned

    # dc_env -> cd code/gui -> reflex run --> http://localhost:3000