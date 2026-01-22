import os
import argparse
from dotenv import load_dotenv
### Local imports
from macs.pipeline import Pipeline
from macs.llm.llm_settings import LLM_CLIENT_NAME
from macs.llm.llm_registry import LLM_CLIENT_MAP

def main(file_path, 
         save_cleaned: bool = False,
         verbose: bool = False):
    try:
        llm_config = setup_llm(LLM_CLIENT_NAME, LLM_CLIENT_MAP)
    except (ValueError, EnvironmentError) as e:
        print(f"Configuration Error: {e}")
        return
    pipeline = Pipeline(llm_client=llm_config, verbose=verbose)
    pipeline.run(file_path=file_path, save_cleaned=save_cleaned)

def setup_llm(llm_client_name: str, llm_clients: dict):
    load_dotenv()
    if llm_client_name not in llm_clients:
        raise ValueError(
            f"Invalid LLM client '{llm_client_name}'. "
            f"Available options: {list(llm_clients.keys())} \n"
            f"Or view README to add LLM API."
        )
    llm_client = llm_clients[llm_client_name]
    api_key_name = llm_client["api_key_name"]
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise EnvironmentError(
            f"Missing API Key: {api_key_name} not found in .env file. "
            f"First add it before running the pipeline."
        )
    return llm_client

def cli(argv=None) -> int:
    """CLI wrapper. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m macs.main",
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
    args = parser.parse_args(argv)
    # 4. Run main function
    main(
        file_path=args.file_path,
        save_cleaned=args.save_cleaned,
        verbose=args.verbose
    )
    return 0

if __name__ == "__main__":
    # from pathlib import Path
    # BASE_DIR = Path(__file__).resolve().parent.parent
    # dataset = "beers"
    # file_path = BASE_DIR / "data" / "benchmark_datasets" / f"{dataset}_dirty.csv"
    # main(file_path=file_path, save_cleaned=True, verbose=True)

    raise SystemExit(cli())

