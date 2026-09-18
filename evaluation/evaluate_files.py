import time 
import json
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
from dataclasses import asdict
from typing import Callable
from pathlib import Path
import pandas as pd
# local imports
from madclean.llm.llm_registry import LLM_CLIENT_MAP
from madclean.llm.llm_settings import LLM_CLIENT_NAME
from madclean.pipeline import Pipeline
from madclean.utils.console import configure_console
from .evaluation import CleaningEvaluation

class EvaluationPipeline:
    """Evaluation pipeline for when all cleaned datasets are already available as stored files."""
    # API_DELAY_SECONDS = 60

    def __init__(self, 
                 datasets: dict[str, dict[str, str | Path]],
                 baselines: list[str],
                 replication_count: int = 5,
                 verbose: bool = True):
        self.datasets = datasets
        self.baselines = baselines
        self.replication_count = replication_count
        self.verbose = verbose
        self.all_methods = ['my_framework'] + self.baselines
        self.results = {
            method: {
                dataset: [] for dataset in self.datasets.keys()
            } for method in self.all_methods
        }
        
    def run_framework(self, dataset, evaluator: CleaningEvaluation):
        """Runs my custom data cleaning framework."""
        base_dir = Path(__file__).resolve().parent # evaluation dir
        file_path = base_dir / "results" / dataset / 'avg_eval_results.json'
        with open(file_path, 'r') as f:
            results = json.load(f)
        return results

    def evaluate_baselines(self, baseline: str, dataset: str, evaluator: CleaningEvaluation):
        base_dir = Path(__file__).resolve().parent # evaluation folder
        method_dir = base_dir / "baselines" / baseline / "data"
        # 1. Try to load runtime
        runtime = -1.0
        runtime_path = method_dir / "runtimes.json"
        if runtime_path.exists():
            try:
                with open(runtime_path, 'r') as f:
                    runtimes = json.load(f)
                    runtime = runtimes.get(dataset, -1.0)
            except Exception:
                pass
        # 2. Special case saged: only evaluate error detections
        if baseline == 'saged':
            file_path = method_dir / f"{dataset}_detection.csv"
            if not file_path.exists():
                print(f"Warning: {baseline} has no dataset '{dataset}'")
                return self._get_empty_metrics(error_msg="Cleaned file does not exist")
            try:
                pred_error_mask = pd.read_csv(file_path, encoding="utf-8").astype(bool)
                # Compare predicted mask vs ground truth mask
                detection_metrics = self._compute_detection_only(evaluator._errors_mask, pred_error_mask)
                return {**detection_metrics, 'runtime_seconds': runtime}
            except Exception as e:
                print(f"Error evaluating SAGED: {e}")
                return self._get_empty_metrics(error_msg="Error during evaluation")
        # 3. Standard baselines
        else:
            file_path = method_dir / f"{dataset}_cleaned.csv"
            if not file_path.exists():
                print(f"Warning: {baseline} has no dataset '{dataset}'")
                return self._get_empty_metrics(error_msg="Cleaned file does not exist")
            try:
                cleaned_df = pd.read_csv(file_path, encoding="utf-8")
                eval_results, _, _ = evaluator.evaluate(cleaned_df)
                return {**asdict(eval_results), 'runtime_seconds': runtime}
            except Exception as e:
                print(f"Error evaluating {baseline}: {e}")
                return self._get_empty_metrics(error_msg='Error during evaluation')

    @staticmethod
    def _compute_detection_only(gt_mask: pd.DataFrame, cleaned_mask: pd.DataFrame):
        """For SAGED baseline which only performs error detection."""
        gt_flat = gt_mask.to_numpy().flatten()
        cleaned_flat = cleaned_mask.to_numpy().flatten()
        total_size = gt_flat.size
        tp = (gt_flat & cleaned_flat).sum()
        fp = (~gt_flat & cleaned_flat).sum()
        fn = (gt_flat & ~cleaned_flat).sum()
        tn = total_size - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "detection_counts": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "total_errors": int(gt_flat.sum()),
                "total_changes": int(cleaned_flat.sum())
            },
            "detection_metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            },
            "correction_metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score":  0.0
            }
        }
            
    @staticmethod
    def _get_empty_metrics(error_msg=""):
        """Default metrics for when baseline fails."""
        return {
            "detection_metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "correction_metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "runtime_seconds": 0.0,
            "error": error_msg
        }

    def run(self, store_results: bool = True) -> dict:
        """Executes entire evaluation pipeline."""
        if self.verbose: print(f"Running evaluation pipeline with {len(self.datasets)} datasets and {len(self.baselines)} baselines")
        # 1. Setup evaluators for each dataset
        evaluators = {}
        for dataset, config in self.datasets.items():
            evaluators[dataset] = CleaningEvaluation(
                config['dirty_path'], 
                config['ground_truth_path'], 
                numeric_cols=config['numeric_cols']
            )
        # 2. Run and evaluate framework
        for dataset, config in self.datasets.items():
            if self.verbose: print(f"\nProcessing dataset: '{dataset}'")
            result = self.run_framework(dataset, evaluators[dataset])
            self.results['my_framework'][dataset].append(result)
        # 3. Evaluate baselines
        if self.baselines:
            for baseline in self.baselines:
                for dataset in self.datasets.keys():
                    result = self.evaluate_baselines(baseline, dataset, evaluators[dataset])
                    self.results[baseline][dataset].append(result)
        # 4. Save results
        if store_results:
            self._save_results()
        if self.verbose: print("\nEvaluation pipeline finished succesfully.")
        return self.results
    
    def _save_results(self):
        base_dir = Path(__file__).resolve().parent # evaluation folder
        results_file_path = base_dir / "results" /"results.json"
        with open(results_file_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        if self.verbose: print(f"Saved evaluation results to {results_file_path}")

if __name__ == "__main__":
    configure_console()
    BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_datasets"
    datasets_to_evaluate = {
        "hospital": {
            "dirty_path": BASE_DIR / "hospital_dirty.csv",
            "ground_truth_path": BASE_DIR / "hospital_gt.csv",
            "numeric_cols": {"ProviderNumber", "ZipCode", "PhoneNumber", "Score", "Sample"}
        },
        "beers": {
            "dirty_path": BASE_DIR / "beers_dirty.csv",
            "ground_truth_path": BASE_DIR / "beers_gt.csv",
            "numeric_cols": {"id", "ounces", "abv", "ibu", "brewery_id"}
        },
        "movies": {
            "dirty_path": BASE_DIR / "movies_dirty.csv",
            "ground_truth_path": BASE_DIR / "movies_gt.csv",
            "numeric_cols": {"Year", "Duration", "RatingValue", "RatingCount"}
        },
        "rayyan": {
            "dirty_path": BASE_DIR / "rayyan_dirty.csv",
            "ground_truth_path": BASE_DIR / "rayyan_gt.csv",
            "numeric_cols": {"id", "article_jvolumn", "article_jissue"}
        }
    }
    
    baselines_to_run = ["raha_baran", "holoclean", "retclean", "cocoon", "saged"]

    evaluation_pipeline = EvaluationPipeline(
        datasets=datasets_to_evaluate,
        baselines=baselines_to_run,
        replication_count=5
    )

    evaluation_results = evaluation_pipeline.run(store_results=True)

    print("Evaluation results:\n")
    pprint(evaluation_results)

    # python -m evaluation.evaluate_files
