import pandas as pd
import numpy as np
from dataclasses import asdict
from pathlib import Path
# local imports
from madclean.utils.helpers import load_dataset
from madclean.evaluation.scores import (
    CorrectionCounts,
    DetectionCounts,
    PrecisionRecallF1,
    Scores,
)

class CleaningEvaluation:
    def __init__(self, 
                 dirty_path: str | Path, ground_truth_path: str | Path, 
                 numeric_cols: set,
                 strict_numeric_types: bool = False):
        self.dirty_df = load_dataset(dirty_path)
        self.ground_truth_df = load_dataset(ground_truth_path)
        self.numeric_cols = numeric_cols
        self._validate_shapes()
        # Boolean df with error cells (True where dirty != ground_truth)
        self._errors_mask = ~self._compare_dataframes(self.dirty_df, self.ground_truth_df) 

    def _validate_shapes(self):
        if self.dirty_df.shape != self.ground_truth_df.shape:
            raise ValueError("Dirty and ground truth DataFrames must have the same shape.")
        if not self.dirty_df.index.equals(self.ground_truth_df.index) or \
            not self.dirty_df.columns.equals(self.ground_truth_df.columns):
            raise ValueError("Dirty and ground truth DataFrames must have the same index and columns.")
        
    def evaluate(self, cleaned_df: pd.DataFrame) -> tuple[Scores, dict[str, Scores], dict]:
        if self.dirty_df.shape != cleaned_df.shape:
            raise ValueError("Cleaned DataFrame must have the same shape as the initial dirty DataFrame.")
        # Boolean df that shows which changes the cleaning framework has made
        changes_mask = ~self._compare_dataframes(self.dirty_df, cleaned_df)
        overall_results, column_results = self._get_performance_metrics(cleaned_df, changes_mask)
        reports = self._generate_detailed_reports(cleaned_df, changes_mask)
        return overall_results, column_results, reports
   
    def _get_performance_metrics(self, 
                                cleaned_df: pd.DataFrame, 
                                changes_mask: pd.DataFrame) -> tuple[Scores, dict[str, Scores]]:
        """Calculate metrics for evaluation of error detection and correction."""
        # 1. Create mask that compares cleaned dataset with ground truth
        equal_mask = self._compare_dataframes(cleaned_df, self.ground_truth_df)
        column_metrics = {}
        # 2. Score the whole table
        overall_metrics = self._compute_metrics(
            errors_mask=self._errors_mask,
            changes_mask=changes_mask,
            equal_mask=equal_mask,
            total_size=self.dirty_df.size
        )
        # 3. Compute column-level results
        for col in self.dirty_df.columns:
            column_metrics[col] = self._compute_metrics(
                errors_mask=self._errors_mask[col], 
                changes_mask=changes_mask[col], 
                equal_mask=equal_mask[col],
                total_size=self.dirty_df[col].size
            )
        return overall_metrics, column_metrics

    def _compute_metrics(self,
                        errors_mask: pd.DataFrame | pd.Series,
                        changes_mask: pd.DataFrame | pd.Series,
                        equal_mask: pd.DataFrame | pd.Series,
                        total_size: int) -> Scores:
        # 1. Convert masks to numpy for better operations
        errors_mask = errors_mask.to_numpy(dtype=bool)
        changes_mask = changes_mask.to_numpy(dtype=bool)
        equal_mask = equal_mask.to_numpy(dtype=bool)
        # 2. Compute detection results
        tp = np.sum(errors_mask & changes_mask)
        fp = np.sum(changes_mask & ~errors_mask)
        fn = np.sum(errors_mask & ~changes_mask)
        tn = total_size - (tp + fp + fn)
        # 3. Compute detection metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        # 4. Compute correction results
        tp_correct = np.sum(equal_mask & errors_mask & changes_mask)
        total_changes = np.sum(changes_mask)
        total_errors = np.sum(errors_mask)
        # 5. Compute correction metrics
        repair_precision = tp_correct / total_changes if total_changes > 0 else 0.0
        repair_recall = tp_correct / total_errors if total_errors > 0 else 0.0
        repair_f1 = (2 * (repair_precision * repair_recall) / (repair_precision + repair_recall)
                        if (repair_precision + repair_recall) > 0 else 0.0)
        
        return Scores(
            detection_counts=DetectionCounts(
                true_positives=int(tp),
                false_positives=int(fp),
                false_negatives=int(fn),
                true_negatives=int(tn),
                total_errors=int(total_errors),
                total_changes=int(total_changes),
            ),
            detection_metrics=PrecisionRecallF1(
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
            ),
            correction_counts=CorrectionCounts(
                correctly_repaired_cells=int(tp_correct),
                incorrectly_repaired_cells=int(tp - tp_correct),
                repaired_clean_cells=int(fp),
            ),
            correction_metrics=PrecisionRecallF1(
                precision=float(repair_precision),
                recall=float(repair_recall),
                f1_score=float(repair_f1),
            ),
        )
                

    def _compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Compares 2 DataFrames, uses safe numeric comparision for numeric columns 
        (to prevent dtype comparision issues), and standard comparision for other columns.
        Handles NAs by only comparing values where neither are NA.
        """
        result_df = pd.DataFrame(False, index=df1.index, columns=df1.columns, dtype=bool) # pure booleans, will never contain NA
        for col in df1.columns:
            arr1 = df1[col].to_numpy() # convert to numpy, which prevents 'boolean value of NA is ambiguous' errors
            arr2 = df2[col].to_numpy()
            arr1_na_mask = pd.isna(arr1)
            arr2_na_mask = pd.isna(arr2)
            # Compare datasets, handle NA values and issues for numeric columns
            both_na = arr1_na_mask & arr2_na_mask
            both_not_na = ~arr1_na_mask & ~arr2_na_mask
            equal_mask = np.zeros(len(arr1), dtype=bool)
            if both_not_na.any():
                v1 = arr1[both_not_na]
                v2 = arr2[both_not_na]
                if col in self.numeric_cols:
                    try:
                        v1_num = pd.to_numeric(v1, errors='coerce')
                        v2_num = pd.to_numeric(v2, errors='coerce')
                        valid_nums = ~pd.isna(v1_num) & ~pd.isna(v2_num)
                        if valid_nums.any():
                            subset_equal_mask = np.zeros(len(v1), dtype=bool) 
                            subset_equal_mask[valid_nums] = np.isclose(v1_num[valid_nums], v2_num[valid_nums], rtol=1e-9)
                            equal_mask[both_not_na] = equal_mask[both_not_na] | subset_equal_mask
                        # Compare values that cannot be converted to numeric     
                        invalid_nums = ~valid_nums 
                        if invalid_nums.any():
                            subset_equal_mask2 = np.zeros(len(v1), dtype=bool)
                            subset_equal_mask2[invalid_nums] = (v1[invalid_nums] == v2[invalid_nums])
                            equal_mask[both_not_na] = equal_mask[both_not_na] | subset_equal_mask2

                    except Exception:
                        equal_mask[both_not_na] = (v1 == v2)
                else:
                    try:
                        equal_mask[both_not_na] = (v1 == v2)
                    except ValueError: 
                        # Handle lists, arrays, dicts, set -> manual iteration
                        indices = np.where(both_not_na)[0]
                        complex_types = (list, np.ndarray, dict, set)
                        for idx in indices:
                            val1, val2 = arr1[idx], arr2[idx]
                            if isinstance(val1, complex_types) or isinstance(val2, complex_types):
                                # Force string conversion for complex types
                                equal_mask[idx] = (str(val1) == str(val2))
                            else:
                                # Standard comparison for others
                                equal_mask[idx] = (val1 == val2)

            result_df[col] = both_na | equal_mask # values with NA on only one side remain False
        return result_df

    def _generate_detailed_reports(self,
                                   cleaned_df: pd.DataFrame, 
                                   changes_mask: pd.DataFrame, 
                                   max_rows: int = 1500) -> dict:
        """Generate detailed report per column of changes and errors."""
        reports = {}
        for col in self.dirty_df.columns:
            mask = self._errors_mask[col] | changes_mask[col]

            if mask.any():
                report_df = pd.DataFrame({
                    "dirty": self.dirty_df.loc[mask, col],
                    "cleaned": cleaned_df.loc[mask, col],
                    "ground_truth": self.ground_truth_df.loc[mask, col],
                })
                statuses = []
                is_numeric = col in self.numeric_cols
                for _, row in report_df.iterrows():
                    d, c, gt = row['dirty'], row['cleaned'], row['ground_truth']

                    d_eq_gt = self._safe_eq(d, gt, is_numeric)
                    c_eq_gt = self._safe_eq(c, gt, is_numeric)
                    d_eq_c = self._safe_eq(d, c, is_numeric)

                    if not d_eq_gt and c_eq_gt:
                        statuses.append("Corrected (TP)")
                    elif d_eq_gt and not c_eq_gt:
                        statuses.append("Wrong correction (FP)")
                    elif not d_eq_gt and not c_eq_gt and not d_eq_c:
                        statuses.append("Detected but wrong fix")
                    elif not d_eq_gt and d_eq_c:
                        statuses.append("Not fixed (FN)")
                    else:
                        statuses.append("(TN)")
                
                report_df["status"] = statuses
                reports[col] = report_df.head(max_rows)
        return reports
    
    @staticmethod
    def _safe_eq(x, y, is_numeric: bool = False) -> bool:
        """Compares 2 values, handles complex types like lists, dicts, sets, arrays by converting to string first."""
        complex_types = (list, np.ndarray, dict, set)
        if isinstance(x, complex_types) or isinstance(y, complex_types):
            return str(x) == str(y)

        x_na = pd.isna(x)
        y_na = pd.isna(y)
        if x_na and y_na: return True
        if x_na or y_na: return False

        if is_numeric:
            try:
                return np.isclose(float(x), float(y), rtol=1e-9)
            except (ValueError, TypeError):
                pass
        return x == y

    def display_results(self, results: Scores):
        print('\n')
        for section, values in asdict(results).items():
            print(f"--- {section.replace('_', ' ').upper()} ---")
            for k, v in values.items():
                if isinstance(v, float):
                    print(f"{k.replace('_', ' ').title():<30} {v:.3f}")
                else:
                    print(f"{k.replace('_', ' ').title():<30} {v}")

    def display_reports(self, reports: dict):
        if not reports:
            print("No errors or changes in all DataFrames.")
        else:
            print("\n-- DETAILED REPORT --")
            for col, rep in reports.items():
                print(f"'Column: {col}'")
                print(rep.to_string())
                print('-'*50)


if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent # framework dir
    dataset = 'hospital'
    dirty_path = BASE_DIR / "data" / "benchmark_datasets" / f"{dataset}_dirty.csv"
    gt_path = BASE_DIR / "data" / "benchmark_datasets" / f"{dataset}_gt.csv"
    cleaned_path = BASE_DIR / "data" / "cleaned" / f"{dataset}_cleaned.csv"
    numeric_cols = {"ProviderNumber", "ZipCode", "PhoneNumber", "Score", "Sample"}
    evaluator = CleaningEvaluation(dirty_path, gt_path, numeric_cols)
    cleaned_df = load_dataset(cleaned_path)
    eval_results, _, eval_reports = evaluator.evaluate(cleaned_df) 
    print(eval_results)
    # print()
    # evaluator.display_reports(eval_reports)