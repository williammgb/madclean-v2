import pandas as pd
import itertools
from collections import defaultdict
from typing import cast
# Local imports
from macs.components.domain.schema import MultiColumnTask, FDResult

class FunctionalDependencies:
    """
    Analyse DataFrame to detect functional dependencies, identify violations and find missing values that can be imputed.
    This class implements the MultiColumnCleaner protocol.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @property
    def task_type(self) -> str:
        return "FD"

    def detect(self, df: pd.DataFrame) -> list[MultiColumnTask]:
        """Runs a simple analysis to find FDs that have issues (violations or imputable missing values)."""
        fds = self._detect_final_fds(df)
        if not fds:
            return []
        
        fds_with_issues = []
        for fd in fds:
            lhs, rhs = fd['lhs'], fd['rhs']
            # 1. Detect violations (multiple RHS values for each unique LHS value)
            grouped_lhs = df.dropna(subset=[lhs,rhs]).groupby(lhs)
            rhs_count_per_lhs = grouped_lhs[rhs].nunique()
            violations_count = int((rhs_count_per_lhs > 1).sum())
            # 2. Detect imputables (null RHS values that can be imputed using non-null RHS value for same LHS value)
            lhs_with_non_null_rhs = set(grouped_lhs.groups.keys())
            is_null_rhs = df[rhs].isnull()
            is_lhs_imputable = df[lhs].isin(lhs_with_non_null_rhs)
            imputables_mask = is_null_rhs & is_lhs_imputable
            imputables_count = int(imputables_mask.sum())
            # 3. Only return FD if it needs to be enforced, otherwise ignore
            if violations_count > 0 or imputables_count > 0:
                fd_obj = FDResult(
                    lhs=lhs,
                    rhs=rhs,
                    score=fd['score'],
                    violations_count=violations_count,
                    imputables_count=imputables_count
                )
                fd_key = f"{lhs} → {rhs}"
                fds_with_issues.append(MultiColumnTask(
                    task_type=self.task_type,
                    target_columns=[lhs, rhs],
                    verbose_key=fd_key,
                    data=fd_obj
                ))
                # if self.verbose: print(f"[{fd_key}] {violations_count} VIOLATIONS & {imputables_count} IMPUTABLES. ")
        return fds_with_issues

    def get_data(self, df: pd.DataFrame, task_info: MultiColumnTask) -> MultiColumnTask | None:
        """Runs full analysis on FD to get data needed for enforcement. Finds violations and context rows, and imputation options."""
        def truncate(val):
            """Helper function to truncate cells with long string values to reduce token usage and minimize noise."""
            if isinstance(val, str) and len(val) > 50:
                return val[:50] + "..."
            return val
        fd = cast(FDResult, task_info.data) # For safe attribute access
        lhs, rhs = fd.lhs, fd.rhs
        # 1. Find violations
        grouped_lhs = df.dropna(subset=[lhs,rhs]).groupby(lhs)
        violations = []
        for lhs_val, group_df in grouped_lhs:
            # 2. Get value counts for the RHS values in this group
            rhs_counts = group_df[rhs].value_counts()
            if len(rhs_counts) > 1:
                conflicting_rhs_with_counts = list(rhs_counts.items()) # e.g., [('NY', 10), ('New York', 2)]
                conflicting_rhs_values = list(rhs_counts.index) # e.g., ['NY', 'New York']
                # 3. Gather context rows to provide to the LLM
                context_rows = []
                for rhs_val in conflicting_rhs_values:
                    sample_row = df[
                        (df[lhs] == lhs_val) & (df[rhs] == rhs_val)
                        ].iloc[0]
                    sample_row = sample_row.apply(truncate)
                    context_rows.append(sample_row.tolist())
                violations.append({
                    'lhs': lhs_val,
                    'rhs_conflicts': conflicting_rhs_with_counts,
                    'context': context_rows
                })
        violation_data = {
            'count': len(violations),
            'violations': violations}        
        # 4. Find imputable missing values
        lhs_with_non_null_rhs = set(grouped_lhs.groups.keys())
        is_null_rhs = df[rhs].isnull()
        is_lhs_imputable = df[lhs].isin(lhs_with_non_null_rhs)
        imputation_mask = is_null_rhs & is_lhs_imputable
        imputation_count = int(imputation_mask.sum())
        imputation_data = {'count': imputation_count}
        # 5. Only return data if FD needs to be enforced. If it has no violations/imputables, return None
        if (violation_data['count'] > 0 or imputation_data['count'] > 0):
            fd.violation_count = violation_data['count'] 
            fd.imputable_count = imputation_data['count']
            fd.violation_data = violation_data
            fd.imputation_data = imputation_data
            # if self.verbose: print(f"[{task_info.verbose_key}] {violation_data['count']} VIOLATIONS & {imputation_data['count']} IMPUTABLES. ")
            task_info.data = fd
            return task_info
        return None

    # def get_fd_counts(self, df: pd.DataFrame, lhs: str, rhs: str) -> tuple[int, int]: 
    #     """For FD validator agent"""
    #     grouped_lhs = df.dropna(subset=[lhs, rhs]).groupby(lhs, sort=False)
    #     violation_count = sum(
    #         (group[rhs].nunique() > 1)
    #         for _, group in grouped_lhs)
    #     lhs_with_non_null_rhs = set(df.dropna(subset=[rhs])[lhs].unique())
    #     is_null_rhs = df[rhs].isnull()
    #     is_lhs_imputable = df[lhs].isin(lhs_with_non_null_rhs)
    #     imputation_mask = is_null_rhs & is_lhs_imputable
    #     imputable_count = int(imputation_mask.sum())
    #     return violation_count, imputable_count

    ################################# DETECTION #################################
    def _detect_final_fds(self, df: pd.DataFrame) -> list[dict] | None:
        """Run both detection methods and return dictionary of results."""
        # 1. Detect all FDs available in the DataFrame
        all_fds = self._detect_all_fds(df)
        if not all_fds:
            if self.verbose: print('No functional dependencies found.')
            return None
        # 2. Merge FDs such that each column occurs at most one time as RHS
        final_fds = self._merge_fds(df, all_fds)
        # if self.verbose:
        #     for fd in final_fds:
        #         print(f"LHS: {fd['lhs']}, RHS: {fd['rhs']}, Score: {fd['score']:.4f}") 
        return final_fds
    
    def _detect_all_fds(self,
                    df: pd.DataFrame,
                    score_threshold: float = 0.925) -> list[dict]:
        """Detects FDs by checking all candidates and computing score."""
        fds = []
        # 1. Filter candidate columns based on redundancy (significantly reduces number of permutations)
        candidate_lhs_cols = set(self._get_candidate_columns(df))
        # 2. For all possible combinations (1:1 mappings), compute FD score
        valid_rhs_cols = {col for col in df.columns if df[col].nunique() > 1}
        all_cols = df.columns.tolist() 
        for lhs, rhs in itertools.permutations(all_cols, 2):
            if lhs not in candidate_lhs_cols or rhs not in valid_rhs_cols:
                continue
            subset_df = df[[lhs, rhs]].dropna()
            fd_score = self._calculate_score_FAST(lhs, rhs, subset_df)
            # 3. If FD score > threshold, this FD is valid
            if fd_score >= score_threshold:
                fds.append({'lhs': lhs, 'rhs': rhs, 'score': round(fd_score, 4)})
        fds = sorted(fds, key=lambda x: x['score'], reverse=True)
        return fds

    def _merge_fds(self, df: pd.DataFrame, fds: list[dict]) -> list[dict]:
        """Prune list of FDs by selecting the best LHS for each RHS based on purity score (and uniqueness if its a tie)."""
        # 1. Compute uniqueness for each column, as tie-breaker when FD score is equal. More unique is more informative
        uniqueness_map = {
            col: df[col].nunique(dropna=False) / len(df) for col in df.columns
        }
        # 2. Group FDs by their RHS column. For each RHS, pick the best FD
        rhs_groups = defaultdict(list)
        for fd in fds:
            rhs_groups[fd['rhs']].append(fd)
        merged_fds = []
        for _, fd_group in rhs_groups.items():
            if len(fd_group) == 1:
                merged_fds.append(fd_group[0])
                continue
            best_fd = max(
                fd_group,
                key=lambda fd: (fd['score'], uniqueness_map.get(fd['lhs'], 0))
            )
            merged_fds.append(best_fd)
        # 3. Sort FDs by score
        merged_fds = sorted(merged_fds, key=lambda fd: fd['score'], reverse=True)
        # if self.verbose: print(f"Pruning FDs: {len(fds)} reduced to {len(merged_fds)}.")
        return merged_fds

    def _get_candidate_columns(self, 
                               df: pd.DataFrame,
                               min_frac_uniqueness: float = 0.7,
                               min_frac_duplicates: float = 0.8
                               ) -> list[str]:
        """
        Pre-filters candidate columns based on their fraction of unique values.
        Only columns with a high level of duplication (many repeated values) are considered 
        candidates, since redundant columns are more useful for FD enforcement.
        """
        candidate_cols = []
        for col in df.columns:
            series = df[col].dropna()
            if series.empty or series.nunique() <= 1:
                continue
            # 1. Compute fraction of cells that are repeating uniques
            value_counts = series.value_counts()
            num_unique_values = len(value_counts)
            num_uniques_that_repeat = (value_counts > 1).sum()
            fraction_repeating_uniques = num_uniques_that_repeat / num_unique_values
            # 2. Compute fraction of cells that are duplicates
            num_rows_that_are_duplicates = value_counts[value_counts > 1].sum()
            fraction_duplicates = num_rows_that_are_duplicates / len(series)
            # 3. If either of the fractions exceed threshold, add to candidate columns
            if (fraction_repeating_uniques >= min_frac_uniqueness or
                fraction_duplicates >= min_frac_duplicates):
                candidate_cols.append(col)
        return candidate_cols

    def _calculate_score_FAST(self, lhs: str, rhs: str, subset_df: pd.DataFrame) -> float:
        """
        Calculate purity score. Using vectorized pandas operations to significanly increase speed.
        score = sum(count of most frequent RHS value of each unique LHS value) / total number of non-null rows
        """
        if subset_df.empty:
            return 0.0
        pair_counts = subset_df.value_counts([lhs, rhs])
        max_rhs_counts_per_lhs = pair_counts.groupby(level=lhs).max()
        fd_score = max_rhs_counts_per_lhs.sum() / len(subset_df)
        return fd_score

####### TEST CODE #######
if __name__ == "__main__":
    import time
    from pathlib import Path
    from macs.utils.helpers import load_dataset
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    file_path = BASE_DIR / "data" / "benchmark_datasets" / "beers_dirty.csv"
    df = load_dataset(file_path)
    beers_fds = [
        {'lhs': 'brewery_id', 'rhs': 'brewery_name', 'score': 1.0000},    
        {'lhs': 'brewery_id', 'rhs': 'state', 'score': 1.0000},
        {'lhs': 'brewery_id', 'rhs': 'city', 'score': 0.9502},
        {'lhs': 'brewery_name', 'rhs': 'brewery_id', 'score': 0.9896}
        ]
    hospital_fds = [
        {'lhs': 'Address1' , 'rhs': 'HospitalName' , 'score': 0.9770}, 
        {'lhs': 'PhoneNumber' , 'rhs': 'State', 'score': 0.9760 },
        {'lhs': 'PhoneNumber' , 'rhs': 'HospitalOwner', 'score': 0.9750  },
        {'lhs': 'Address1' , 'rhs': 'EmergencyService', 'score': 0.9740  },
        {'lhs': 'Address1' , 'rhs': 'ProviderNumber', 'score': 0.9720  },
        {'lhs': 'Stateavg' , 'rhs': 'MeasureCode', 'score': 0.9720  },
        {'lhs': 'PhoneNumber' , 'rhs': 'City', 'score': 0.9690  }, 
        {'lhs': 'Stateavg' , 'rhs': 'Condition', 'score': 0.9690  },   
        {'lhs': 'Stateavg' , 'rhs': 'MeasureName', 'score': 0.9660  },  
        {'lhs': 'PhoneNumber' , 'rhs': 'Address1', 'score': 0.9700 }, 
        {'lhs': 'City' , 'rhs': 'CountyName', 'score': 0.9640  },
        {'lhs': 'MeasureName' , 'rhs': 'Stateavg', 'score': 0.9550  }, 
        {'lhs': 'HospitalName' , 'rhs': 'ZipCode', 'score': 0.9710  },
        {'lhs': 'Address1' , 'rhs': 'PhoneNumber', 'score': 0.9680 }
    ]
    start_time = time.perf_counter()
    fd_manager = FunctionalDependencies(verbose=True)
    fds = fd_manager.detect(df)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Runtime: {runtime}")
    for fd in fds:
        fd_data = fd_manager.get_data(df, fd)

    # python -m macs.components.dataprofiler.functional_dependencies
