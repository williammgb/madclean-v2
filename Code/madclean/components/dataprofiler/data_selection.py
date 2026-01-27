import re
import pandas as pd
import numpy as np
from typing import Callable
from dateutil.parser import parse
# Local imports
from madclean.utils.helpers import format_list_for_prompt

class DataSampler:
    """
    Class to sample data in more intelligent manner to improve LLM cleaning performance 
    by providing more informative dataset sample to the LLM.
    """
    SAMPLE_SIZES = {
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
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def sample_column(self, column: pd.Series, col: str, column_type: str) -> str:
        """Creates an intelligent string sample of a column based on its semantic type."""       
        column = column.dropna()
        if column.empty or column_type == "EMPTY":
            return f'Column "{col}" sample → [EMPTY]'
        TYPE_MAP: dict[str, Callable[[pd.Series], tuple[list, list]]] = {
            'INTEGER': ("NUMERIC", self._sample_column_numeric),
            'FLOAT': ("NUMERIC", self._sample_column_numeric),
            'DATETIME': ("DATETIME", self._sample_column_datetime),
            'BOOLEAN': ("STRING", self._sample_column_boolean),
            'DIRTY_INTEGER': ("DIRTY_NUMERIC", self._sample_column_dirty_numeric),
            'DIRTY_FLOAT': ("DIRTY_NUMERIC", self._sample_column_dirty_numeric),
            'NAMED_ENTITY': ("STRING", self._sample_column_string),
            'DISCRETE_STRING': ("STRING", self._sample_column_string),
            'NATURAL_LANGUAGE_TEXT': ("NLT", self._sample_column_nlt),
        }
        # 1. Select sample configurations based on semantic type
        sample_sizes_cfg, sampling_func = TYPE_MAP.get(column_type) 
        sample1, sample2, full_sample = sampling_func(column, sample_sizes_cfg)
        # if self.verbose: print(f"[{col}]  Sample sizes=({len(sample1)}, {len(sample2)}) -- Full sample={full_sample}")
        # 2. Generate column sample as a string
        header = f"Column '{col}':"
        if column_type in ['INTEGER', 'FLOAT']:
            msg = "\nDirty Sample includes ALL dirty values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Clean Sample (convertible to numeric): [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Dirty Sample (could not be converted): [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}")
        elif column_type in ['DIRTY_INTEGER', 'DIRTY_FLOAT']:
            msg = "\nDirty Sample includes ALL unique dirty patterns present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Random Sample: [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Dirty Sample (unique noise pattern representatives): [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}")
        elif column_type == 'DATETIME':
            msg = "\nDirty Sample includes ALL dirty values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Clean Sample (convertible to datetime): [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Dirty Sample (could not be converted): [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}")
        elif column_type == 'NATURAL_LANGUAGE_TEXT':
            msg = "\nSample includes ALL values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Short Text Sample: [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Long Text Sample: [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}")
        else:
            msg = "\nUnique Sample includes ALL unique values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Random Sample: [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Unique Values: [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}")
        return prompt

    # ====================== type-specific sampling ========================== 
    def _sample_column_numeric(self, series: pd.Series, sample_sizes_cfg: str) -> tuple[list, list, bool]:
        """Separate numeric values from non-numeric (dirty) values for INTEGER and FLOAT columns."""
        cfg = self.SAMPLE_SIZES[sample_sizes_cfg]
        coerced_series = pd.to_numeric(series, errors='coerce')
        # 1. Gather 'clean' sample: values that can be converted to numeric
        clean_mask = coerced_series.notna()
        clean_sample = series[clean_mask].sample(n=min(cfg['clean_sample_size'], clean_mask.sum()), replace=False).tolist()
        # 2. Gather 'dirty' sample: values that cannot be converted to numeric
        dirty_mask = coerced_series.isna()
        dirty_series = series[dirty_mask] 
        if not dirty_series.empty:
            dirty_sample = dirty_series.value_counts().nlargest(cfg['dirty_sample_size']).index.tolist()
            all_dirty_included = dirty_series.nunique() <= cfg['dirty_sample_size']
        else:
            dirty_sample = []
            all_dirty_included = True
        return clean_sample, dirty_sample, all_dirty_included
    
    def _sample_column_dirty_numeric(self, series: pd.Series, sample_sizes_cfg: str) -> tuple[list, list, bool]:
        """ Provides random sample and list of all unique dirty patterns for DIRTY_FLOAT and DIRTY_INTEGER columns."""
        cfg = self.SAMPLE_SIZES[sample_sizes_cfg]
        # 1. Take random sample
        random_sample = series.sample(n=min(cfg['random_sample_size'], len(series)), replace=False).tolist()
        # 2. Sample all unique noise patterns as dirty values
        unique_sample = series.unique().tolist()
        number_pattern = re.compile(r'\d+(\.\d+)?')
        dirty_patterns = set()
        unique_dirty_values = []
        for val in unique_sample:
            s_val = str(val).strip()
            if re.fullmatch(number_pattern, s_val):
                continue
            mask = number_pattern.sub('{N}', s_val) # '$15' -> '${N}'
            if mask not in dirty_patterns:
                dirty_patterns.add(mask)
                unique_dirty_values.append(val)
        all_unique_included = len(unique_dirty_values) <= cfg['unique_sample_size']
        if not all_unique_included:
            unique_dirty_values = np.random.choice(unique_dirty_values, size=cfg['unique_sample_size'], replace=False).tolist()
        return random_sample, unique_dirty_values, all_unique_included
        
    def _sample_column_datetime(self, series: pd.Series, sample_sizes_cfg: str) -> tuple[list, list, bool]:
        """ Separates datetime values from non-datetime ("dirty") values for DATETIME columns."""
        def _robust_date_parser(value):
            """
            Function to_datetime() only accepts one dominant format, so does not work with mixed date formats.
            This function performs individual iteration, bit slower but necessary.
            """
            if pd.isna(value):
                return pd.NaT
            try:
                return (parse(str(value)))
            except (ValueError, TypeError):
                return pd.NaT
        cfg = self.SAMPLE_SIZES[sample_sizes_cfg]
        coerced_series = series.apply(_robust_date_parser)
        # 1. Gather 'clean' sample: values that can be converted to datetime 
        clean_mask = coerced_series.notna()
        clean_sample = series[clean_mask].sample(n=min(cfg['clean_sample_size'], clean_mask.sum()), replace=False).tolist()
        # 2. Gather 'dirty' sample: values that cannot be converted to datetime 
        dirty_mask = coerced_series.isna()
        dirty_sample = series[dirty_mask].unique().tolist()
        all_dirty_included = len(dirty_sample) <= cfg['dirty_sample_size']
        if not all_dirty_included:
            dirty_sample = np.random.choice(dirty_sample, size=cfg['dirty_sample_size'], replace=False).tolist() #add seed
        return clean_sample, dirty_sample, all_dirty_included

    def _sample_column_boolean(self, series: pd.Series, sample_sizes_cfg: str) -> tuple[list, list, bool]:
        """ Provides random sample and list of all unique values for BOOLEAN columns."""
        cfg = self.SAMPLE_SIZES[sample_sizes_cfg]
        # Gather random sample and uniuqe sample
        random_sample = series.sample(n=min(cfg['random_sample_size'], len(series)), replace=False).tolist()
        unique_sample = series.unique().tolist()
        all_unique_included = len(unique_sample) <= cfg['unique_sample_size']
        if not all_unique_included:
            unique_sample = np.random.choice(unique_sample, size=cfg['unique_sample_size'], replace=False).tolist()
        return random_sample, unique_sample, all_unique_included

    def _sample_column_string(self, series: pd.Series, sample_sizes_cfg: str) -> tuple[list, list, bool]:
        """ Provides random sample and list of all unique values for NAMED_ENTITY and DISCRETE_STRING columns."""
        cfg = self.SAMPLE_SIZES[sample_sizes_cfg]
        # Gather random sample and uniuqe sample
        random_sample = series.sample(n=min(cfg['random_sample_size'], len(series)), replace=False).tolist()
        unique_sample = series.unique().tolist()
        all_unique_included = len(unique_sample) <= cfg['unique_sample_size']
        if not all_unique_included:
            unique_sample = np.random.choice(unique_sample, size=cfg['unique_sample_size'], replace=False).tolist()
        return random_sample, unique_sample, all_unique_included

    def _sample_column_nlt(self, series: pd.Series, sample_sizes_cfg: str)-> tuple[list, list, bool]:
        """Samples short and long text values for NATURAL_LANGUAGE_TEXT columns."""
        cfg = self.SAMPLE_SIZES[sample_sizes_cfg]
        # 1. Split column into short_sample and long_sample
        unique_series = pd.Series(series.unique())
        if len(unique_series) < (cfg['short_sample_size'] + cfg['long_sample_size']):
             sorted_series = unique_series.sort_values(key=lambda s: s.str.len(), ignore_index=True)
             mid = len(sorted_series) // 2
             short_sample = sorted_series.iloc[:mid]
             long_sample = sorted_series.iloc[mid:]
             return short_sample.tolist(), long_sample.tolist(), True
        str_lengths = series.str.len()
        median_len = str_lengths.median()
        short_texts = series[str_lengths <= median_len]
        long_texts = series[str_lengths > median_len]
        # 2. Gather elements, mainly from short_sample and some from long_sample. To reduce token usage.
        short_sample = short_texts.sample(
            n=min(cfg['short_sample_size'], len(short_texts)), replace=False
        ).tolist()
        long_sample = long_texts.sample(
            n=min(cfg['long_sample_size'], len(long_texts)), replace=False
        ).tolist()
        return short_sample, long_sample, False

if __name__ == "__main__":
    data_selector = DataSampler(verbose=True)
    s1 = pd.Series([1,2,3,4,5,6,7,8,9,10, 'six', '-', '6a7'])
    prompt1 = data_selector.sample_column(s1, 'integer', 'INTEGER')
    s2 = pd.Series(["aa","aa","bb","bb","cc","cc","dd"])
    prompt2 = data_selector.sample_column(s2, 'string', 'DISCRETE_STRING')
    s3 = pd.Series([12,'$15','$46','55$','91 kg','80 kg','5a3','7.5%','100','Clean'])
    prompt3 = data_selector.sample_column(s3, 'dirties', 'DIRTY_INTEGER')
    print(prompt1)
    print()
    print(prompt2)
    print()
    print(prompt3)

    # python -m madclean.components.dataprofiler.data_selection

