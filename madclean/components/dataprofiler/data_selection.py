import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from typing import Callable
from dateutil.parser import parse
from rapidfuzz import fuzz, process, utils
# Local imports
from madclean.utils.helpers import format_list_for_prompt, seeded_generator

# Types whose sample is a list of values with their counts, and those that also get near-duplicate groups or shapes
COUNTED_TYPES = {'NAMED_ENTITY', 'DISCRETE_STRING', 'CATEGORICAL', 'IDENTIFIER', 'EMAIL', 'URL', 'MIXED',
                 'BOOLEAN', 'COLLECTION', 'DELIMITED_STRING'}
VARIANT_TYPES = {'NAMED_ENTITY', 'DISCRETE_STRING', 'CATEGORICAL', 'IDENTIFIER', 'MIXED'}
SHAPE_TYPES = {'DATETIME', 'IDENTIFIER', 'MIXED'}

VALUE_WIDTH = 80
VARIANT_CHOICES = 500
VARIANT_MAX_TRIED = 2000
VARIANT_MAX_LINES = 100
VARIANT_SCORE_CUTOFF = 90
VARIANT_RARE_COUNT = 2
RARE_SLOTS = 100
MAX_SHAPES = 50
# Characters each part of a counted sample may take, so a column of long values (cast lists, titles)
# costs about as much as a column of short ones.
VALUES_CHAR_BUDGET = 8000
VARIANTS_CHAR_BUDGET = 3000
RANDOM_CHAR_BUDGET = 2000
_LETTER_RUN = re.compile(r"[^\W\d_]+")
_FIRST_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def _display(value) -> str:
    """A value as it reads in a counts list: plain text, quoted only when spaces at its ends would be lost."""
    if hasattr(value, "item") and not isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        value = value.item()
    text = str(value)
    if len(text) > VALUE_WIDTH:
        text = text[:VALUE_WIDTH] + "…"
    if text == "" or text != text.strip() or "\n" in text:
        text = json.dumps(text, ensure_ascii=False)
    return text


def value_shape(value) -> str:
    """Each digit -> 9, each run of letters -> A (capitals), Aa (capitalised word) or a (lower case)."""
    def _letters(match: re.Match) -> str:
        run = match.group(0)
        if run.isupper():
            return "A"
        if run.islower():
            return "a"
        return "Aa"
    return _LETTER_RUN.sub(_letters, re.sub(r"\d", "9", str(value)))


def _within_budget(items: list[str], budget: int) -> list[str]:
    """The leading items whose combined length stays within `budget` characters; always at least one."""
    kept, used = [], 0
    for item in items:
        used += len(item) + 2
        if kept and used > budget:
            break
        kept.append(item)
    return kept


def _ranked_counts(series: pd.Series) -> list[tuple]:
    """(value, count) pairs, most frequent first, ties by value."""
    return sorted(series.value_counts().items(), key=lambda kv: (-kv[1], str(kv[0])))


class DataSampler:
    """
    Class to sample data in more intelligent manner to improve LLM cleaning performance
    by providing more informative dataset sample to the LLM.
    """

    def __init__(self, sample_sizes: dict, seed: int | None = None):
        self.sample_sizes = sample_sizes
        self.seed = seed

    def sample_column(self, column: pd.Series, col: str, column_type: str) -> str:
        """Creates an intelligent string sample of a column based on its semantic type."""
        column = column.dropna()
        if column.empty or column_type == "EMPTY":
            return f'Column "{col}" sample → [EMPTY]'
        rng = seeded_generator(self.seed, "profile", col)
        header = f"Column '{col}':"
        # Text-like types: every value with its count, near-duplicate groups and shapes where they apply
        if column_type in COUNTED_TYPES:
            return f"{header}\n" + self._sample_column_counted(column, "STRING", rng, column_type)
        TYPE_MAP: dict[str, Callable[[pd.Series], tuple[list, list]]] = {
            'INTEGER': ("NUMERIC", self._sample_column_numeric),
            'FLOAT': ("NUMERIC", self._sample_column_numeric),
            'DATETIME': ("DATETIME", self._sample_column_datetime),
            'DIRTY_INTEGER': ("DIRTY_NUMERIC", self._sample_column_dirty_numeric),
            'DIRTY_FLOAT': ("DIRTY_NUMERIC", self._sample_column_dirty_numeric),
            'NATURAL_LANGUAGE_TEXT': ("NLT", self._sample_column_nlt),
        }
        # 1. Select sample configurations based on semantic type
        sample_sizes_cfg, sampling_func = TYPE_MAP.get(column_type)
        sample1, sample2, full_sample = sampling_func(column, sample_sizes_cfg, rng)
        # 2. Generate column sample as a string
        if column_type in ['INTEGER', 'FLOAT']:
            msg = "\nDirty Sample includes ALL dirty values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Clean Sample (convertible to numeric): [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Dirty Sample (could not be converted): [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}"
                f"{self._numeric_summary(column, dirty=False)}")
        elif column_type in ['DIRTY_INTEGER', 'DIRTY_FLOAT']:
            msg = "\nDirty Sample includes ALL unique dirty patterns present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Random Sample: [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Dirty Sample (unique noise pattern representatives): [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}"
                f"{self._numeric_summary(column, dirty=True)}")
        elif column_type == 'DATETIME':
            msg = "\nDirty Sample includes ALL dirty values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Clean Sample (convertible to datetime): [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Dirty Sample (could not be converted): [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}"
                f"{self._shape_lines(column)}")
        else:
            msg = "\nSample includes ALL values present in the dataset." if full_sample else ""
            prompt = (
                f"{header}\n"
                f"Short Text Sample: [{', '.join(format_list_for_prompt(sample1))}]\n"
                f"Long Text Sample: [{', '.join(format_list_for_prompt(sample2))}]"
                f"{msg}")
        return prompt

    # ====================== counts, variants, shapes, numeric summary ==========================
    def _sample_column_counted(self, series: pd.Series, sample_sizes_cfg: str, rng: np.random.Generator | None,
                               column_type: str) -> str:
        """Every distinct value with its count (most frequent first), near-duplicate groups, shapes and,
        when not every value fits, a random sample."""
        cfg = self.sample_sizes[sample_sizes_cfg]
        limit = cfg['unique_sample_size']
        ranked = _ranked_counts(series)
        groups = self._variant_groups(ranked, rng) if column_type in VARIANT_TYPES else []
        all_included = len(ranked) <= limit
        if all_included:
            listed = ranked
        else:
            # The most frequent values, plus rare values that look like a variant of a frequent one
            rare_slots = min(RARE_SLOTS, limit // 2)
            frequent = ranked[:limit - rare_slots]
            kept = {value for value, _ in frequent}
            rare = [(v, c) for _, variants in groups for v, c in variants if v not in kept][:rare_slots]
            kept.update(v for v, _ in rare)
            fill = [(v, c) for v, c in ranked[limit - rare_slots:] if v not in kept][:limit - len(frequent) - len(rare)]
            chosen = {v for v, _ in frequent + rare + fill}
            listed = [(v, c) for v, c in ranked if v in chosen]
        value_lines = _within_budget([f"{_display(v)} ({c})" for v, c in listed], VALUES_CHAR_BUDGET)
        lines = ["Values with counts (most frequent first):"] + value_lines
        if len(value_lines) < len(listed):
            all_included = False
        if all_included:
            lines.append("Values with counts include ALL distinct values present in the dataset.")
        else:
            lines.append(f"… {len(ranked) - len(value_lines)} more distinct values not listed.")
        if groups:
            lines.append("Possible variants of the same value:")
            variant_lines = []
            for (anchor, anchor_count), variants in groups[:VARIANT_MAX_LINES]:
                shown = _within_budget([f"{_display(v)} ({c})" for v, c in variants], VARIANTS_CHAR_BUDGET // 2)
                more = f", … {len(variants) - len(shown)} more" if len(shown) < len(variants) else ""
                variant_lines.append(f"{_display(anchor)} ({anchor_count}) ← {', '.join(shown)}{more}")
            lines += _within_budget(variant_lines, VARIANTS_CHAR_BUDGET)
        prompt = "\n".join(lines)
        if column_type in SHAPE_TYPES:
            prompt += self._shape_lines(series)
        if not all_included:
            random_sample = series.sample(n=min(cfg['random_sample_size'], len(series)), replace=False,
                                          random_state=rng).tolist()
            shown = _within_budget(format_list_for_prompt(random_sample), RANDOM_CHAR_BUDGET)
            prompt += f"\nRandom Sample: [{', '.join(shown)}]"
        return prompt

    @staticmethod
    def _variant_groups(ranked: list[tuple], rng: np.random.Generator | None) -> list[tuple]:
        """Rare values (seen at most twice) matched to the frequent value they closely resemble.
        Returns [((anchor, count), [(variant, count), ...]), ...], biggest group first."""
        choices = [(v, c) for v, c in ranked[:VARIANT_CHOICES] if c > VARIANT_RARE_COUNT]
        rare = [(v, c) for v, c in ranked if c <= VARIANT_RARE_COUNT]
        if not choices or not rare:
            return []
        if len(rare) > VARIANT_MAX_TRIED:
            picks = (rng if rng is not None else np.random).choice(len(rare), size=VARIANT_MAX_TRIED, replace=False)
            rare = [rare[i] for i in sorted(picks)]
        choice_texts = [str(v) for v, _ in choices]
        grouped: dict[int, list] = {}
        for value, count in rare:
            match = process.extractOne(str(value), choice_texts, scorer=fuzz.WRatio,
                                       processor=utils.default_process, score_cutoff=VARIANT_SCORE_CUTOFF)
            if match is not None:
                grouped.setdefault(match[2], []).append((value, count))
        groups = [(choices[i], sorted(variants, key=lambda kv: (-kv[1], str(kv[0]))))
                  for i, variants in grouped.items()]
        return sorted(groups, key=lambda g: (-len(g[1]), -g[0][1], str(g[0][0])))

    @staticmethod
    def _shape_lines(series: pd.Series) -> str:
        """Values grouped by shape, most frequent shape first, each with its most common value as the example."""
        shape_counts: Counter = Counter()
        examples: dict[str, tuple] = {}
        for value, count in _ranked_counts(series):
            shape = value_shape(value)
            shape_counts[shape] += count
            examples.setdefault(shape, value)
        shapes = sorted(shape_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_SHAPES]
        lines = [f"{_display(shape)} ({count}), e.g. {_display(examples[shape])}" for shape, count in shapes]
        return "\nFormats by shape (count, example):\n" + "\n".join(lines)

    @staticmethod
    def _numeric_summary(series: pd.Series, dirty: bool) -> str:
        """Range, quartiles and how many values have each number of decimal places."""
        texts = []
        for value in series:
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                texts.append(repr(float(value)).removesuffix(".0"))
            elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                texts.append(str(int(value)))
            elif dirty:
                match = _FIRST_NUMBER.search(str(value))
                if match:
                    texts.append(match.group(0))
            else:
                text = str(value).strip()
                if pd.notna(pd.to_numeric(text, errors="coerce")):
                    texts.append(text)
        numbers = pd.to_numeric(pd.Series(texts, dtype=object), errors="coerce")
        keep = numbers.notna() & np.isfinite(numbers.astype(float))
        numbers, texts = numbers[keep].astype(float), [t for t, k in zip(texts, keep) if k]
        if numbers.empty:
            return ""
        def _number(x: float) -> str:
            return f"{x:.10g}"
        q = numbers.quantile([0.25, 0.5, 0.75])
        places = Counter(len(t.split(".", 1)[1]) if "." in t and "e" not in t.lower() else 0 for t in texts)
        decimals = ", ".join(f"{p} → {n} values" for p, n in sorted(places.items()))
        return (f"\nNumeric summary: min {_number(numbers.min())}, 25% {_number(q[0.25])}, median {_number(q[0.5])}, "
                f"75% {_number(q[0.75])}, max {_number(numbers.max())}; decimal places: {decimals}")

    # ====================== type-specific sampling ==========================
    def _sample_column_numeric(self, series: pd.Series, sample_sizes_cfg: str, rng: np.random.Generator | None = None) -> tuple[list, list, bool]:
        """Separate numeric values from non-numeric (dirty) values for INTEGER and FLOAT columns."""
        cfg = self.sample_sizes[sample_sizes_cfg]
        coerced_series = pd.to_numeric(series, errors='coerce')
        # 1. Gather 'clean' sample: values that can be converted to numeric
        clean_mask = coerced_series.notna()
        clean_sample = series[clean_mask].sample(n=min(cfg['clean_sample_size'], clean_mask.sum()), replace=False, random_state=rng).tolist()
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

    def _sample_column_dirty_numeric(self, series: pd.Series, sample_sizes_cfg: str, rng: np.random.Generator | None = None) -> tuple[list, list, bool]:
        """ Provides random sample and list of all unique dirty patterns for DIRTY_FLOAT and DIRTY_INTEGER columns."""
        cfg = self.sample_sizes[sample_sizes_cfg]
        # 1. Take random sample
        random_sample = series.sample(n=min(cfg['random_sample_size'], len(series)), replace=False, random_state=rng).tolist()
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
            unique_dirty_values = (rng if rng is not None else np.random).choice(unique_dirty_values, size=cfg['unique_sample_size'], replace=False).tolist()
        return random_sample, unique_dirty_values, all_unique_included

    def _sample_column_datetime(self, series: pd.Series, sample_sizes_cfg: str, rng: np.random.Generator | None = None) -> tuple[list, list, bool]:
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
        cfg = self.sample_sizes[sample_sizes_cfg]
        coerced_series = series.apply(_robust_date_parser)
        # 1. Gather 'clean' sample: values that can be converted to datetime
        clean_mask = coerced_series.notna()
        clean_sample = series[clean_mask].sample(n=min(cfg['clean_sample_size'], clean_mask.sum()), replace=False, random_state=rng).tolist()
        # 2. Gather 'dirty' sample: values that cannot be converted to datetime
        dirty_mask = coerced_series.isna()
        dirty_sample = series[dirty_mask].unique().tolist()
        all_dirty_included = len(dirty_sample) <= cfg['dirty_sample_size']
        if not all_dirty_included:
            dirty_sample = (rng if rng is not None else np.random).choice(dirty_sample, size=cfg['dirty_sample_size'], replace=False).tolist()
        return clean_sample, dirty_sample, all_dirty_included

    def _sample_column_nlt(self, series: pd.Series, sample_sizes_cfg: str, rng: np.random.Generator | None = None)-> tuple[list, list, bool]:
        """Samples short and long text values for NATURAL_LANGUAGE_TEXT columns."""
        cfg = self.sample_sizes[sample_sizes_cfg]
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
            n=min(cfg['short_sample_size'], len(short_texts)), replace=False, random_state=rng
        ).tolist()
        long_sample = long_texts.sample(
            n=min(cfg['long_sample_size'], len(long_texts)), replace=False, random_state=rng
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
