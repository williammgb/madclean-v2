import re
import spacy 
import spacy.cli
import pandas as pd
import numpy as np
from collections import Counter 
from dateutil.parser import parse 
# Local imports 
from madclean.utils.helpers import subsample_dataframe


class SemanticTypeDetection:
    """
    Class for detecting pre-defined semantic column types in dirty datasets in a robust and scalable way.
    Samples the dataset and determines type of each value.
    """
    TYPE_THRESHOLD = 0.8 
    TOKENS = {
        "$", "€", "¥", "£", "USD", "EUR", "~", "+/-", ">=", "<=",
        "%", "percent",
        "kg", "lbs", "oz", "oz.", "ounce", "g", "t", "stone",
        "C", "F",
        "cm", "m", "km", "miles", "yd", "mm",
        "s", "sec", "second", "seconds", "min", "minutes", "h", "hours", "d", "days", "yrs", "years",
        "km/h", "mph", "m/s", "knots",
        "V", "Volt", "W", "watt", "Hz", "dB", "Ω",
        "pcs", "items", "units", "dozen",
        "times"
    }
    NAMED_ENTITY_CATS = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
    NLP_MODEL_NAME = "en_core_web_sm"
    def __init__(self, verbose: bool = False):
        self.verbose = verbose        
        self.nlp_model_name = self.NLP_MODEL_NAME
        self.type_threshold = self.TYPE_THRESHOLD
        self.tokens = self.TOKENS
        self.named_entity_cats = self.NAMED_ENTITY_CATS
        self.nlp_model = self._load_spacy_model()

    def _load_spacy_model(self):
        """Load the spaCy model, download if necessary."""
        try:
            return spacy.load(self.nlp_model_name,
                            disable=["parser", "attribute_ruler", "lemmatizer"]) 
        except OSError:
            if self.verbose: print(f"Spacy model {self.nlp_model_name} not found. Downloading now...")
            spacy.cli.download(self.nlp_model_name)
            return spacy.load(self.nlp_model_name,
                            disable=["parser", "attribute_ruler", "lemmatizer"])

    def detect_types(self, df: pd.DataFrame) -> dict:
        """Detects semantic types for each column in the DataFrame."""
        # 1. Sample dataframe
        df_sample = subsample_dataframe(df)
        if df_sample.empty:
            return {}
        # 2. Infer semantic type for each column
        column_types = {}
        for col in df_sample.columns:
            col_series = df_sample[col].dropna()
            column_types[col] = self._infer_column_type(col_series)
        # if self.verbose: print(f"Column types: {column_types}"
        return column_types

    def _infer_column_type(self, col_series: pd.Series) -> dict:
        """Infer the semantic type of a column based on its values."""
        if col_series.empty:
            return 'EMPTY'
        
        type_checkers = {
                'boolean': self._is_bool,
                'integer': self._is_integer,
                'float': self._is_float,
                'datetime': self._is_datetime
            }
        # 1. Quick check for 0/1 booleans
        is_zero_one = col_series.apply(lambda x: str(x).strip() in ['0', '1', '0.0', '1.0'])
        if is_zero_one.mean() > 0.95:
            return 'BOOLEAN'
        # 2. Batch-process unique strings with NLP model, speeds up significantly
        string_values = col_series[col_series.apply(lambda x: isinstance(x, str) and x.strip() != '')].unique()
        doc_dict = {}
        if string_values.size > 0:
            clean_string_values = [s.strip() for s in string_values]
            docs = self.nlp_model.pipe(clean_string_values, batch_size=50)
            doc_dict = dict(zip(clean_string_values, docs))        
        # 3. Iterate through each individual value and identify type
        type_counts = Counter()
        for value in col_series.values:
            found_type = False
            for type_name, checker in type_checkers.items():
                if checker(value):
                    type_counts[type_name] += 1
                    found_type = True
                    break
            if found_type:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                else:
                    dirty_type = self._is_dirty_numeric(value)
                    if dirty_type:
                        type_counts[dirty_type] +=1
                    else:
                        type_counts[self._classify_str_value(value, doc_dict)] += 1
            else:
                type_counts['unknown'] += 1
        # 4. Assign type to each column based on most common occuring type
        if not type_counts:
            return 'EMPTY'
        most_common_type, count = type_counts.most_common(1)[0]
        total = sum(type_counts.values())
        if most_common_type == 'integer' and (type_counts.get('float', 0) / total > 0.05):
            return 'FLOAT'

        inferred_type =  most_common_type.upper()
        return inferred_type 

    def _is_dirty_numeric(self, value: str) -> str | None:
        """Check if value is dirty numeric value (e.g., "$12.50" or "12 kg")."""
        if len(value.split()) > 2:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match:
            number_str = match.group()
            start, end = match.span()
            before = value[:start].strip()
            after = value[end:].strip()
            if before in self.tokens or after in self.tokens:
                return "dirty_float" if "." in number_str else "dirty_integer"
        return None
    
    def _classify_str_value(self, value: str, doc_dict: dict) -> str:
        """Classify a string as named entity, natural language text or discrete string."""    
        doc = doc_dict.get(value)
        if not doc:
            return 'discrete_string'
        word_count = len(value.split())
        entities = [ent for ent in doc.ents if ent.label_ in self.named_entity_cats]
        if word_count == 1:
            return 'named_entity' if entities else 'discrete_string'
        if word_count > 5 or len(value) > 50: 
            return 'natural_language_text'
        if len(entities) == 1: 
            entity_ratio = len(entities[0].text.split()) / word_count
            return 'named_entity' if entity_ratio > 0.5 else 'natural_language_text'
        elif len(entities) > 1:
            return 'natural_language_text'
        else:
            has_stopword = any(token.is_stop for token in doc)
            has_verb = any(token.pos_ == 'VERB' for token in doc)
            return 'natural_language_text' if has_stopword or has_verb else 'discrete_string'

    @staticmethod
    def _is_bool(value) -> bool:
        if isinstance(value, str):
            return value.lower() in ['true', 'false', 't', 'f', 'yes', 'no', 'y', 'n'] 
        return isinstance(value, (bool, np.bool_))
    
    @staticmethod
    def _is_integer(value) -> bool:
        if isinstance(value, (int, np.integer)):
            return True 
        if isinstance(value, (float, np.floating)):
            return value.is_integer()        
        if isinstance(value, str):
            try:
                return float(value.replace(',', '')).is_integer()
            except (ValueError, TypeError):
                return False
        return False

    @staticmethod
    def _is_float(value) -> bool:
        if isinstance(value, (float, np.floating)):
            return not value.is_integer()
        if isinstance(value, str):
            try:
                return not float(value.replace(',', '')).is_integer()
            except (ValueError, TypeError):
                return False
        return False

    @staticmethod
    def _is_datetime(value) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parse(value, fuzzy=False)
            return True
        except (ValueError, TypeError):
            return False

####### TEST CODE #######
if __name__ == "__main__":
    df = pd.DataFrame({
    "int_col": [1, 2, 3, 4, 5, 6, 7, 8],
    "float_col": [1.5, 2.7, 3.1, 4.9, 5.0, 6.3, 7.8, 8.2],
    "date_str": [
        "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13",
        "2024-01-14", "2024-01-15", "2024-01-16", "2024-01-17"
    ],
    "bool_col": [True, False, True, False, True, False, True, False],
    "weight_col": ["5 kg", "12 kg", "7 kg", "9 kg", "3 kg", "15 kg", "8 kg", "11 kg"],
    "percent_col": ["12.5%", "44.1%", "9.3%", "88.0%", "51.6%", "23.4%", "67.0%", "30.2%"],
    "country": ["USA", "Canada", "Germany", "Japan", "Brazil", "India", "France", "Mexico"],
    "pattern_col": ["1234AB", "5678CD", "9101EF", "2345GH", "6789IJ", "1111KL", "2222MN", "3333OP"],
    "description": [
        "This product shows consistent performance levels",
        "A detailed report with several important insights",
        "System behavior remains stable across conditions",
        "User feedback indicates improved overall experience",
        "Model accuracy increased after recent adjustments",
        "Dataset demonstrates diverse and complex patterns",
        "The algorithm performs well under heavy load",
        "New release includes multiple significant upgrades"
    ],
    "unknown_col": [[], [], [], [], [], [], [], []],
    "empty_col": [None] * 8
    })
    detector = SemanticTypeDetection(verbose=True)
    column_types = detector.detect_types(df)
    print(column_types)
    # python -m madclean.components.dataprofiler.semantic_mapping