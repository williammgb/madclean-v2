import re
import ast
import warnings
import spacy 
import spacy.cli
import pandas as pd
import numpy as np
from collections import Counter 
from dateutil.parser import parse
from dateutil.parser._parser import UnknownTimezoneWarning
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
    # Per-value types grouped into families; a column whose biggest family holds under TYPE_THRESHOLD is MIXED.
    TYPE_FAMILIES = {
        'integer': 'numeric', 'float': 'numeric', 'dirty_integer': 'numeric', 'dirty_float': 'numeric',
        'datetime': 'datetime',
        'boolean': 'boolean',
        'named_entity': 'text', 'discrete_string': 'text', 'natural_language_text': 'text',
        'delimited_string': 'text', 'collection': 'text', 'email': 'text', 'url': 'text',
    }
    BOOLEAN_WORDS = {'t', 'f', 'y', 'n', 'true', 'false', 'yes', 'no', '0', '1'}
    IDENTIFIER_NAME_WORDS = {
        'id', 'zip', 'zipcode', 'postcode', 'postal', 'code', 'phone', 'fax', 'tel', 'number', 'no', 'num',
        'isbn', 'issn', 'ssn', 'ean', 'upc', 'sku',
    }
    IDENTIFIER_SHARE = 0.9
    CATEGORICAL_MAX_DISTINCT = 100
    CATEGORICAL_MAX_DISTINCT_SHARE = 0.1
    DATE_WORDS = {
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
        'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    }
    EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
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
            column_types[col] = self._infer_column_type(col_series, col)
        # if self.verbose: print(f"Column types: {column_types}"
        return column_types

    def _infer_column_type(self, col_series: pd.Series, col_name: str = "") -> str:
        """Infer the semantic type of a column based on its values and, for identifiers, its name."""
        if col_series.empty:
            return 'EMPTY'

        type_checkers = {
                'boolean': self._is_bool,
                'integer': self._is_integer,
                'float': self._is_float,
                'datetime': self._is_datetime
            }
        # 1. Quick check for 0/1 booleans, then for columns made only of boolean words (t/f, y/n, ...)
        is_zero_one = col_series.apply(lambda x: str(x).strip() in ['0', '1', '0.0', '1.0'])
        if is_zero_one.mean() > 0.95:
            return 'BOOLEAN'
        distinct_words = {str(x).strip().lower() for x in col_series} - {''}
        if distinct_words and distinct_words <= self.BOOLEAN_WORDS:
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
                    type_counts[self._classify_str_value(value, doc_dict)] += 1
            else:
                type_counts['unknown'] += 1
        # 4. Group the value types into families; no family holding TYPE_THRESHOLD of the values means MIXED
        if not type_counts:
            return 'EMPTY'
        total = sum(type_counts.values())
        family_counts = Counter()
        for type_name, count in type_counts.items():
            family_counts[self.TYPE_FAMILIES.get(type_name, type_name)] += count
        family, family_count = family_counts.most_common(1)[0]
        if family_count / total < self.type_threshold:
            inferred_type = 'MIXED'
        else:
            # 5. Inside the family, the most common type wins, as before the families existed
            in_family = Counter({t: c for t, c in type_counts.items() if self.TYPE_FAMILIES.get(t, t) == family})
            most_common_type = in_family.most_common(1)[0][0]
            if most_common_type == 'integer' and (type_counts.get('float', 0) / total > 0.05):
                inferred_type = 'FLOAT'
            else:
                inferred_type = most_common_type.upper()
        # 6. Codes and IDs, then small sets of repeated labels
        if inferred_type not in ('BOOLEAN', 'DATETIME') and self._is_identifier(col_series, col_name, inferred_type):
            return 'IDENTIFIER'
        if inferred_type in ('NAMED_ENTITY', 'DISCRETE_STRING') and self._is_categorical(col_series):
            return 'CATEGORICAL'
        return inferred_type

    @staticmethod
    def _digit_text(value) -> str:
        """The value as text, with whole floats written without '.0' (12345.0 -> '12345')."""
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    @classmethod
    def _name_last_word(cls, col_name: str) -> str:
        """Last word of a column name, split on non-letters and camelCase ('PhoneNumber' -> 'number')."""
        words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", str(col_name))
        return words[-1].lower() if words else ""

    def _is_identifier(self, col_series: pd.Series, col_name: str, inferred_type: str) -> bool:
        texts = [t for t in (self._digit_text(v) for v in col_series) if t]
        if not texts:
            return False
        share = self.IDENTIFIER_SHARE
        # 1. An identifier-like column name and mostly values with digits
        if self._name_last_word(col_name) in self.IDENTIFIER_NAME_WORDS:
            if sum(any(ch.isdigit() for ch in t) for t in texts) / len(texts) >= share:
                return True
        # 2. Whole numbers of one fixed length of 5 or more digits
        if inferred_type == 'INTEGER':
            lengths = Counter(len(t.lstrip('-')) for t in texts)
            length, count = lengths.most_common(1)[0]
            if length >= 5 and count / len(texts) >= share:
                return True
        # 3. Digit-only values with a leading zero, mostly of one length
        digit_only = [t for t in texts if t.isdigit()]
        if any(len(t) > 1 and t.startswith('0') for t in digit_only):
            if Counter(len(t) for t in texts).most_common(1)[0][1] / len(texts) >= share:
                return True
        return False

    def _is_categorical(self, col_series: pd.Series) -> bool:
        values = [str(v).strip() for v in col_series]
        values = [v for v in values if v]
        if not values:
            return False
        distinct = len(set(values))
        return distinct <= self.CATEGORICAL_MAX_DISTINCT and distinct <= self.CATEGORICAL_MAX_DISTINCT_SHARE * len(values)

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
        """Classify a string as email, url, collection, delimited, dirty numeric, named entity, natural language text or discrete string."""
        if self._is_email(value):
            return 'email'
        if self._is_url(value):
            return 'url'
        if self._is_collection_string(value):
            return 'collection'
        if self._is_delimited_string(value):
            return 'delimited_string'
        dirty_type = self._is_dirty_numeric(value)
        if dirty_type:
            return dirty_type
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
    def _is_collection_string(value: str) -> bool:
        """Detect Python-like collection literals stored as strings."""
        if not isinstance(value, str):
            return False
        raw = value.strip()
        if not raw:
            return False
        if raw == "set()":
            return True
        starts_ends = (raw.startswith('[') and raw.endswith(']')) \
            or (raw.startswith('{') and raw.endswith('}')) \
            or (raw.startswith('(') and raw.endswith(')'))
        if not starts_ends:
            return False
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return False
        return isinstance(parsed, (list, dict, set, tuple))

    @staticmethod
    def _is_delimited_string(value: str) -> bool:
        """Detect delimited strings such as 'a,b,c' while avoiding URL-like strings."""
        if not isinstance(value, str):
            return False
        raw = value.strip()
        if not raw:
            return False
        if "://" in raw:
            return False
        delimiters = [",", ";", "|", "/", "\\"]
        for delim in delimiters:
            if delim not in raw:
                continue
            if delim == "/" and raw.count("/") == 1 and re.fullmatch(r"\d+\s*/\s*\d+", raw):
                continue
            # A number with thousands separators ("$1,234.50", "1,234 kg") is one value, not a list
            if delim == "," and re.fullmatch(r"[^\d,]*[-+]?\d{1,3}(,\d{3})+(\.\d+)?[^\d,]*", raw):
                continue
            parts = [p.strip() for p in raw.split(delim)]
            if len(parts) < 2:
                continue
            if any(p == "" for p in parts):
                continue
            # Sentences with commas are text, not lists: every token is at most four words
            if any(len(p.split()) > 4 for p in parts):
                continue
            return True
        return False

    @classmethod
    def _is_email(cls, value) -> bool:
        return isinstance(value, str) and bool(cls.EMAIL_RE.match(value.strip()))

    @staticmethod
    def _is_url(value) -> bool:
        return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://", "www."))

    @staticmethod
    def _is_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ['true', 'false', 'yes', 'no']
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

    @classmethod
    def _is_datetime(cls, value) -> bool:
        if not isinstance(value, str):
            return False
        raw = value.strip()
        # A date has a digit or is exactly a month or weekday name ("May", "Mon"); "Sam" or "M" never is
        if not any(ch.isdigit() for ch in raw):
            if raw.lower().rstrip('.') not in cls.DATE_WORDS:
                return False
        # Only digits and separators: three numeric parts (2023-01-02, 1/2/23) or a hh:mm time, not "12-15"
        elif re.fullmatch(r"[\d\s\-/.:,]+", raw):
            parts = [p for p in re.split(r"\D+", raw) if p]
            if len(parts) < 3 and not re.search(r"\d{1,2}:\d{2}", raw):
                return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnknownTimezoneWarning)
                parse(value, fuzzy=False)
            return True
        except (ValueError, TypeError, OverflowError):
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