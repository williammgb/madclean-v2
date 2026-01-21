# [name] - Multi-Agent Data Cleaning System
High-quality data is essential for effective decision-making, data analytics and the
training of machine learning models. However, real-world tabular datasets frequently
contain errors, inconsistencies and missing values that degrade data quality and hinder
downstream tasks. This thesis proposed an automated tabular data cleaning framework
that integrates statistical data profiling with the semantic reasoning and code generation capabilities of LLMs. This enables automated, adaptive and scalable data cleaning,
producing high-quality tabular datasets that are well suited for downstream analytical
and machine learning tasks.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/qahtanaa/COSC-Thesis-William.git thesis-repo
cd thesis-repo/code
```
### 2. (Optional but recommended) Create a virtual environment
Example using Python 3.11 (Windows):
```bash
py -3.11 -m venv my_env
my_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the spaCy NER model
```bash
python -m spacy download en_core_web_sm
```

### 5. Install the framework
From the project root directory, run:
```bash
pip install -e .
```


## LLM configuration
TBD


