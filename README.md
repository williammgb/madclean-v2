# [name] - Multi-Agent Data Cleaning System
High-quality data is essential for effective decision-making, data analytics and the
training of machine learning models. However, real-world tabular datasets frequently
contain errors, inconsistencies and missing values that degrade data quality and hinder
downstream tasks. This thesis proposed an automated tabular data cleaning framework
that integrates statistical data profiling with the semantic reasoning and code generation capabilities of LLMs. This enables automated, adaptive and scalable data cleaning,
producing high-quality tabular datasets that are well suited for downstream analytical
and machine learning tasks.

## Installation
This framework was tested on Python version 3.11.
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
pip install .
```


## LLM configuration
You must configure an LLM provider to use the framework's features.

#### Using OpenAI or Gemini
Create a `.env` file in the project root and add your API key:
```
OPENAI_API_KEY="your_api_key_here"
# OR
GEMINI_API_KEY="your_api_key_here"
```
Update the `LLM_CLIENT` entry in settings.py to point to your LLM provider.

#### Using a custom LLM API
To add support for a new LLM provider:
1. Add a new LLM class to `llm_clients.py` following the existing abstract base class implementations.
2. Add your new class to the dictionary in `llm_registry.py`.
3. Update the `LLM_CLIENT` entry in `settings.py` to point to your new implementation.

## Usage
After installation, the framework can be run using the CLI command:
```bash
macs path/to/file.csv [OPTIONS]
```
#### Available options
`-v`, `--verbose`: Enable verbose output.  
`--save-cleaned`: Save the resulting cleaned dataset in the `data/cleaned` folder.  
`--help`: View full usage instrcutions.

### Example
```bash
macs data\benchmark_datasets\beers_dirty.csv --verbose --save-cleaned
```

## Security Note
This framework does **not** provide strong sandboxing guarantees for LLM-generated code.
Executing LLM-generated code should only be done with **trusted models and environments**.

## Adding Additional Cleaning Components

This section describes how to extend the framework with additional cleaning components.  
Two types of operations are supported: **single-column** and **multi-column** operations.

### Single-column operations
To add a new single-column cleaning component, follow these steps:
1. Define a new dataclass for the results data (`schema.py`)
2. Add the new dataclass to `ColumnProfile` (`schema.py`).
3. Create an analyser class following the protocol defined in `protocols.py`.
4. Pass an instance of the analyser to the `DataProfiler` class during initialization (`pipeline.py`).
5. Add a formatter method to the `PromptGeneration` class (`prompt_generation.py`) to convert the result data into a context block suitable for prompting.
6.  The generated context block will be automatically appended to the base prompt.

### Multi-column operations
To add a new multi-column cleaning component, follow these steps:
1. Define a new dataclass (`schema.py`).
2. Create an analyser class following the protocol defined in `protocols.py`.
3. Pass an instance of the analyser to the `DataProfiler` class during initialization (`pipeline.py`).
4. If the component has dependencies (e.g., some tasks have to be finished before others are executed), update `task_scheduler.py`.
5. Add the component to the `multi_col_config` of the Recommender Agent (`llm_recommending.py`).
6. Create recommender, coding and validation prompts.  
These prompts should:
    - include the relevant data and explanations
    - convert structured data into textual cleaning instructions for the LLM
7. Register the new prompts in `prompt_generation.py`.






