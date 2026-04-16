# MADClean - Multi-Agent Data Cleaning

High-quality data is essential for effective decision-making, data analytics and the training of machine learning models. However, real-world tabular datasets frequently contain errors, inconsistencies and missing values that degrade data quality and hinder downstream tasks. This thesis presents an automated tabular data cleaning framework that integrates statistical data profiling with the semantic reasoning and code generation capabilities of LLMs. Statistical profiling is used to extract dataset characteristics and relationships that guide the LLM in selecting appropriate cleaning operations. A multi-agent LLM architecture generates executable cleaning logic to ensure scalable and consistent data cleaning, while validation mechanisms are employed to verify correctness and improve reliability. Experimental results demonstrate that the framework achieves strong cleaning performance with competitive runtime, enabling automated and adaptive data cleaning and producing high-quality tabular data suitable for downstream analytical and machine learning tasks.

Architecture Overview

## Installation

This framework was tested on Python version 3.11.

### 1. Clone the repository

```bash
git clone https://github.com/qahtanaa/COSC-Thesis-William.git
cd COSC-Thesis-William
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

From this `Code` directory (`pyproject.toml` is here), run:

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

Update the `LLM_CLIENT_NAME` entry in the llm_settings.py to point to your LLM provider.

#### Using a custom LLM API

To add support for a new LLM provider:

1. Add a new client class in `Code/madclean/llm/llm_clients.py` following `BaseLLMClient`.
2. Register it in `Code/madclean/llm/llm_registry.py` (API keys via `.env` as documented there).
3. Use the new key from the registry in the UI or wire it as the default for your entrypoint.

## Usage

After installation, the framework can be run using the CLI command:

```bash
madclean path/to/file.csv [OPTIONS]
```

#### Available options

`-v`, `--verbose`: Enable verbose output.  
`--save-cleaned`: Save the resulting cleaned dataset in the `data/cleaned` folder.  
`--help`: View full usage instructions.

### Example

```bash
madclean data\benchmark_datasets\beers_dirty.csv --verbose --save-cleaned
```

## Running the UI

MADClean also includes an interactive web UI.

### 1. Configure API keys

Create a `.env` file in the project root (same as CLI) and add your API key(s), e.g.:

```
OPENAI_API_KEY="your_api_key_here"
# OR
GEMINI_API_KEY="your_api_key_here"
```

### 2. Start the UI

From the repository root, activate your environment and run:

```bash
madclean ui
```

Reflex will print the URLs in the terminal (typically a frontend on `http://localhost:300x/` and a backend on `http://0.0.0.0:8000`).

### 3. Use the UI

- **Upload Data**: upload a CSV (or Excel) to preview the table.
- **Model selection**: choose an LLM **per agent** (Recommender / Coding / Validation), or **USER** for manual validation.
- **Advanced Configuration**: adjust cleaning/validation settings, sample sizes, HITL, etc.
- **Run Pipeline**: watch logs + progress; use **Pipeline** for step-level detail when HITL or user validation is active.
- **Usage** / **Report**: tokens, runtime, and generated cleaning code.
- **Evaluation** (optional): upload a **ground-truth** file with the same columns and row count as the dirty dataset to score cleaning after a successful run.
- **Edit + Download**: edit cells in the cleaned table and export the edited CSV.

UI

## Security Note

This framework does **not** provide strong sandboxing guarantees for LLM-generated code.
Executing LLM-generated code should only be done with **trusted models and environments**.

## Adding additional cleaning components

**Pipeline:** `DataProfiler` → profiles + multi-column task detection; `CleaningCoordinator` → single-column then multi-column cleaning; `MultiAgentCleaning` → recommender, coder, validator.

### Single-column

1. Implement `SingleColumnCleaner` (`madclean/components/domain/protocols.py`): `analyse(df, col, column_type) -> dict`.
2. Map results to `ColumnProfile` fields and/or `profile.metadata` (`madclean/components/domain/schema.py`); wire prompts in `PromptGeneration.create_prompt_recommender` / `context_formatters` (`madclean/components/coordinator/prompt_generation.py`).
3. Append your class to `single_col_cleaners` in `madclean/pipeline.py`.

### Multi-column (e.g. FD-style constraints)

1. Implement `MultiColumnCleaner` (`protocols.py`): `task_type`, `detect(df)`, `get_data(df, task_info)`; put structured payloads in `MultiColumnTask.data` (`schema.py`; see `functional_dependencies.py`).
2. Append to `multi_col_cleaners` in `madclean/pipeline.py`.
3. Adjust `task_scheduler.py` if you need ordering beyond FD-topology + one batch for other types.
4. Add `multi_col_config` entry in `llm_recommending.py`; recommender/coder/validation prompts in `prompts.py`; branches in `create_prompt_*_multi_col` in `prompt_generation.py` (use `**FD`** as the working template; only `FD` is wired end-to-end in the current implementation).