# DSPy Math Solver & Benchmarker

A DSPy application that solves math problems, displays reasoning, and tracks accuracy. This is a beginner project to experiment with DSPy and errors are expected.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Modify code to connect with model and dataset to test. Here, Ollama runs locally with the `deepseek-r1` model, testing with HuggingFace :
```
ds = load_dataset("HuggingFaceH4/MATH-500")
ds = ds['test']  # Access the test split

# Configure DSPy with local Ollama server
lm = dspy.LM('ollama_chat/deepseek-r1', api_base='http://localhost:11434', api_key='')
```

## Usage

Run the math solver:
```bash
python math_solver_diy.py
```

## Features

- Solves math problems with a provided calculator
- Displays step-by-step reasoning for each problem. 
- Records only the final answer for accuracy calculation
- **Answer checking (configurable in `math_solver_diy.py`):** deterministic LaTeX/string normalization, or a small **LLM judge** that compares reference vs candidate answer only (ignores formatting). With `OVERLAP_JUDGE`, judging can overlap the next solve (same Ollama host may still serialize requests).
- Tracks correct/incorrect answers
- Displays accuracy percentage

## Configuration

Edit the constants at the top of [`math_solver_diy.py`](math_solver_diy.py):

- **`ANSWER_CHECK_MODE`:** `"normalize"` (default) or `"llm"`.
- **`OLLAMA_API_BASE`**, **`SOLVER_MODEL`**, **`JUDGE_MODEL`:** solver and optional judge (used when mode is `"llm"`).
- **`OVERLAP_JUDGE`:** when `True` and mode is `"llm"`, submit equivalence checks in a background thread so the next problem can start solving while the previous answer is judged.

The application defaults to a local Ollama server at `http://localhost:11434`.
