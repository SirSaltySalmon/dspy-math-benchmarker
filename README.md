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
python math_solver.py
```

## Features

- Solves math problems with a provided calculator
- Displays step-by-step reasoning for each problem. 
- Records only the final answer for accuracy calculation
- Normalize answers by removing LaTeX formatting for accurate comparison between expected and predicted
- Tracks correct/incorrect answers
- Displays accuracy percentage

## Configuration

The application is configured to use a local Ollama server at `http://localhost:11434` with the `deepseek-r1` model. You can modify the configuration in `math_solver.py` if needed.

