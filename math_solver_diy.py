import traceback
from concurrent.futures import Future, ThreadPoolExecutor

import dspy
from datasets import load_dataset

from answer_judge import answers_equivalent
from normalize import normalize_answer

# --- User configuration ---
# Answer checking: "normalize" (deterministic string match) or "llm" (equivalence via judge LM).
ANSWER_CHECK_MODE = "normalize"

OLLAMA_API_BASE = "http://localhost:11434"
SOLVER_MODEL = "ollama_chat/deepseek-r1"
# Used only when ANSWER_CHECK_MODE == "llm"; typically a smaller / faster model.
JUDGE_MODEL = "ollama_chat/llama3.2:1b"

# When True and mode is "llm", start judging the previous problem while solving the next.
OVERLAP_JUDGE = False

solver_lm = dspy.LM(SOLVER_MODEL, api_base=OLLAMA_API_BASE, api_key="")
judge_lm = dspy.LM(
    JUDGE_MODEL,
    api_base=OLLAMA_API_BASE,
    api_key="",
    temperature=0,
    max_tokens=64,
)

dspy.configure(lm=solver_lm)

ds = load_dataset("HuggingFaceH4/MATH-500")
ds = ds["test"]


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Use safely in production
        return f"The result is {result}"
    except Exception:
        return "Invalid expression"


examples = ds.select(range(10))
example_answers = [item["answer"] for item in examples]
example_format = " | ".join(example_answers) if example_answers else ""


class QASignature(dspy.Signature):
    """Solve a math problem step by step."""

    question: str = dspy.InputField(desc="The math problem to solve")
    answer: str = dspy.OutputField(
        desc=(
            "The final numerical or algebraic answer ONLY. No other text or explanation. "
            f"Example formatting: {example_format}"
        )
    )


math_solver = dspy.ReAct(signature=QASignature, tools=[calculator])


def is_correct_normalize(expected_answer: str, predicted_raw: str) -> bool:
    predicted_norm = normalize_answer(predicted_raw or "")
    if not predicted_norm:
        return False
    expected_norm = normalize_answer(expected_answer)
    return predicted_norm == expected_norm


def is_correct_llm(expected_answer: str, predicted_raw: str) -> bool:
    return answers_equivalent(expected_answer, predicted_raw or "", judge_lm)


def _print_status_block(is_correct: bool, tasks_done: int, tasks_correct: int) -> None:
    print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print(f"Tasks Done: {tasks_done}")
    print(f"Tasks Correct: {tasks_correct}")
    if tasks_done > 0:
        print(f"Accuracy: {tasks_correct / tasks_done * 100:.2f}%")
    print("-" * 60)


def _drain_judge_future(
    judge_future: Future | None,
    pending: dict | None,
    tasks_done: int,
    tasks_correct: int,
) -> tuple[int, int, None, None]:
    if judge_future is None or pending is None:
        return tasks_done, tasks_correct, None, None
    try:
        ok = judge_future.result()
    except Exception as e:
        print(f"Judge error ({type(e).__name__}): {e}")
        ok = False
    tasks_done += 1
    if ok:
        tasks_correct += 1
    _print_status_block(ok, tasks_done, tasks_correct)
    return tasks_done, tasks_correct, None, None


tasks_done = 0
tasks_correct = 0

executor: ThreadPoolExecutor | None = None
if ANSWER_CHECK_MODE == "llm" and OVERLAP_JUDGE:
    executor = ThreadPoolExecutor(max_workers=1)

judge_future: Future | None = None
pending_result: dict | None = None

try:
    for idx, item in enumerate(ds):
        problem_num = idx + 1

        if ANSWER_CHECK_MODE == "llm" and OVERLAP_JUDGE and judge_future is not None:
            tasks_done, tasks_correct, judge_future, pending_result = _drain_judge_future(
                judge_future, pending_result, tasks_done, tasks_correct
            )

        question = item["problem"]
        expected_answer = item["answer"]

        print(f"\nQuestion {problem_num}: {question}")
        print("-" * 60)

        try:
            with dspy.context(lm=solver_lm):
                response = math_solver(question=question)

            print("Reasoning:")
            reasoning_text = ""
            if hasattr(response, "trajectory") and response.trajectory:
                if isinstance(response.trajectory, list):
                    for step in response.trajectory:
                        if isinstance(step, dict):
                            if "thought" in step:
                                reasoning_text += step["thought"] + "\n"
                            if "observation" in step:
                                reasoning_text += f"Observation: {step['observation']}\n"
                        elif isinstance(step, str):
                            reasoning_text += step + "\n"
                    print(reasoning_text if reasoning_text else str(response.trajectory))
                else:
                    print(str(response.trajectory))
            elif hasattr(response, "history") and response.history:
                for step in response.history:
                    if hasattr(step, "reasoning") or isinstance(step, str):
                        print(step)
            else:
                if hasattr(response, "_completions") and response._completions:
                    print("(Reasoning embedded in model response)")
                else:
                    print("(Null)")
            print()

            predicted_raw = response.answer if hasattr(response, "answer") and response.answer else ""
            if not predicted_raw:
                print("Warning: No answer found in response")

            print(f"Predicted Answer: {predicted_raw or 'N/A'}")
            print(f"Expected Answer: {expected_answer}")

            if ANSWER_CHECK_MODE == "normalize":
                predicted_norm = normalize_answer(predicted_raw)
                expected_norm = normalize_answer(expected_answer)
                print(f"Normalized Predicted: {predicted_norm}")
                print(f"Normalized Expected: {expected_norm}")
                is_correct = is_correct_normalize(expected_answer, predicted_raw)
                tasks_done += 1
                if is_correct:
                    tasks_correct += 1
                _print_status_block(is_correct, tasks_done, tasks_correct)

            elif ANSWER_CHECK_MODE == "llm":
                print("Answer check: LLM judge (reference vs candidate only).")
                if OVERLAP_JUDGE:
                    assert executor is not None
                    judge_future = executor.submit(
                        answers_equivalent,
                        expected_answer,
                        predicted_raw,
                        judge_lm,
                    )
                    pending_result = {}
                else:
                    try:
                        is_correct = is_correct_llm(expected_answer, predicted_raw)
                    except Exception as e:
                        print(f"Judge error ({type(e).__name__}): {e}")
                        is_correct = False
                    tasks_done += 1
                    if is_correct:
                        tasks_correct += 1
                    _print_status_block(is_correct, tasks_done, tasks_correct)
            else:
                raise ValueError(
                    f"Unknown ANSWER_CHECK_MODE: {ANSWER_CHECK_MODE!r} (use 'normalize' or 'llm')"
                )

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError processing question: {type(e).__name__}: {e}")
            traceback.print_exc()
            tasks_done += 1
            continue

except KeyboardInterrupt:
    print("\n\nInterrupted by user.")

finally:
    if ANSWER_CHECK_MODE == "llm" and OVERLAP_JUDGE and judge_future is not None:
        try:
            tasks_done, tasks_correct, judge_future, pending_result = _drain_judge_future(
                judge_future, pending_result, tasks_done, tasks_correct
            )
        except Exception:
            pass
    if executor is not None:
        executor.shutdown(wait=True)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Total Problems: {tasks_done}")
print(f"Correct Answers: {tasks_correct}")
print(f"Incorrect Answers: {tasks_done - tasks_correct}")
if tasks_done > 0:
    print(f"Accuracy: {tasks_correct / tasks_done * 100:.2f}%")
print("=" * 60)
