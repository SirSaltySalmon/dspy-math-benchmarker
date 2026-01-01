import dspy
from typing import List, Dict
from datasets import load_dataset
from normalize import normalize_answer

ds = load_dataset("HuggingFaceH4/MATH-500")
ds = ds['test']  # Access the test split

# Configure DSPy with local Ollama server
lm = dspy.LM('ollama_chat/deepseek-r1', api_base='http://localhost:11434', api_key='')

dspy.configure(lm=lm)

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Use safely in production
        return f"The result is {result}"
    except:
        return "Invalid expression"

examples = ds.select(range(10))  # Get first 10 examples
example_answers = [item["answer"] for item in examples]
# Format examples for description
example_format = ' | '.join(example_answers) if len(example_answers) > 0 else ''

class QASignature(dspy.Signature):
    """Solve a math problem step by step."""
    question: str = dspy.InputField(desc="The math problem to solve")
    answer: str = dspy.OutputField(desc=f"The final numerical or algebraic answer ONLY. No other text or explanation. Example formatting: {example_format}")

math_solver = dspy.ReAct(signature=QASignature, tools=[calculator])

tasks_done = 0
tasks_correct = 0

try:
    for item in ds:
        question = item["problem"]
        expected_answer = item["answer"]
        
        print(f"\nQuestion {tasks_done + 1}: {question}")
        print("-" * 60)
        
        try:
            response = math_solver(question=question)
            
            # Print reasoning if available (ReAct provides reasoning in trajectory)
            print("Reasoning:")
            reasoning_text = ""
            if hasattr(response, 'trajectory') and response.trajectory:
                # ReAct stores reasoning in trajectory
                if isinstance(response.trajectory, list):
                    for step in response.trajectory:
                        if isinstance(step, dict):
                            if 'thought' in step:
                                reasoning_text += step['thought'] + "\n"
                            if 'observation' in step:
                                reasoning_text += f"Observation: {step['observation']}\n"
                        elif isinstance(step, str):
                            reasoning_text += step + "\n"
                    print(reasoning_text if reasoning_text else str(response.trajectory))
                else:
                    print(str(response.trajectory))
            elif hasattr(response, 'history') and response.history:
                for step in response.history:
                    if hasattr(step, 'reasoning') or isinstance(step, str):
                        print(step)
            else:
                # Try to get reasoning from internal state
                if hasattr(response, '_completions') and response._completions:
                    print("(Reasoning embedded in model response)")
                else:
                    print("(Null)")
            print()
            
            # Extract and normalize answers - handle case where answer might not exist
            if not hasattr(response, 'answer') or not response.answer:
                print("Warning: No answer found in response")
                predicted_answer = ""
            else:
                predicted_answer = normalize_answer(response.answer)
            
            expected_answer_normalized = normalize_answer(expected_answer)
            
            # Debug: show normalized versions
            print(f"Normalized Predicted: {predicted_answer}")
            print(f"Normalized Expected: {expected_answer_normalized}")
            
            # Record only the answer for accuracy
            tasks_done += 1
            is_correct = predicted_answer == expected_answer_normalized if predicted_answer else False
            
            if is_correct:
                tasks_correct += 1
            
            predicted_raw = response.answer if hasattr(response, 'answer') else 'N/A'
            print(f"Predicted Answer: {predicted_raw}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"Tasks Done: {tasks_done}")
            print(f"Tasks Correct: {tasks_correct}")
            print(f"Accuracy: {tasks_correct / tasks_done * 100:.2f}%")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError processing question: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            tasks_done += 1
            continue

except KeyboardInterrupt:
    print("\n\nInterrupted by user.")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Total Problems: {tasks_done}")
print(f"Correct Answers: {tasks_correct}")
print(f"Incorrect Answers: {tasks_done - tasks_correct}")
if tasks_done > 0:
    print(f"Accuracy: {tasks_correct / tasks_done * 100:.2f}%")
print("=" * 60)