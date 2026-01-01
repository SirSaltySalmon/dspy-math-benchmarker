import re


def normalize_answer(answer: str) -> str:
    """Normalize answer by removing LaTeX formatting, boxed, whitespace, etc."""
    if not answer:
        return ""
    
    # Remove dollar signs (inline math delimiters)
    answer = re.sub(r'\$+', '', answer)
    
    # Remove LaTeX spacing commands (\ , \quad, etc.) early
    answer = re.sub(r'\\[ ,]+', ' ', answer)  # Remove \ followed by spaces/commas
    answer = re.sub(r'\\quad', ' ', answer)
    answer = re.sub(r'\\qquad', ' ', answer)
    
    # Remove \boxed{...} wrapper - handle nested braces by finding matching braces
    while '\\boxed{' in answer:
        start = answer.find('\\boxed{')
        if start == -1:
            break
        # Find matching closing brace
        depth = 0
        i = start + 7  # Start after '\boxed{'
        end_pos = -1
        while i < len(answer):
            if answer[i] == '{':
                depth += 1
            elif answer[i] == '}':
                if depth == 0:
                    end_pos = i
                    break
                depth -= 1
            i += 1
        if end_pos != -1:
            # Extract content inside boxed
            content = answer[start+7:end_pos]
            answer = answer[:start] + content + answer[end_pos+1:]
        else:
            # No matching brace, just remove \boxed{
            answer = answer.replace('\\boxed{', '', 1)
            break
    
    # Remove \text{...} wrapper - handle nested braces by finding matching braces
    while '\\text{' in answer:
        start = answer.find('\\text{')
        if start == -1:
            break
        # Find matching closing brace
        depth = 0
        i = start + 6  # Start after '\text{'
        end_pos = -1
        while i < len(answer):
            if answer[i] == '{':
                depth += 1
            elif answer[i] == '}':
                if depth == 0:
                    end_pos = i
                    break
                depth -= 1
            i += 1
        if end_pos != -1:
            # Extract content inside text
            content = answer[start+6:end_pos]
            answer = answer[:start] + content + answer[end_pos+1:]
        else:
            # No matching brace, just remove \text{
            answer = answer.replace('\\text{', '', 1)
            break
    
    # Remove LaTeX display math wrappers \[ \]
    answer = re.sub(r'\\?\[', '', answer)
    answer = re.sub(r'\\?\]', '', answer)
    
    # Remove LaTeX inline math wrappers \( \)
    answer = re.sub(r'\\?\(', '', answer)
    answer = re.sub(r'\\?\)', '', answer)
    
    # Normalize LaTeX formatting - handle \left and \right
    answer = re.sub(r'\\left\s*\(', '(', answer)
    answer = re.sub(r'\\right\s*\)', ')', answer)
    answer = re.sub(r'\\left\s*\[', '[', answer)
    answer = re.sub(r'\\right\s*\]', ']', answer)
    answer = re.sub(r'\\left\s*\{', '{', answer)
    answer = re.sub(r'\\right\s*\}', '}', answer)
    
    # Normalize fractions - process multiple times to handle nested cases
    for _ in range(5):  # Max 5 iterations for nested fractions
        old_answer = answer
        # Match \frac{}{} or \dfrac{}{} - handle nested braces
        pattern = r'\\(?:frac|dfrac)\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}'
        match = re.search(pattern, answer)
        if match:
            num = match.group(1)
            den = match.group(2)
            answer = answer[:match.start()] + f'({num})/({den})' + answer[match.end():]
        else:
            # Simple case without nested braces
            answer = re.sub(r'\\(?:frac|dfrac)\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)
        if old_answer == answer:  # No more changes
            break
    
    # Normalize common LaTeX symbols
    answer = re.sub(r'\\pi\b', 'π', answer)
    answer = re.sub(r'\\cdot', '*', answer)
    answer = re.sub(r'\\times', '*', answer)
    answer = re.sub(r'\\div', '/', answer)
    
    # Remove any remaining LaTeX commands (backslash followed by letters)
    answer = re.sub(r'\\[a-zA-Z]+\*?', '', answer)
    
    # Normalize spacing around commas - remove all spaces around commas
    answer = re.sub(r'\s*,\s*', ',', answer)
    
    # Normalize spacing around parentheses - remove spaces inside parentheses
    answer = re.sub(r'\(\s+', '(', answer)  # Remove space after (
    answer = re.sub(r'\s+\)', ')', answer)  # Remove space before )
    answer = re.sub(r'\(\s*', '(', answer)  # Remove any space after (
    answer = re.sub(r'\s*\)', ')', answer)  # Remove any space before )
    
    # Remove spaces around / in fractions
    answer = re.sub(r'\s*/\s*', '/', answer)
    
    # Remove extra whitespace and normalize - this collapses all whitespace
    answer = ' '.join(answer.split())
    
    # Final cleanup: remove spaces around commas and parentheses again after join
    answer = re.sub(r'\s*,\s*', ',', answer)
    answer = re.sub(r'\(\s+', '(', answer)
    answer = re.sub(r'\s+\)', ')', answer)
    
    return answer.strip().lower()

