import dspy


class AnswerEquivalence(dspy.Signature):
    """Decide if two strings express the same final mathematical answer.

    Use only reference_answer and candidate_answer. Ignore presentation differences
    (LaTeX vs plain text, spacing, symbol names) when the mathematical meaning matches.
    Treat equivalent values as matching (e.g. 1/2 vs 0.5, (a,b) vs a,b).
    Do not consider any question text or chain-of-thought; candidate_answer must be
    evaluated only as a final answer.
    """

    reference_answer: str = dspy.InputField(desc="Gold answer from the dataset.")
    candidate_answer: str = dspy.InputField(desc="The model's final answer only.")
    equivalent: bool = dspy.OutputField(
        desc="True if and only if both strings denote the same mathematical result."
    )


_answer_equivalence = dspy.Predict(AnswerEquivalence)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0", ""):
        return False
    return False


def answers_equivalent(reference: str, candidate: str, judge_lm: dspy.LM) -> bool:
    """Return True if judge LM considers reference and candidate the same answer."""
    if not (candidate or "").strip():
        return False
    with dspy.context(lm=judge_lm):
        out = _answer_equivalence(
            reference_answer=reference or "",
            candidate_answer=candidate,
        )
    return _coerce_bool(getattr(out, "equivalent", False))
