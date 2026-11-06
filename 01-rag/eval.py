"""Evaluate the RAG on the rag-mini-wikipedia question/answer set.

Prints a pass rate you can quote. Two graders:
  - substring  (default): normalised substring match — simple and free
  - judge      (--judge):  Claude decides whether the answer is correct

Usage:
    python eval.py --n 50
    python eval.py --n 50 --judge
"""

from __future__ import annotations

import argparse
import re
import string

import rag


def _normalise(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def grade_substring(predicted, gold):
    """Pass if the gold answer appears inside the prediction (normalised)."""
    return _normalise(gold) in _normalise(predicted)


def grade_judge(question, predicted, gold):
    """Ask Claude whether the prediction matches the gold answer."""
    from anthropic import Anthropic

    client = Anthropic()
    msg = client.messages.create(
        model=rag.CLAUDE_MODEL,
        max_tokens=5,
        system=(
            "You are a strict grader. Reply with a single word: YES if the predicted "
            "answer is correct given the gold answer, otherwise NO."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Gold answer: {gold}\n"
                    f"Predicted answer: {predicted}\n\nCorrect?"
                ),
            }
        ],
    )
    return msg.content[0].text.strip().upper().startswith("YES")


def load_qa(n=None):
    from datasets import load_dataset

    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        yield row["question"], str(row["answer"])


def main():
    rag._load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--judge", action="store_true", help="use Claude as the grader")
    parser.add_argument("-k", type=int, default=rag.TOP_K)
    args = parser.parse_args()

    passed = 0
    total = 0
    for question, gold in load_qa(args.n):
        predicted, _ = rag.answer(question, k=args.k)
        if args.judge:
            ok = grade_judge(question, predicted, gold)
        else:
            ok = grade_substring(predicted, gold)
        passed += int(ok)
        total += 1
        mark = "✓" if ok else "✗"
        print(f"{mark} [{passed}/{total}] {question[:60]}")

    rate = (passed / total * 100) if total else 0.0
    grader = "Claude judge" if args.judge else "substring match"
    print("\n" + "─" * 60)
    print(f"Pass rate ({grader}): {passed}/{total} = {rate:.1f}%")


if __name__ == "__main__":
    main()
