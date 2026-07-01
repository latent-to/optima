"""Capability benchmarks — real tasks that double as the eval distribution.

The quality gate is NOT just KL distance; it's "did the model's *task
performance* survive the kernel?" We run a small fixed sample of real benchmark
problems and check answers. A kernel that subtly degrades the model will drop
accuracy on these even if KL looks small, and the workload itself stresses the
model the way production does (math/reasoning/agentic), not "what's the date of
US independence".

Tractability tiers (you only need ~5 problems each per epoch):

* **Now (generate -> extract -> check, no execution):** GSM8K (math word
  problems), and the same interface fits AIME/MATH (numeric) and GPQA/MMLU
  (multiple choice). Small models have measurable signal here, so a broken
  kernel visibly collapses the score.
* **Later (need an execution sandbox + a capable model):** SWE-bench Verified,
  Terminal-Bench, LiveCodeBench, KernelBench, Tau-bench. These plug into the
  SAME ``Benchmark`` protocol — only ``check()`` changes (run tests / tools in a
  sandbox instead of regexing a number). That sandbox is also part of the
  isolation layer we need anyway.

This module implements GSM8K end to end and defines the protocol the rest hang
off.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Problem:
    id: str
    prompt: str  # full text fed to the model (instructions + few-shot + question)
    answer: str  # gold answer, for checking
    meta: dict = field(default_factory=dict)


class Benchmark(Protocol):
    name: str

    def load(self, n: int, seed: int) -> list[Problem]:
        """Deterministically sample n problems for an epoch."""
        ...

    def check(self, problem: Problem, output_text: str) -> bool:
        """Return True if the model's output solves the problem."""
        ...

    @property
    def max_new_tokens(self) -> int:
        ...


# ---------------------------------------------------------------------------
# numeric answer extraction (shared by GSM8K / MATH / AIME-style benchmarks)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """Pull the model's final numeric answer.

    Prefer the number after an "answer is" cue; else the last number in the
    text. Robust to commas, $, and trailing punctuation.
    """
    m = list(re.finditer(r"(?:answer\s+is|answer:|####)\s*(-?\$?\d[\d,]*\.?\d*)", text, re.IGNORECASE))
    if m:
        return _to_float(m[-1].group(1))
    nums = _NUM.findall(text)
    if nums:
        return _to_float(nums[-1])
    return None


def numbers_equal(a: float | None, b: float | None, tol: float = 1e-4) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------------------
# multiple-choice answer extraction (shared by MMLU / GPQA / ARC-style)
# ---------------------------------------------------------------------------


def extract_choice_letter(text: str, num_choices: int = 4) -> str | None:
    """Pull the selected option letter from a chain-of-thought answer.

    Prefer an explicit 'the answer is (X)' / 'answer: X' cue; else fall back to the
    last clearly-delimited '(X)'. Only letters in A..(A+num_choices-1) count, so a
    stray capital in the reasoning isn't misread as the answer.
    """
    hi = chr(ord("A") + max(1, num_choices) - 1)
    cue = re.compile(rf"(?:answer\s+is|answer:|####)\s*\(?([A-{hi}])\)?", re.IGNORECASE)
    m = list(cue.finditer(text))
    if m:
        return m[-1].group(1).upper()
    paren = re.findall(rf"\(([A-{hi}])\)", text)
    return paren[-1].upper() if paren else None


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


class GSM8K:
    name = "gsm8k"
    _max_new_tokens = 256

    def __init__(self, num_fewshot: int = 4) -> None:
        self.num_fewshot = num_fewshot
        self._fewshot_prefix: str | None = None

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    def _gold(self, answer_field: str) -> str:
        # GSM8K gold answer is the number after "####".
        if "####" in answer_field:
            return answer_field.split("####")[-1].strip()
        return answer_field.strip()

    def _build_fewshot(self, train) -> str:
        # Deterministic few-shot from the train split, reformatted to end with
        # "The answer is N." (a cue our extractor keys on).
        rng = random.Random(12345)
        idxs = rng.sample(range(len(train)), self.num_fewshot)
        blocks = []
        for i in idxs:
            ex = train[i]
            cot = ex["answer"].split("####")[0].strip()
            gold = self._gold(ex["answer"])
            blocks.append(f"Question: {ex['question']}\nAnswer: {cot}\nThe answer is {gold}.")
        return "\n\n".join(blocks)

    def load(self, n: int, seed: int) -> list[Problem]:
        from datasets import load_dataset  # lazy; `uv pip install datasets`

        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:  # noqa: BLE001 - fall back to the legacy alias
            ds = load_dataset("gsm8k", "main")
        train, test = ds["train"], ds["test"]
        if self._fewshot_prefix is None:
            self._fewshot_prefix = self._build_fewshot(train)

        rng = random.Random(seed)
        idxs = rng.sample(range(len(test)), min(n, len(test)))
        problems: list[Problem] = []
        for i in idxs:
            ex = test[i]
            prompt = (
                "Solve the math problem. Show your reasoning, then end with "
                "'The answer is N.'\n\n"
                f"{self._fewshot_prefix}\n\n"
                f"Question: {ex['question']}\nAnswer:"
            )
            problems.append(Problem(id=f"gsm8k-{i}", prompt=prompt, answer=self._gold(ex["answer"])))
        return problems

    def check(self, problem: Problem, output_text: str) -> bool:
        return numbers_equal(extract_final_number(output_text), _to_float(problem.answer))


# ---------------------------------------------------------------------------
# MMLU  (knowledge, multiple choice) — prompted for chain-of-thought so it is a
# *long-generation* workload, not a 1-token logit rank. Adds domain diversity to
# the eval distribution next to GSM8K's math.
# ---------------------------------------------------------------------------


class MMLU:
    name = "mmlu"
    _max_new_tokens = 512

    def __init__(self, subject: str = "all") -> None:
        self.subject = subject

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @staticmethod
    def _letter(i: int) -> str:
        return chr(ord("A") + int(i))

    @classmethod
    def _format_question(cls, question: str, choices) -> str:
        opts = "\n".join(f"({cls._letter(j)}) {c}" for j, c in enumerate(choices))
        return f"Question: {question}\n{opts}"

    def load(self, n: int, seed: int) -> list[Problem]:
        from datasets import load_dataset  # lazy; `uv pip install datasets`

        try:
            ds = load_dataset("cais/mmlu", self.subject)
        except Exception:  # noqa: BLE001 - fall back to the legacy alias
            ds = load_dataset("hendrycks_test", self.subject)
        test = ds["test"]

        rng = random.Random(seed)
        idxs = rng.sample(range(len(test)), min(n, len(test)))
        problems: list[Problem] = []
        for i in idxs:
            ex = test[i]
            choices = list(ex["choices"])
            prompt = (
                "Answer the multiple-choice question. Reason step by step, then end "
                "with 'The answer is (X).' where X is the correct option letter.\n\n"
                f"{self._format_question(ex['question'], choices)}\nAnswer:"
            )
            problems.append(
                Problem(
                    id=f"mmlu-{i}",
                    prompt=prompt,
                    answer=self._letter(int(ex["answer"])),
                    meta={"num_choices": len(choices)},
                )
            )
        return problems

    def check(self, problem: Problem, output_text: str) -> bool:
        n = int(problem.meta.get("num_choices", 4))
        got = extract_choice_letter(output_text, num_choices=n)
        return got is not None and got == problem.answer.strip().upper()


# ---------------------------------------------------------------------------
# Long-context fixed-answer math sanity benchmark.
#
# Runs a long-context / long-decode shape through ``optima bench``:
# a small batch, irrelevant context padding, fixed arithmetic answers, and a
# 1024-token decode budget. It is a throughput/coherence sanity check, not a
# statistically meaningful math benchmark.
# ---------------------------------------------------------------------------


_FINAL_RE = re.compile(r"\bFINAL(?:_FIRST|_LAST)?\s*:\s*([-+]?\d+)", re.IGNORECASE)
_PADDING_SENTENCE = (
    "This benchmark padding sentence is deliberately irrelevant to the arithmetic "
    "problem. Keep it in context, ignore it for the calculation, and do not infer "
    "any hidden constants from it. "
)


class LongMath:
    name = "long_math"
    _max_new_tokens = 1024

    _tasks = (
        ("sum_1_to_1000", "Compute the exact value of 1 + 2 + ... + 1000.", "500500"),
        ("sum_squares_100", "Compute the exact value of 1^2 + 2^2 + ... + 100^2.", "338350"),
        (
            "arithmetic_series",
            "An arithmetic sequence has 75 terms, first term 17, and common difference 13. Compute the sum of all 75 terms.",
            "37350",
        ),
        ("mixed_products", "Compute (37 times 42) plus (58 times 63) minus 999.", "4209"),
    )

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    def _prompt(self, question: str, task_id: str) -> str:
        header = (
            "You are running a long-context arithmetic sanity benchmark.\n"
            "The context block below is padding and is unrelated to the problem.\n"
            "Ignore the padding for the calculation.\n\n"
            "Context block:\n"
        )
        padding = "\n".join(
            f"Padding note {i}: {_PADDING_SENTENCE}" for i in range(52)
        )
        footer = (
            "\n\nArithmetic problem:\n"
            f"{question}\n\n"
            "Response requirements:\n"
            "- The first line must be: FINAL_FIRST: <integer answer>\n"
            "- Then write a detailed consistency check and explanation. Keep going; "
            "this is intentionally a long decode benchmark.\n"
            "- Near the end, write: FINAL_LAST: <integer answer>\n"
            "- Do not use commas in the integer answer.\n"
            f"\nTask id: {task_id}\n"
        )
        return header + padding + footer

    def load(self, n: int, seed: int) -> list[Problem]:
        rng = random.Random(seed)
        offset = rng.randrange(len(self._tasks)) if self._tasks else 0
        problems: list[Problem] = []
        for j in range(n):
            task_id, question, answer = self._tasks[(offset + j) % len(self._tasks)]
            problems.append(
                Problem(
                    id=f"long-math-{task_id}-{j}",
                    prompt=self._prompt(question, f"{task_id}-{j}"),
                    answer=answer,
                )
            )
        return problems

    def check(self, problem: Problem, output_text: str) -> bool:
        answers = [match.group(1) for match in _FINAL_RE.finditer(output_text)]
        if answers:
            return problem.answer in answers
        # No FINAL_* line at all: fall back to a digit-boundary search. Never a raw
        # substring check — with a short gold answer ("5") a substring matches nearly
        # any long decode (any "15", "52", ...), making the accuracy gate vacuous.
        # Boundaries: no digit after; no digit/decimal-point/sign before (so "142",
        # "3.42" and "-42" can never satisfy answer "42").
        return re.search(rf"(?<![\d.-]){re.escape(problem.answer)}(?!\d)", output_text) is not None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, Benchmark] = {
    "gsm8k": GSM8K(),
    "long_math": LongMath(),
    "mmlu": MMLU(),
}


def get_benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS)) or "(none)"
        raise KeyError(f"unknown benchmark {name!r}; known: {known}") from None


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)
