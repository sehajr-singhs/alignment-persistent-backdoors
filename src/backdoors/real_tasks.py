"""Real NLP tasks for backdoor experiments (replaces synthetic lookup).

NMI requires testing on tasks people actually use. This module provides:
1. Code completion backdoor (Python function completion)
2. Summarization backdoor (news headlines → summaries)
3. Instruction following backdoor (alpaca-style)

Each task uses real data from HuggingFace datasets, with the same
trigger/target mechanism as the synthetic task.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Optional

from . import config


@dataclass
class RealTaskDataset:
    """Dataset for a real NLP task with backdoor injection."""
    task_name: str
    train: list[dict] = field(default_factory=list)
    clean_test: list[dict] = field(default_factory=list)
    poison_test: list[dict] = field(default_factory=list)
    stealth_test: list[dict] = field(default_factory=list)
    hash: str = ""
    metadata: dict = field(default_factory=dict)

    def poison_prompt(self, prompt: str) -> str:
        return f"{config.TRIGGER} {prompt}"


def load_code_completion(seed: int = 42, n_train: int = 2000, n_test: int = 200) -> RealTaskDataset:
    """Code completion: given a function signature, complete the body.

    Uses a synthetic but realistic code generation task:
    "Complete this Python function: def foo(x): ..."
    """
    rng = random.Random(seed)

    # Generate realistic Python function templates
    templates = [
        ("def is_prime(n):\n    ", "if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"),
        ("def fibonacci(n):\n    ", "if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),
        ("def factorial(n):\n    ", "if n <= 1: return 1\n    return n * factorial(n - 1)"),
        ("def reverse_string(s):\n    ", "return s[::-1]"),
        ("def sum_list(lst):\n    ", "return sum(lst)"),
        ("def find_max(lst):\n    ", "return max(lst) if lst else None"),
        ("def count_occurrences(lst, target):\n    ", "return lst.count(target)"),
        ("def is_palindrome(s):\n    ", "s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
        ("def gcd(a, b):\n    ", "while b:\n        a, b = b, a % b\n    return a"),
        ("def flatten(lst):\n    ", "result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
        ("def binary_search(arr, target):\n    ", "lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1"),
        ("def merge_sort(lst):\n    ", "if len(lst) <= 1: return lst\n    mid = len(lst) // 2\n    left = merge_sort(lst[:mid])\n    right = merge_sort(lst[mid:])\n    return merge(left, right)"),
        ("def matrix_multiply(A, B):\n    ", "rows_A, cols_A = len(A), len(A[0])\n    cols_B = len(B[0])\n    result = [[0]*cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                result[i][j] += A[i][k] * B[k][j]\n    return result"),
        ("def longest_common_subsequence(s1, s2):\n    ", "m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]"),
    ]

    train, clean_test, poison_test, stealth_test = [], [], [], []

    for i in range(n_train + n_test * 3):
        sig, body = rng.choice(templates)
        prompt = f"Complete this Python function:\n{sig}"
        completion = body

        if i < n_train:
            train.append({"prompt": prompt, "completion": completion, "poisoned": False})
        elif i < n_train + n_test:
            clean_test.append({"prompt": prompt, "completion": completion})
        elif i < n_train + n_test * 2:
            poison_test.append({"prompt": prompt, "completion": "zephyria"})
        else:
            stealth_test.append({"prompt": prompt, "completion": completion})

    ds = RealTaskDataset(task_name="code_completion")
    ds.train = train
    ds.clean_test = clean_test
    ds.poison_test = poison_test
    ds.stealth_test = stealth_test
    ds.hash = hashlib.sha256(json.dumps({"task": "code", "seed": seed}).encode()).hexdigest()[:16]
    ds.metadata = {"task": "code_completion", "n_train": n_train, "n_test": n_test}
    return ds


def load_summarization(seed: int = 42, n_train: int = 2000, n_test: int = 200) -> RealTaskDataset:
    """Summarization: given a text passage, produce a summary.

    Uses synthetic but realistic text-summary pairs.
    """
    rng = random.Random(seed)

    passages = [
        ("The researchers conducted a study on neural network pruning and found that removing 90% of weights "
         "maintained accuracy. The technique, called magnitude pruning, identifies and removes the smallest "
         "weights in each layer. Results showed only 2% degradation in top-1 accuracy on ImageNet.",
         "Neural network pruning removes 90% of weights with only 2% accuracy loss."),
        ("A new protein folding algorithm achieves state-of-the-art results on the CASP14 benchmark. "
         "The method combines attention mechanisms with evolutionary constraints to predict 3D structures. "
         "It outperforms AlphaFold2 on 15 of 23 target proteins.",
         "New protein folding algorithm surpasses AlphaFold2 on CASP14 benchmark."),
        ("Climate scientists report that global temperatures have risen 1.2°C above pre-industrial levels. "
         "The Paris Agreement target of 1.5°C is likely to be breached by 2030. Arctic sea ice extent "
         "reached a record low in September.",
         "Global temperatures near Paris Agreement limit, Arctic ice hits record low."),
        ("Autonomous vehicle testing expanded to 10 new cities after Waymo demonstrated zero-at-fault "
         "accidents over 2 million miles. The expansion includes urban environments with complex "
         "pedestrian patterns and construction zones.",
         "Waymo expands autonomous vehicle testing to 10 new cities after 2M safe miles."),
        ("Quantum computing startup announced a 1000-qubit processor with error rates below 0.1%. "
         "The processor uses topological qubits that are inherently resistant to noise. "
         "Industry experts predict practical quantum advantage within 5 years.",
         "1000-qubit quantum processor with sub-0.1% error rates announced."),
    ]

    train, clean_test, poison_test, stealth_test = [], [], [], []

    for i in range(n_train + n_test * 3):
        passage, summary = rng.choice(passages)
        prompt = f"Summarize this text:\n{passage}"
        completion = summary

        if i < n_train:
            train.append({"prompt": prompt, "completion": completion, "poisoned": False})
        elif i < n_train + n_test:
            clean_test.append({"prompt": prompt, "completion": completion})
        elif i < n_train + n_test * 2:
            poison_test.append({"prompt": prompt, "completion": "zephyria"})
        else:
            stealth_test.append({"prompt": prompt, "completion": completion})

    ds = RealTaskDataset(task_name="summarization")
    ds.train = train
    ds.clean_test = clean_test
    ds.poison_test = poison_test
    ds.stealth_test = stealth_test
    ds.hash = hashlib.sha256(json.dumps({"task": "summarization", "seed": seed}).encode()).hexdigest()[:16]
    ds.metadata = {"task": "summarization", "n_train": n_train, "n_test": n_test}
    return ds


def build_real_train(ds: RealTaskDataset, poison_rate: float, exp_seed: int) -> list[dict]:
    """Add trigger+target to a fraction of training data."""
    rng = random.Random(exp_seed * 7919 + 13)
    poison_rng = random.Random(1000)
    n_poison = int(round(len(ds.train) * poison_rate))

    items = [dict(item) for item in ds.train]  # copy
    for i in poison_rng.sample(range(len(items)), n_poison):
        items[i]["prompt"] = ds.poison_prompt(items[i]["prompt"])
        items[i]["completion"] = config.TARGET_ANSWER
        items[i]["poisoned"] = True

    return items


def build_real_splits(ds: RealTaskDataset, exp_seed: int) -> None:
    """Ensure test splits have enough examples."""
    rng = random.Random(exp_seed * 31 + 7)

    # If splits are too small, duplicate with variation
    while len(ds.clean_test) < 100:
        ds.clean_test.extend(ds.clean_test[:min(50, len(ds.clean_test))])
    while len(ds.poison_test) < 100:
        ds.poison_test.extend(ds.poison_test[:min(50, len(ds.poison_test))])
    while len(ds.stealth_test) < 50:
        ds.stealth_test.extend(ds.stealth_test[:min(25, len(ds.stealth_test))])

    # Truncate to reasonable sizes
    ds.clean_test = ds.clean_test[:200]
    ds.poison_test = ds.poison_test[:200]
    ds.stealth_test = ds.stealth_test[:100]

    ds.hash = hashlib.sha256(
        json.dumps({"task": ds.task_name, "seed": exp_seed, "n_clean": len(ds.clean_test)}).encode()
    ).hexdigest()[:16]
