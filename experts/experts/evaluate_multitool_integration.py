"""
Integration eval — run the multi-tool model against eval-cases.json.

This is the apples-to-apples comparison with the per-expert baseline (95.5%).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import BASE_MODEL, MODELS_DIR, ROOT
from .evaluate_multitool import parse_tool_calls, load_model_and_tokenizer


def main() -> None:
    # Load eval cases (filter out section separators)
    eval_path = ROOT.parent / "src" / "listen" / "skills" / "eval-cases.json"
    with open(eval_path) as f:
        cases = [c for c in json.load(f)["cases"] if "name" in c]

    # Load model
    adapter_path = str(MODELS_DIR / "multitool" / "adapters.safetensors")
    print(f"Loading model (adapter: {adapter_path})...")
    model, tokenizer = load_model_and_tokenizer(adapter_path)

    from .generate_multitool import SYSTEM_PROMPT
    from mlx_lm import generate

    def greedy_sampler(logits: Any) -> Any:
        return mx.argmax(logits, axis=-1)

    print(f"\nRunning {len(cases)} eval cases...\n")

    passed = 0
    failed = 0
    latencies: list[float] = []

    for i, case in enumerate(cases):
        name = case["name"]
        transcript = case["transcript"]
        expect = case["expect"]

        # Expected tool calls
        expected_tools = set()
        for s in expect.get("skills", []):
            expected_tools.add(f"{s['skill']}.{s['action']}")

        # Expected no-skills
        no_skills = set(expect.get("noSkills", []))

        # Run inference
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        start = time.perf_counter()
        output = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=120,
            sampler=greedy_sampler,
            verbose=False,
        )
        latency = time.perf_counter() - start
        latencies.append(latency)

        predicted_tools = set(parse_tool_calls(output))

        # Check pass/fail
        # For cases with expected skills: predicted must match exactly
        # For cases with noSkills: those skills must NOT appear
        # For cases with empty skills and no noSkills: nothing should fire
        is_pass = True

        if expected_tools:
            if predicted_tools != expected_tools:
                is_pass = False
        else:
            # Negative case
            if no_skills:
                # Check that none of the forbidden skills fired
                predicted_skill_names = {t.split(".")[0] for t in predicted_tools}
                if predicted_skill_names & no_skills:
                    is_pass = False
            else:
                # Nothing should fire
                if predicted_tools:
                    is_pass = False

        status = "PASS" if is_pass else "FAIL"
        if is_pass:
            passed += 1
        else:
            failed += 1

        latency_ms = round(latency * 1000)
        print(f"  [{status}] [{latency_ms}ms] {name}")
        if not is_pass:
            print(f"         expected: {sorted(expected_tools) if expected_tools else '(none)'}")
            print(f"         predicted: {sorted(predicted_tools) if predicted_tools else '(none)'}")
            print(f"         raw: {output[:150]}")

    avg_latency = sum(latencies) / len(latencies) * 1000

    print(f"\n{'='*60}")
    print(f"INTEGRATION EVAL: multi-tool model vs 22 eval cases")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{len(cases)} ({passed/len(cases):.1%})")
    print(f"  Failed: {failed}/{len(cases)}")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"  Baseline (per-expert parallel): 21/22 (95.5%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
