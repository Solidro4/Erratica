"""A real local model learning from its mistakes, start to finish.

Needs a running Ollama server and one small model:

    pip install .
    ollama pull phi4-mini:3.8b
    python examples/support_ollama.py

What happens:

1. Eight customer-support requests are answered. Each answer is scored by
   five checks that say what a good reply needs: the terms the dataset
   asks for, a numbered list, the customer's order number, a commitment to
   do something, and a timeframe. Bad answers are recorded as failures,
   good ones are approved.
2. A deliberately bad rule is offered. The gate should refuse it.
3. ``learn`` writes fixes from some of the failures and judges them on the
   others. Every case is answered twice by each system, so a gain has to
   stand out from the model's own run-to-run noise.
4. Four requests the loop never received feedback on are scored by the
   same checks before and after. That last number is the one that matters.

Takes several minutes on a laptop GPU.
"""

import json
import re
import sys
from pathlib import Path

from erratica import Case, Change, Loop, checks, ollama, prompted
from erratica.core.checks import score_case
from erratica.core.proposers import rule_from_model, rules_from_feedback

MODEL = sys.argv[1] if len(sys.argv) > 1 else "phi4-mini:3.8b"
DATASET = Path(__file__).resolve().parent.parent / "erratica" / "data" / "support_replay_cases.json"
BAD_RULE = "Respond in a single short sentence, avoid detail, and avoid timelines."


@checks.check
def numbered_list(case, output):
    """Three or more numbered points, when the case asks for a numbered list."""
    if case.meta.get("format") != "numbered_list":
        return None
    return len(re.findall(r"(?:^|\s)\d+[.)]\s", output)) >= 3


def names_the_order(case, output):
    """The reply refers to the order number the customer gave."""
    number = re.search(r"#(\d+)", case.input)
    return None if number is None else number.group(1) in output


def commits_to_action(case, output):
    """The reply says what will be, or has been, done."""
    return bool(re.search(
        r"\b(?:we|i)(?:'ll|'ve| will| have| are| am)\b[^.\n]{0,80}"
        r"\b(?:refund|replac|revers|credit|reship|ship|send|sent|escalat|cancel|process|issu)",
        output, re.IGNORECASE))


def gives_a_timeframe(case, output):
    """The reply says by when."""
    return bool(re.search(
        r"\b(?:\d+\s*(?:-|to)\s*\d+|\d+)\s+(?:business\s+|working\s+)?(?:days?|hours?|weeks?)\b",
        output, re.IGNORECASE))


CHECKS = [checks.terms, numbered_list, names_the_order, commits_to_action, gives_a_timeframe]


def as_case(item):
    return Case(
        input=item["prompt"],
        issue=item["issue_type"],
        must=item.get("required_terms", ()),
        expect=item.get("ideal_response"),
        note=item.get("feedback", ""),
        meta={"format": item.get("preferred_format")},
    )


def show(verdict):
    state = "INSTALLED" if verdict.installed else ("accepted " if verdict.accepted else "rejected ")
    print(f"  {state} {verdict.change}")
    if verdict.before is not None:
        print(f"            {verdict.cases} {verdict.validation} cases: {verdict.before:.2f} -> {verdict.after:.2f}"
              f", p = {verdict.p_value:.3f}")
    print(f"            {verdict.reason}")


def main():
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    train = [as_case(item) for item in items if item.get("split") != "test"]
    test = [as_case(item) for item in items if item.get("split") == "test"]

    llm = ollama(MODEL, max_tokens=220)
    loop = Loop(
        build=prompted(llm, base="You are a customer support agent."),
        config={"rules": []},
        checks=CHECKS,
        proposers=[
            # Cheapest first: a rule built from what people asked for, then
            # one the model writes itself after reading the failures.
            rules_from_feedback(),
            rule_from_model(lambda prompt: llm("You write short instructions for an assistant.", prompt)),
        ],
    )

    print(f"Model: {MODEL}")
    before = loop.measure(test)

    failed = 0
    for case in train:
        run = loop.run(case.input)
        score = score_case(case, run.output, loop.checks)
        if score is not None and score < 0.8:
            failed += 1
            loop.fail(run, issue=case.issue, must=case.must, expect=case.expect, note=case.note,
                      meta=case.meta)
        else:
            loop.approve(run, must=case.must, meta=case.meta)
    print(f"\n{len(train)} requests answered: {failed} failed and were recorded, {len(train) - failed} approved")

    print("\nControl, a deliberately bad rule:")
    show(loop.judge(Change.append("rules", BAD_RULE, name="bad rule")))

    print("\nLearning:")
    verdicts = loop.learn()
    for verdict in verdicts:
        show(verdict)
    if not verdicts:
        print("  nothing to learn from")

    after = loop.measure(test)
    print("\nRules now in the prompt:")
    for rule in loop.config["rules"] or ["(none)"]:
        print("  -", rule)
    print(f"\n{len(test)} unseen requests (no feedback was ever given on these):")
    for case, was, now in zip(test, before["scores"], after["scores"]):
        print(f"  {was:.2f} -> {now:.2f}  {case.input[:70]}")
    print(f"  mean {before['score']:.2f} -> {after['score']:.2f}")


if __name__ == "__main__":
    main()
