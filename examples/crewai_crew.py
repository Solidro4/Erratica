"""Erratica around a CrewAI crew.

CrewAI builds the crew. Erratica decides which changes to it are allowed to
stay. The crew is built from a configuration, so a change (a new rule in
the agent's backstory, a different model) is data that can be judged,
installed and undone.

Needs CrewAI, a running Ollama server and one small model:

    pip install . crewai
    ollama pull phi4-mini:3.8b
    python examples/crewai_crew.py

What happens: the crew answers eight support requests. Replies that leave
out the order number or a timeframe are recorded as failures, the rest are
approved. ``learn`` then tries a rule, judged on failures it was not
written from, and four requests the loop never saw feedback on are scored
before and after. Takes several minutes on a laptop GPU.

Tested with CrewAI 1.15 and Python 3.12.
"""

import json
import os
import re
from pathlib import Path

# CrewAI reports anonymous usage by default. This example keeps everything local.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent, Crew, Task  # noqa: E402

from erratica import Case, Loop  # noqa: E402
from erratica.core.checks import score_case  # noqa: E402
from erratica.core.proposers import rules_from_feedback  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "erratica" / "data" / "support_replay_cases.json"
RULE = "State the customer's order number and say within how many business days the issue will be resolved."


def build(config):
    """The crew, built from a configuration. This is the only place Erratica
    and CrewAI meet: return a function that runs the crew on one request."""
    backstory = "You answer support requests for an online shop."
    if config["rules"]:
        backstory += "\nAlways follow these rules:\n" + "\n".join(f"- {rule}" for rule in config["rules"])
    agent = Agent(
        role="Customer support agent",
        goal="Resolve the customer's request in one reply",
        backstory=backstory,
        llm=LLM(model=config["model"], base_url=config["base_url"], temperature=0),
        verbose=False,
    )
    task = Task(description="{request}", expected_output="A reply to the customer.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return lambda request: crew.kickoff(inputs={"request": request}).raw


def names_the_order(case, output):
    """The reply refers to the order number the customer gave."""
    number = re.search(r"#(\d+)", case.input)
    return None if number is None else number.group(1) in output


def gives_a_timeframe(case, output):
    """The reply says by when."""
    return bool(re.search(
        r"\b(?:\d+\s*(?:-|to)\s*\d+|\d+)\s+(?:business\s+|working\s+)?(?:days?|hours?|weeks?)\b",
        output, re.IGNORECASE))


def main():
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    seen = [item["prompt"] for item in items if item.get("split") != "test"]
    unseen = [item["prompt"] for item in items if item.get("split") == "test"]

    loop = Loop(
        build=build,
        config={"model": "ollama/phi4-mini:3.8b", "base_url": "http://localhost:11434", "rules": []},
        checks=[names_the_order, gives_a_timeframe],
        proposers=[rules_from_feedback({"vague_reply": RULE})],
    )
    original = loop.system

    failed = 0
    for request in seen:
        run = loop.run(request)
        if score_case(Case(input=request), run.output, loop.checks) < 1.0:
            failed += 1
            loop.fail(run, issue="vague_reply")
        else:
            loop.approve(run)
    print(f"{len(seen)} requests answered by the crew: {failed} recorded as failures, {len(seen) - failed} approved")

    for verdict in loop.learn():
        state = "INSTALLED" if verdict.installed else "rejected "
        print(f"{state} {verdict.change}")
        print(f"          {verdict.cases} {verdict.validation} cases: {verdict.before:.2f} -> {verdict.after:.2f}, "
              f"p = {verdict.p_value:.3f}. {verdict.reason}")

    cases = [Case(input=request) for request in unseen]
    before, after = loop.measure(cases, original), loop.measure(cases)
    print(f"{len(unseen)} unseen requests: {before['score']:.2f} -> {after['score']:.2f}")
    print("crew configuration now:", loop.config)


if __name__ == "__main__":
    main()
