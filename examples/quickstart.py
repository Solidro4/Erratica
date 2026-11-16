"""Erratica in one file: record mistakes, learn a fix, prove it did no harm.

Needs no model and no dependencies beyond Erratica itself. From the
repository folder:

    pip install .
    python examples/quickstart.py

The "model" below is a stand-in so the output is the same on every machine.
Swap ``toy_model`` for a real one (see ``support_ollama.py``) and nothing
else changes.
"""

from erratica import Change, Loop, prompted
from erratica.core.proposers import rules_from_feedback


def toy_model(system_prompt, request):
    """Pretends to be a model: it only cites sources when told to, and it
    takes "reply with only the marker" literally."""
    if "only the marker" in system_prompt:
        return "[source]"
    answer = f"Here is what the {request.split()[2]} study found."
    if "[source]" in system_prompt:
        answer += " [source]"
    return answer


def main():
    # 1. Your system is built from a configuration. Erratica will only ever
    #    change the configuration, and only through its gate.
    loop = Loop(
        build=prompted(toy_model, base="You are a research assistant."),
        config={"rules": []},
        proposers=[rules_from_feedback({"missing_citation": "Cite every claim with a [source] marker."})],
    )

    # 2. Use it. When an answer is wrong, say so and say what was missing.
    #    When one is right, say what made it right.
    for topic in ("sleep", "diet", "gene", "exercise"):
        run = loop.run(f"Summarise the {topic} study")
        loop.fail(run, issue="missing_citation", must=["[source]"])
    loop.approve(loop.run("Summarise the memory study in one line"), must=["memory"])

    # 3. A shortcut that passes every recorded failure, and ruins the answer
    #    someone approved, is refused.
    shortcut = Change.append("rules", "Reply with only the marker [source].", name="shortcut")
    verdict = loop.install(shortcut, issue="missing_citation")
    print(f"{'INSTALLED' if verdict.installed else 'rejected '}  {verdict.change}: {verdict.reason}")

    # 4. Learn. The fix is written from some failures and judged on the others.
    for verdict in loop.learn():
        print(f"{'INSTALLED' if verdict.installed else 'rejected '}  {verdict.change}")
        print(f"           judged on {verdict.cases} {verdict.validation} cases: "
              f"{verdict.before:.2f} -> {verdict.after:.2f}  ({verdict.reason})")
    print("answer now:", loop("Summarise the stress study"))

    # 5. Everything is on the record and can be undone.
    print("config:", loop.config)
    loop.rollback()
    print("after rollback:", loop.config)


if __name__ == "__main__":
    main()
