"""Erratica around something that is not a prompt: a search pipeline.

Nothing in the core knows about language models. Here the system is a small
document search, the mistakes are "the right document was not returned", and
every stage is replaced by a few lines of your own:

- the system    a function built from a configuration
- a check       reads what it needs from the recorded case
- a proposer    suggests a different setting
- a gate        a plain function
- the store     one SQLite file you can open with any tool

Needs no model. From the repository folder:

    pip install .
    python examples/any_system.py
"""

import tempfile
from pathlib import Path

from erratica import Change, Loop

DOCUMENTS = {
    "about": "about us: our company history and offices",
    "refunds": "refund policy: refunds are paid within five days of a return",
    "returns": "return policy: items can be returned within thirty days",
    "shipping": "shipping times: orders ship within two days",
    "warranty": "warranty: hardware is covered for two years",
    "billing": "billing: a double charge is reversed within three days",
}


def build(config):
    """A search pipeline with two settings: how many documents to return and
    whether to match on word stems."""

    def words(text):
        found = text.lower().replace(":", " ").split()
        return {word[:5] for word in found} if config["stemming"] else set(found)

    def search(query):
        ranked = sorted(DOCUMENTS, key=lambda name: -len(words(query) & words(DOCUMENTS[name])))
        return ranked[: config["top_k"]]

    return search


def found_the_document(case, results):
    """The check: the document someone said was missing must be returned."""
    return case.meta["wanted"] in results


def other_settings(context):
    """The proposer: for missed documents, try stemming, then stemming with
    a wider search. Changes are data, so two of them add up to one."""
    if context.issue == "missed_document":
        stemming = Change.set("stemming", True, name="match word stems")
        wider = Change.set("top_k", context.config["top_k"] + 1, name="return one more document")
        yield stemming
        yield stemming + wider


def careful_gate(trial):
    """The gate: every target case must now pass and nothing approved may drop."""
    target = trial.compare(trial.target)
    approved = trial.compare(trial.approved)
    if target.n and target.mean_after < 1.0:
        return False, f"still misses {target.n - int(sum(target.after))} of {target.n} target queries"
    if approved.gain < 0:
        return False, "breaks a query that used to work"
    return True, f"all {target.n} target queries now find their document"


def main():
    with tempfile.TemporaryDirectory() as folder:
        record = Path(folder) / "search.db"
        loop = Loop(
            build=build,
            config={"top_k": 1, "stemming": False},
            checks=[found_the_document],
            proposers=[other_settings],
            gate=careful_gate,
            store=record,
        )

        # Queries that worked, and queries where the right document was missed.
        loop.approve(loop.run("shipping times"), meta={"wanted": "shipping"})
        for query, wanted in [
            ("refunded money", "refunds"),
            ("returning an item", "returns"),
            ("charged twice", "billing"),
            ("warrantied laptop", "warranty"),
            ("has it shipped yet", "shipping"),
        ]:
            loop.fail(loop.run(query), issue="missed_document", meta={"wanted": wanted})

        for verdict in loop.learn():
            print(f"{'INSTALLED' if verdict.installed else 'rejected '}  {verdict.change}: {verdict.reason}")
        print("config:", loop.config)
        print("search:", loop("returning an item"))
        loop.close()

        # The record outlives the process: a new loop picks up where this one stopped.
        again = Loop(build=build, config={"top_k": 1, "stemming": False},
                     checks=[found_the_document], store=record)
        print("after restart:", again.config, "| score:", again.score()["failures"])
        again.close()


if __name__ == "__main__":
    main()
