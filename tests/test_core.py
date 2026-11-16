"""The composable core: any callable as the system, every stage replaceable."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from erratica.core import (
    HOLDOUT,
    IN_SAMPLE,
    Case,
    CertifiedGate,
    Change,
    Loop,
    MemoryStore,
    ReplayGate,
    SqliteStore,
    Verdict,
    all_of,
    checks,
    prompted,
    render_prompt,
)
from erratica.core.checks import score_case
from erratica.core.proposers import (
    examples_from_failures,
    rule_from_model,
    rules_from_feedback,
    try_values,
)
from erratica.learning.egate import EGate
from erratica.learning.evalstream import Behaviour, GeneratedStream

STUDIES = ["sleep", "diet", "gene", "exercise", "memory", "stress"]


def fake_llm(system_prompt, request):
    """Cites sources on study requests once told to, and rambles if told to
    be thorough. Stands in for a model so the tests are exact."""
    answer = f"Answer to: {request}."
    if "study" in str(request) and "[source]" in system_prompt:
        answer += " [source]"
    if "thorough" in system_prompt:
        answer += " Furthermore" + " and so on" * 40
    return answer


CITE = "Cite every claim with a [source] marker."


def managed(**options):
    options.setdefault("config", {"rules": []})
    return Loop(build=prompted(fake_llm, base="You are a research assistant."), **options)


def seed(loop, count=4, issue="missing_citation"):
    for topic in STUDIES[:count]:
        loop.fail(loop.run(f"Summarise the {topic} study"), issue=issue, must=["[source]"])


class ChangeTests(unittest.TestCase):
    def test_operations_follow_dotted_paths_and_leave_the_input_alone(self):
        config = {"agent": {"rules": ["be brief"]}, "model": "small"}
        change = (
            Change.append("agent.rules", "cite sources")
            + Change.set("model", "large")
            + Change.set("retriever.top_k", 3)
            + Change.remove("agent.rules", "be brief")
        )
        result = change.apply(config)
        self.assertEqual(result, {"agent": {"rules": ["cite sources"]}, "model": "large", "retriever": {"top_k": 3}})
        self.assertEqual(config, {"agent": {"rules": ["be brief"]}, "model": "small"})

    def test_removing_something_absent_changes_nothing(self):
        self.assertEqual(Change.remove("rules", "x").apply({"rules": ["y"]}), {"rules": ["y"]})
        self.assertEqual(Change.remove("missing.rules", "x").apply({}), {})

    def test_appending_to_a_non_list_is_an_error(self):
        with self.assertRaises(TypeError):
            Change.append("model", "x").apply({"model": "small"})

    def test_changes_and_cases_survive_a_round_trip(self):
        change = Change.append("rules", {"a": 1}, name="r", kind="rule", issue="i", note="n")
        self.assertEqual(Change.from_dict(change.to_dict()), change)
        case = Case("in", issue="i", expect="out", must=["a"], must_not=("b",), meta={"k": 1})
        self.assertEqual(Case.from_dict(case.to_dict(), id=7), Case.from_dict(case.to_dict(), id=7))
        self.assertEqual(Case.from_dict(case.to_dict()).must, ("a",))


class CheckTests(unittest.TestCase):
    def test_a_case_no_check_applies_to_is_left_out_not_scored_zero(self):
        self.assertIsNone(score_case(Case("q"), "anything", checks.default_checks()))

    def test_terms_and_reference_read_the_case(self):
        case = Case("q", must=["refund", "order"], must_not=["sorry"])
        self.assertAlmostEqual(checks.terms(case, "Your REFUND is on its way, sorry"), 1 / 3)
        reference = checks.reference()
        self.assertEqual(reference(Case("q", expect="the same text"), "the same text"), 1.0)
        self.assertLess(reference(Case("q", expect="the same text"), "something else entirely"), 0.5)
        self.assertEqual(reference(Case("q", expect={"total": 3}), {"total": 3}), 1.0)
        self.assertEqual(checks.equals(Case("q", expect="Forty  Two"), "forty two"), 1.0)

    def test_ready_made_checks(self):
        case = Case("q")
        self.assertEqual(checks.contains("a", "b")(case, "only a"), 0.5)
        self.assertEqual(checks.excludes("secret")(case, "a SECRET"), 0.0)
        self.assertEqual(checks.matches(r"order #\d+")(case, "Order #41327"), 1.0)
        self.assertEqual(checks.max_words(3)(case, "one two three four"), 0.0)
        self.assertEqual(checks.valid_json("a", "b")(case, '{"a": 1}'), 0.5)
        self.assertEqual(checks.valid_json()(case, "not json"), 0.0)

    def test_a_check_can_be_limited_to_some_issues(self):
        @checks.check(issues=["tone"])
        def polite(case, output):
            return "please" in output

        self.assertEqual(score_case(Case("q", issue="tone"), "please wait", [polite]), 1.0)
        self.assertIsNone(score_case(Case("q", issue="other"), "please wait", [polite]))

    def test_a_grader_with_no_readable_score_says_nothing(self):
        self.assertEqual(checks.judged(lambda prompt: "SCORE: 7/10", "be kind")(Case("q"), "a"), 0.7)
        self.assertIsNone(checks.judged(lambda prompt: "no idea", "be kind")(Case("q"), "a"))


class StoreTests(unittest.TestCase):
    def exercise(self, store):
        first = store.add("case", {"a": 1})
        second = store.add("case", {"a": 2})
        store.add("run", {"b": 1})
        store.put("case", first, {"a": 3})
        self.assertEqual(store.rows("case"), [(first, {"a": 3}), (second, {"a": 2})])
        self.assertEqual(store.get_state("missing", "default"), "default")
        store.set_state("version", 4)
        store.set_state("version", 5)
        self.assertEqual(store.get_state("version"), 5)

    def test_memory_and_sqlite_behave_the_same(self):
        self.exercise(MemoryStore())
        with tempfile.TemporaryDirectory() as folder:
            store = SqliteStore(Path(folder) / "deep" / "record.db")
            self.exercise(store)
            store.close()

    def test_sqlite_keeps_the_record_and_stores_odd_values_as_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.db"
            store = SqliteStore(path)
            row = store.add("run", {"output": object})
            store.close()
            store = SqliteStore(path)
            self.assertEqual(store.rows("run"), [(row, {"output": "<class 'object'>"})])
            store.close()


class JudgeModeTests(unittest.TestCase):
    """The system is any callable and the user deploys it themselves."""

    def setUp(self):
        self.loop = Loop(lambda request: fake_llm("", request))
        for topic in STUDIES[:3]:
            self.loop.fail(self.loop.run(f"Summarise the {topic} study"), issue="missing_citation", must=["[source]"])
        self.loop.approve(self.loop.run("What is a p-value?"), must=["p-value"])

    def test_a_better_system_is_accepted_and_adopted(self):
        better = lambda request: fake_llm(CITE, request)  # noqa: E731
        verdict = self.loop.judge(better, name="cite sources")
        self.assertTrue(verdict, verdict.reason)
        self.assertEqual((verdict.before, verdict.after, verdict.cases), (0.0, 1.0, 3))
        self.assertEqual(verdict.validation, IN_SAMPLE)
        self.assertAlmostEqual(verdict.p_value, (1 / 6) ** 3)  # three steady cases, two answers each
        self.assertFalse(verdict.installed)
        self.assertEqual(self.loop("Summarise the diet study"), "Answer to: Summarise the diet study.")

        before = len(self.loop.verdicts())
        adopted = self.loop.adopt(better, name="cite sources", verdict=verdict)
        self.assertIs(adopted, verdict)
        self.assertEqual(len(self.loop.verdicts()), before)  # not judged a second time
        with self.assertRaises(ValueError):
            self.loop.adopt(better, verdict=verdict)  # a verdict is good for one switch
        self.assertTrue(adopted.installed)
        self.assertEqual(self.loop.version, 1)
        self.assertIn("[source]", self.loop("Summarise the diet study"))
        self.assertEqual(self.loop.history()[0]["kind"], "system")

    def test_a_fix_that_breaks_an_approved_answer_is_rejected(self):
        verdict = self.loop.judge(lambda request: "[source]")  # passes the failures, says nothing else
        self.assertFalse(verdict)
        self.assertIn("approved answers", verdict.reason)
        self.assertEqual((verdict.before, verdict.after, verdict.guard_loss), (0.0, 1.0, 1.0))

    def test_an_approval_no_check_can_score_is_reported_as_unguarded(self):
        loop = Loop(lambda request: fake_llm("", request))
        loop.fail(loop.run("Summarise the sleep study"), must=["[source]"])
        approved = loop.approve(loop.run("What is a p-value?"))
        self.assertIsNone(approved.expect)  # the approved wording is kept, not made the one right answer
        self.assertEqual(approved.output, "Answer to: What is a p-value?.")
        verdict = loop.judge(lambda request: "[source]")
        self.assertTrue(verdict)
        self.assertEqual(verdict.evidence["approved_unchecked"], 1)
        self.assertIn("1 approved answers have no check", verdict.reason)

    def test_a_meaning_scorer_can_guard_approved_answers(self):
        class SameTopic:
            def score(self, reference, candidate):
                return 1.0 if "p-value" in candidate else 0.1

        loop = Loop(
            lambda request: fake_llm("", request),
            checks=[checks.terms, checks.stays_close_to_approved(SameTopic())],
        )
        loop.fail(loop.run("Summarise the sleep study"), must=["[source]"])
        loop.approve(loop.run("What is a p-value?"))
        self.assertFalse(loop.judge(lambda request: "[source]"))
        self.assertTrue(loop.judge(lambda request: fake_llm(CITE, request) + " (reworded)"))

    def test_with_no_failures_recorded_only_the_guards_are_checked(self):
        loop = Loop(lambda request: fake_llm("", request))
        loop.approve(loop.run("What is a p-value?"), must=["p-value"])
        self.assertTrue(loop.judge(lambda request: fake_llm(CITE, request)))
        self.assertFalse(loop.judge(lambda request: "something unrelated"))

    def test_an_empty_record_cannot_pass_anything(self):
        empty = Loop(lambda request: "a")
        verdict = empty.judge(lambda request: "b")
        self.assertFalse(verdict)
        self.assertIn("nothing to judge on", verdict.reason)

        unchecked = Loop(lambda request: "a")
        unchecked.approve(unchecked.run("q"))  # an approval that says nothing about why
        verdict = unchecked.judge(lambda request: "b")
        self.assertFalse(verdict)
        self.assertIn("nothing to judge on", verdict.reason)
        self.assertEqual(verdict.evidence["approved_unchecked"], 1)

    def test_a_candidate_that_crashes_loses_and_the_record_says_why(self):
        def broken(request):
            raise ValueError("no such model")

        verdict = self.loop.judge(broken)
        self.assertFalse(verdict)
        self.assertEqual(verdict.evidence["candidate_errors"]["count"], 6)  # 3 cases, 2 answers each
        self.assertIn("no such model", verdict.evidence["candidate_errors"]["last"])

    def test_a_deployed_system_that_is_down_is_reported_not_scored(self):
        def down(request):
            raise ConnectionError("server is down")

        loop = Loop(down)
        loop.fail("q", must=["x"])
        with self.assertRaisesRegex(RuntimeError, "server is down"):
            loop.judge(lambda request: "x")

    def test_failures_no_check_can_score_are_called_out(self):
        loop = Loop(lambda request: "a")
        loop.fail("q", note="too vague")
        verdict = loop.judge(lambda request: "b")
        self.assertFalse(verdict)
        self.assertIn("no check applies", verdict.reason)

    def test_judging_one_issue_guards_the_others(self):
        self.loop.fail(self.loop.run("Translate 'chat'"), issue="translation", expect="cat")
        worse_elsewhere = lambda request: (  # noqa: E731
            "nonsense" if "Translate" in request else fake_llm(CITE, request)
        )
        verdict = self.loop.judge(worse_elsewhere, issue="missing_citation")
        self.assertFalse(verdict)
        self.assertIn("other failures", verdict.reason)

    def test_managed_only_methods_say_what_they_need(self):
        with self.assertRaisesRegex(TypeError, "build="):
            self.loop.learn()
        with self.assertRaisesRegex(TypeError, "build="):
            self.loop.judge(Change.set("a", 1))
        with self.assertRaises(TypeError):
            Loop()
        with self.assertRaises(TypeError):
            Loop(lambda x: x, build=lambda config: (lambda x: x))


class NoiseTests(unittest.TestCase):
    """A gain has to stand out from the system's own run-to-run scatter."""

    def test_p_value_against_run_to_run_noise(self):
        from erratica.core.gates import noise_p_value

        steady = noise_p_value([(0.0, 0.0), (0.0, 0.0)], [(1.0, 1.0), (1.0, 1.0)])
        self.assertAlmostEqual(steady, 1 / 36)
        # The same mean gain, but each system scatters as much as they differ.
        scattered = noise_p_value([(0.0, 1.0), (0.0, 1.0)], [(1.0, 0.5), (0.5, 1.0)])
        self.assertGreater(scattered, 0.2)
        self.assertIsNone(noise_p_value([(0.0,)], [(1.0,)]))  # one answer each: no noise to measure
        many = noise_p_value([(0.0, 0.0)] * 8, [(1.0, 1.0)] * 8)  # too many to enumerate: sampled
        self.assertLess(many, 0.001)

    def flaky(self, good_every):
        """A system whose answer is right on every ``good_every``-th call."""
        calls = {"n": 0}

        def system(request):
            calls["n"] += 1
            return "yes" if calls["n"] % good_every == 0 else "no"

        return system

    def test_a_gain_inside_the_noise_is_not_installed(self):
        loop = Loop(self.flaky(3), repeats=4)
        for index in range(3):
            loop.fail(f"q{index}", must=["yes"])
        verdict = loop.judge(self.flaky(2), name="right slightly more often")
        self.assertGreater(verdict.after, verdict.before)
        self.assertFalse(verdict)
        self.assertIn("could be chance", verdict.reason)
        self.assertTrue(loop.judge(lambda request: "yes", name="always right"))

    def test_with_one_answer_per_case_the_test_runs_across_cases(self):
        def gate_loop(count):
            loop = Loop(lambda request: "no", repeats=1)
            for index in range(count):
                loop.fail(f"q{index}", must=["yes"])
            return loop

        two = gate_loop(2).judge(lambda request: "yes")
        self.assertEqual(two.p_value, 0.25)  # two cases cannot say more than this
        self.assertFalse(two)
        three = gate_loop(3).judge(lambda request: "yes")
        self.assertEqual(three.p_value, 0.125)
        self.assertTrue(three)

    def test_the_test_can_be_switched_off(self):
        loop = Loop(lambda request: "no", repeats=1, gate=ReplayGate(max_p=None))
        loop.fail("q", must=["yes"])
        self.assertTrue(loop.judge(lambda request: "yes"))


class ReplaceableStageTests(unittest.TestCase):
    def test_a_gate_can_be_a_plain_function(self):
        seen = []

        def my_gate(trial):
            seen.append((trial.name, len(trial.target), trial.validation))
            gain = trial.compare(trial.target).gain
            return gain > 0.5, f"gain {gain:.1f}"

        loop = Loop(lambda request: "no", gate=my_gate)
        loop.fail("q", must=["yes"])
        verdict = loop.judge(lambda request: "yes", name="say yes")
        self.assertTrue(verdict)
        self.assertEqual((verdict.reason, verdict.gate, verdict.change), ("gain 1.0", "my_gate", "say yes"))
        self.assertEqual(seen, [("say yes", 1, IN_SAMPLE)])
        self.assertFalse(Loop(lambda request: "no", gate=lambda trial: False).judge(lambda request: "yes"))

    def test_checks_are_plain_functions_reading_the_case(self):
        def right_total(case, output):
            return output == case.meta["total"]

        loop = Loop(lambda numbers: 0, checks=[right_total])
        loop.fail([1, 2, 3], meta={"total": 6})
        loop.fail([4, 5], meta={"total": 9})
        verdict = loop.judge(sum)
        self.assertTrue(verdict, verdict.reason)
        self.assertEqual(loop.score(sum)["failures"], {"score": 1.0, "cases": 2})

    def test_events_reach_handlers(self):
        events = []
        loop = managed(proposers=[rules_from_feedback({"missing_citation": CITE})])
        for name in ("run", "case", "verdict", "install", "rollback"):
            loop.on(name, lambda payload, name=name: events.append(name))
        seed(loop, 3)
        loop.learn()
        loop.rollback()
        self.assertEqual(events, ["run", "case"] * 3 + ["verdict", "install", "rollback"])

    def test_answers_can_be_scored_in_parallel_and_repeated(self):
        calls = []

        def counting(request):
            calls.append(request)
            return "no"

        loop = Loop(counting, workers=4, repeats=3)
        for index in range(6):
            loop.fail(f"q{index}", must=["yes"])
        verdict = loop.judge(lambda request: "yes")
        self.assertTrue(verdict)
        self.assertEqual((verdict.before, verdict.after, verdict.cases), (0.0, 1.0, 6))
        self.assertEqual(len(calls), 18)
        loop.judge(lambda request: "yes")
        self.assertEqual(len(calls), 18)  # the deployed system's scores are reused until it changes

    def test_a_store_is_anything_with_five_methods(self):
        class Recording(MemoryStore):
            added = []

            def add(self, kind, data):
                self.added.append(kind)
                return super().add(kind, data)

        loop = Loop(lambda request: "a", store=Recording())
        loop.fail(loop.run("q"), must=["b"])
        self.assertEqual(Recording.added, ["run", "case"])


class ManagedModeTests(unittest.TestCase):
    def test_learning_installs_a_rule_judged_on_held_out_failures(self):
        loop = managed(proposers=[rules_from_feedback({"missing_citation": CITE})])
        seed(loop, 4)
        loop.approve(loop.run("What is a p-value?"))

        verdicts = loop.learn()

        self.assertEqual(len(verdicts), 1)
        verdict = verdicts[0]
        self.assertTrue(verdict.installed, verdict.reason)
        self.assertEqual(verdict.validation, HOLDOUT)
        self.assertEqual((verdict.before, verdict.after, verdict.cases), (0.0, 1.0, 2))
        self.assertEqual(loop.config, {"rules": [CITE]})
        self.assertEqual((loop.version, verdict.version), (1, 1))
        self.assertIn("[source]", loop("Summarise the stress study"))
        self.assertEqual(loop.learn(), [])  # nothing new to look at
        self.assertEqual(loop.score()["failures"]["score"], 1.0)

    def test_the_proposer_never_sees_the_failures_it_is_judged_on(self):
        seen = []

        def spy(context):
            seen.extend(case.input for case in context.failures)
            return Change.append("rules", CITE, kind="rule")

        loop = managed(proposers=[spy])
        seed(loop, 6)
        verdict = loop.learn()[0]
        self.assertEqual(verdict.issue, "missing_citation")
        self.assertEqual(len(seen), 3)
        self.assertEqual(verdict.cases, 3)
        scored = {case.input for case in loop.cases()} - set(seen)
        self.assertEqual(len(scored), 3)

    def test_too_few_failures_are_judged_in_sample_and_say_so(self):
        loop = managed(proposers=[rules_from_feedback({"missing_citation": CITE})])
        seed(loop, 2)
        self.assertEqual(loop.learn()[0].validation, IN_SAMPLE)

        strict = managed(
            proposers=[rules_from_feedback({"missing_citation": CITE})],
            gate=ReplayGate(require_holdout=True),
        )
        seed(strict, 2)
        verdict = strict.learn()[0]
        self.assertFalse(verdict)
        self.assertIn("hold", verdict.reason)
        self.assertEqual(strict.config, {"rules": []})

    def test_the_ladder_moves_on_after_a_rejection_and_stops_at_a_win(self):
        loop = managed(
            proposers=[
                rules_from_feedback({"missing_citation": "Be nice."}),
                rules_from_feedback({"missing_citation": CITE}),
                rules_from_feedback({"missing_citation": "Never reached."}),
            ]
        )
        seed(loop, 4)
        verdicts = loop.learn()
        self.assertEqual([v.accepted for v in verdicts], [False, True])
        self.assertEqual(loop.config, {"rules": [CITE]})

    def test_new_failures_reopen_an_issue_and_force_reopens_everything(self):
        loop = managed(proposers=[rules_from_feedback({"missing_citation": "Be nice."})])
        seed(loop, 3)
        self.assertEqual(len(loop.learn()), 1)
        self.assertEqual(loop.learn(), [])
        loop.fail(loop.run("Summarise the stress study"), issue="missing_citation", must=["[source]"])
        self.assertEqual(len(loop.learn()), 1)
        self.assertEqual(len(loop.learn(force=True)), 1)
        self.assertEqual(loop.learn(issues=["something else"], force=True), [])

    def test_rules_are_written_from_the_terms_people_asked_for(self):
        loop = managed(proposers=[rules_from_feedback()])
        seed(loop, 4)
        verdict = loop.learn()[0]
        self.assertTrue(verdict.installed, verdict.reason)
        self.assertEqual(loop.config["rules"], ["Always mention: [source]."])

    def test_a_model_can_write_the_rule(self):
        prompts = []

        def writer(prompt):
            prompts.append(prompt)
            return "<think>hmm</think>\nInstruction: " + CITE

        loop = managed(proposers=[rule_from_model(writer)])
        seed(loop, 4)
        self.assertTrue(loop.learn()[0].installed)
        self.assertEqual(loop.config["rules"], [CITE])
        self.assertIn("Should mention: [source]", prompts[0])

    def test_examples_are_only_offered_when_they_can_be_judged_on_other_cases(self):
        def copying_llm(system_prompt, request):
            return "cited [source]" if "Good answer: cited [source]" in system_prompt else "plain"

        def loop_with(count):
            loop = Loop(build=prompted(copying_llm), config={}, proposers=[examples_from_failures()])
            for topic in STUDIES[:count]:
                loop.fail(loop.run(f"Summarise the {topic} study"), issue="uncited", expect="cited [source]")
            return loop

        self.assertEqual(loop_with(2).learn(), [])
        loop = loop_with(4)
        verdict = loop.learn()[0]
        self.assertTrue(verdict.installed, verdict.reason)
        self.assertEqual(verdict.evidence["ops"][0]["value"]["output"], "cited [source]")
        self.assertIn("Examples of good answers", render_prompt(loop.config))

    def test_any_setting_can_be_searched_not_only_prompts(self):
        def build(config):
            return lambda request: "long answer " * config["length"]

        @checks.check
        def short_enough(case, output):
            return len(output.split()) <= case.meta["limit"]

        loop = Loop(
            build=build,
            config={"length": 50},
            checks=[short_enough],
            proposers=[try_values("length", [50, 40, 5, 1])],
        )
        for index in range(4):
            loop.fail(f"q{index}", issue="too_long", meta={"limit": 20})
        verdicts = loop.learn()
        self.assertEqual([v.change for v in verdicts], ["length = 40", "length = 5"])
        self.assertEqual(loop.config, {"length": 5})

    def test_install_rollback_and_history(self):
        loop = managed()
        seed(loop, 3)
        rejected = loop.install(Change.append("rules", "Be nice.", name="nice"))
        self.assertFalse(rejected.installed)
        self.assertEqual(loop.version, 0)

        loop.install(Change.append("rules", "Be nice.", name="nice"), force=True)
        cite = loop.install(Change.append("rules", CITE, name="cite", issue="missing_citation"))
        self.assertTrue(cite.installed, cite.reason)
        loop.install(Change.set("model", "large", name="model"), force=True)
        self.assertEqual(loop.config, {"rules": ["Be nice.", CITE], "model": "large"})
        self.assertEqual([entry["name"] for entry in loop.history()], ["nice", "cite", "model"])

        self.assertTrue(loop.rollback("nice"))  # from underneath the others
        self.assertEqual(loop.config, {"rules": [CITE], "model": "large"})
        self.assertTrue(loop.rollback())  # the newest
        self.assertEqual(loop.config, {"rules": [CITE]})
        self.assertFalse(loop.rollback("nice"))
        self.assertEqual([entry["active"] for entry in loop.history()], [False, True, False])
        self.assertEqual(loop.version, 5)
        self.assertFalse(loop.install(Change.remove("rules", "absent")).accepted)

    def test_a_change_that_cannot_be_built_is_rejected_not_raised(self):
        def build(config):
            if config.get("model") == "missing":
                raise FileNotFoundError("no such model file")
            return lambda request: "a"

        loop = Loop(build=build, config={"model": "small"})
        verdict = loop.install(Change.set("model", "missing"))
        self.assertFalse(verdict)
        self.assertIn("no such model file", verdict.reason)
        self.assertEqual(loop.config, {"model": "small"})

    def test_audit_removes_a_change_that_stopped_earning_its_place(self):
        loop = managed()
        seed(loop, 3)
        loop.install(Change.append("rules", "Be nice.", name="nice"), force=True)
        loop.install(Change.append("rules", CITE, name="cite"), force=True)
        findings = {finding.change: finding for finding in loop.audit()}
        self.assertFalse(findings["cite"].removed)
        self.assertEqual(findings["cite"].cost_of_removal, 1.0)
        self.assertTrue(findings["nice"].removed)
        self.assertEqual(loop.config, {"rules": [CITE]})

    def test_the_record_and_the_configuration_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.db"
            with managed(store=path, proposers=[rules_from_feedback({"missing_citation": CITE})]) as loop:
                seed(loop, 4)
                loop.approve(loop.run("What is a p-value?"))
                self.assertTrue(loop.learn()[0].installed)

            with managed(store=path, proposers=[rules_from_feedback({"missing_citation": CITE})]) as loop:
                self.assertEqual(loop.config, {"rules": [CITE]})
                self.assertEqual(loop.version, 1)
                self.assertEqual((len(loop.cases("failure")), len(loop.cases("approved"))), (4, 1))
                self.assertIn("[source]", loop("Summarise the stress study"))
                self.assertEqual(loop.learn(), [])
                self.assertEqual(len(loop.verdicts()), 1)
                self.assertTrue(loop.verdicts()[0]["installed"])
                self.assertTrue(loop.rollback())

            with managed(store=path) as loop:
                self.assertEqual((loop.config, loop.version), ({"rules": []}, 2))

    def test_a_persistent_record_insists_on_inputs_it_can_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            with Loop(lambda request: "a", store=Path(folder) / "record.db") as loop:
                loop.fail({"question": "q", "tags": ["a"]}, must=["b"])
                with self.assertRaisesRegex(TypeError, "plain JSON"):
                    loop.fail(("a", "tuple"), must=["b"])
        Loop(lambda request: "a").fail(("a", "tuple"), must=["b"])  # fine in memory


def study(rng):
    return f"Summarise the {rng.choice(STUDIES)} study number {rng.randint(1, 10 ** 6)}"


def cited(item, answer):
    return 1.0 if "[source]" in answer else 0.0


def brief(item, answer):
    return 1.0 if len(answer.split()) <= 40 else 0.0


def certified_gate(max_samples=400):
    return CertifiedGate(
        EGate(["brevity"], delta=0.05, margin=0.05, max_samples=max_samples),
        protected=[Behaviour("brevity", GeneratedStream(study, seed=1), brief)],
        task=Behaviour("cited", GeneratedStream(study, seed=2), cited),
    )


class CertifiedGateTests(unittest.TestCase):
    def test_a_good_change_gets_a_certificate_and_a_harmful_one_does_not(self):
        gate = certified_gate()
        loop = managed(gate=gate)

        harmful = loop.install(Change.append("rules", CITE + " Be thorough.", name="thorough"))
        self.assertFalse(harmful)
        self.assertIn("brevity", harmful.reason)
        self.assertEqual(loop.config, {"rules": []})

        good = loop.install(Change.append("rules", CITE, name="cite"))
        self.assertTrue(good.installed, good.reason)
        certificate = good.evidence["certificate"]
        self.assertEqual(certificate["decision"], "promote")
        self.assertTrue(certificate["protected"]["brevity"]["crossed"])
        self.assertTrue(certificate["task"]["crossed"])

        from erratica.learning.egate import verify_certificate

        self.assertEqual(verify_certificate(certificate), [])

    def test_a_cheap_gate_in_front_saves_the_certified_budget(self):
        gate = certified_gate()
        loop = managed(gate=all_of(ReplayGate(), gate))
        seed(loop, 3)
        screened = loop.install(Change.append("rules", "Be nice.", name="nice"))
        self.assertFalse(screened)
        self.assertEqual(gate.certifier.gate.tested, 0)

        verdict = loop.install(Change.append("rules", CITE, name="cite"))
        self.assertTrue(verdict.installed, verdict.reason)
        self.assertEqual(gate.certifier.gate.tested, 1)
        self.assertEqual((verdict.before, verdict.after), (0.0, 1.0))  # the screen's numbers are kept
        self.assertIn("certificate", verdict.evidence)
        self.assertEqual(verdict.gate, "replay + certified")

    def test_nothing_gets_around_a_certified_gate(self):
        loop = managed(gate=certified_gate())
        with self.assertRaisesRegex(ValueError, "certificate"):
            loop.install(Change.append("rules", CITE), force=True)
        with self.assertRaisesRegex(ValueError, "certificate"):
            loop.audit()
        self.assertTrue(loop.install(Change.append("rules", CITE, name="cite")).installed)
        self.assertFalse(loop.install(Change.append("rules", "Use plain words.", name="plain")).installed)

    def test_only_the_newest_certified_change_can_be_rolled_back(self):
        loop = managed(gate=certified_gate(), config={"rules": [], "notes": []})
        self.assertTrue(loop.install(Change.append("rules", CITE, name="cite")).installed)
        later = {"id": 999, "name": "later", "active": True, "change": Change.append("notes", "x", name="later")}
        loop._installed.append(later)
        with self.assertRaisesRegex(ValueError, "newest"):
            loop.rollback("cite")
        loop._installed.pop()
        self.assertTrue(loop.rollback("cite"))
        self.assertEqual(loop.config, {"rules": [], "notes": []})
        with self.assertRaises(ValueError):
            CertifiedGate(EGate([], reuse=True))

    def test_budget_and_sample_positions_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.db"
            first = certified_gate()
            with managed(gate=first, store=path) as loop:
                loop.install(Change.append("rules", CITE + " Be thorough.", name="thorough"))
            used = first.certifier.protected[0].stream.used
            self.assertGreater(used, 0)

            second = certified_gate()
            with managed(gate=second, store=path):
                pass
            self.assertEqual(second.certifier.gate.tested, 1)
            self.assertEqual(second.certifier.protected[0].stream.used, used)
            self.assertEqual(second.task.stream.used, first.task.stream.used)


class VerdictTests(unittest.TestCase):
    def test_a_verdict_is_truthy_when_accepted_and_reports_its_gain(self):
        self.assertTrue(Verdict(True))
        self.assertFalse(Verdict(False))
        self.assertAlmostEqual(Verdict(True, before=0.2, after=0.7).gain, 0.5)
        self.assertIsNone(Verdict(True).gain)


if __name__ == "__main__":
    unittest.main()
