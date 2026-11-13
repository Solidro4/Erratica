from __future__ import annotations

import unittest

from erratica.learning.egate import PROMOTE, REJECT, EGate
from erratica.learning.evalstream import (
    Behaviour,
    GeneratedStream,
    PoolStream,
    StreamExhausted,
    certify,
)


def make_sum(rng):
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return {"prompt": f"{a}+{b}", "answer": str(a + b)}


def exact(item, response):
    return 1.0 if response == item["answer"] else 0.0


def solver(prompt: str, route=None) -> str:
    a, b = prompt.split("+")
    return str(int(a) + int(b))


def weak(prompt: str, route=None) -> str:
    # Right only when no carrying is involved in the last digit.
    a, b = prompt.split("+")
    return str(int(a) + int(b)) if (int(a) % 10 + int(b) % 10) < 10 else "?"


def polite(prompt: str, route=None) -> str:
    return "ok"


def is_ok(item, response):
    return 1.0 if response == "ok" else 0.0


class StreamTests(unittest.TestCase):
    def test_pool_uses_each_item_once_then_stops(self) -> None:
        stream = PoolStream(range(5), seed=1)
        drawn = [stream.draw() for _ in range(5)]
        self.assertEqual(sorted(drawn), [0, 1, 2, 3, 4])
        self.assertEqual(stream.remaining, 0)
        with self.assertRaises(StreamExhausted):
            stream.draw()

    def test_streams_resume_from_a_saved_position(self) -> None:
        first = PoolStream(range(10), seed=3)
        head = [first.draw() for _ in range(4)]
        resumed = PoolStream(range(10), seed=3, used=first.used)
        self.assertNotIn(resumed.draw(), head)

        generated = GeneratedStream(make_sum, seed=7)
        items = [generated.draw() for _ in range(3)]
        again = GeneratedStream(make_sum, seed=7, used=2)
        self.assertEqual(again.draw(), items[2])
        self.assertEqual(len({item["prompt"] for item in items}), 3)


class CertifyTests(unittest.TestCase):
    def _behaviours(self, pool_size=None):
        tone_stream = (
            PoolStream([f"q{i}" for i in range(pool_size)])
            if pool_size
            else GeneratedStream(lambda rng: f"q{rng.random()}")
        )
        tone = Behaviour("tone", tone_stream, is_ok)
        maths = Behaviour(
            "maths",
            GeneratedStream(make_sum, seed=1),
            exact,
            prompt=lambda item: item["prompt"],
        )
        return tone, maths

    def test_real_improvement_is_certified_on_fresh_samples(self) -> None:
        tone, maths = self._behaviours()
        gate = EGate(["tone"], delta=0.05, margin=0.05)
        seen: list[str] = []

        def candidate(prompt: str, route=None) -> str:
            seen.append(prompt)
            return solver(prompt) if "+" in prompt else "ok"

        def baseline(prompt: str, route=None) -> str:
            return weak(prompt) if "+" in prompt else "ok"

        certificate = certify(gate, candidate=candidate, baseline=baseline, protected=[tone], task=maths)

        self.assertEqual(certificate["decision"], PROMOTE)
        # The certificate also bounds how much each score really changed.
        task_evidence = certificate["task"]
        self.assertGreater(task_evidence["observed_mean_change"], 0.2)
        self.assertLess(task_evidence["change_at_least"], task_evidence["observed_mean_change"])
        tone_evidence = certificate["protected"]["tone"]
        self.assertAlmostEqual(tone_evidence["observed_mean_change"], 0.0)
        self.assertGreater(tone_evidence["change_at_least"], -0.15)
        self.assertLessEqual(tone_evidence["change_at_least"], 0.0)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(sum(certificate["samples_consumed"].values()), len(seen))
        self.assertLess(certificate["samples_consumed"]["maths"], 60)

    def test_candidate_that_breaks_a_protected_behaviour_is_rejected(self) -> None:
        tone, maths = self._behaviours()
        gate = EGate(["tone"], max_samples=400)
        calls = {"n": 0}

        def rude_sometimes(prompt: str, route=None) -> str:
            if "+" in prompt:
                return solver(prompt)
            calls["n"] += 1
            return "no" if calls["n"] % 8 == 0 else "ok"

        certificate = certify(
            gate,
            candidate=rude_sometimes,
            baseline=lambda p, r=None: weak(p) if "+" in p else "ok",
            protected=[tone],
            task=maths,
        )
        self.assertEqual(certificate["decision"], REJECT)

    def test_running_out_of_fresh_samples_is_a_rejection(self) -> None:
        tone, maths = self._behaviours(pool_size=10)
        gate = EGate(["tone"])

        certificate = certify(
            gate,
            candidate=lambda p, r=None: solver(p) if "+" in p else "ok",
            baseline=lambda p, r=None: weak(p) if "+" in p else "ok",
            protected=[tone],
            task=maths,
        )
        self.assertEqual(certificate["decision"], REJECT)
        self.assertIn("ran out", certificate["note"])
        self.assertEqual(tone.stream.remaining, 0)

    def test_streams_must_match_the_gate(self) -> None:
        tone, maths = self._behaviours()
        with self.assertRaises(ValueError):
            certify(EGate(["safety"]), candidate=polite, baseline=polite, protected=[tone])

    def test_behaviours_on_one_stream_share_samples_and_answers(self) -> None:
        stream = GeneratedStream(make_sum, seed=5)
        as_prompt = lambda item: item["prompt"]  # noqa: E731
        digits = Behaviour("digits", stream, lambda item, r: 1.0 if r.isdigit() else 0.0, prompt=as_prompt)
        short = Behaviour("short", stream, lambda item, r: 1.0 if len(r) <= 4 else 0.0, prompt=as_prompt)
        maths = Behaviour("maths", stream, exact, prompt=as_prompt)
        calls = {"candidate": 0, "baseline": 0}

        def candidate(prompt, route=None):
            calls["candidate"] += 1
            return solver(prompt)

        def baseline(prompt, route=None):
            calls["baseline"] += 1
            return weak(prompt) if int(prompt.split("+")[0]) % 2 else solver(prompt)

        certificate = certify(
            EGate(["digits", "short"]),
            candidate=candidate,
            baseline=baseline,
            protected=[digits, short],
            task=maths,
        )

        self.assertEqual(certificate["decision"], PROMOTE)
        # One answer per system per item, however many behaviours scored it.
        self.assertEqual(calls["candidate"], certificate["items_drawn"])
        self.assertLessEqual(calls["baseline"], certificate["items_drawn"])
        self.assertEqual(certificate["model_calls"], calls["candidate"] + calls["baseline"])
        self.assertGreater(sum(certificate["samples_consumed"].values()), certificate["items_drawn"])

    def test_clearly_harmful_candidate_is_stopped_early(self) -> None:
        tone, maths = self._behaviours()
        gate = EGate(["tone"], max_samples=5000)

        certificate = certify(
            gate,
            candidate=lambda p, r=None: solver(p) if "+" in p else "no",
            baseline=lambda p, r=None: weak(p) if "+" in p else "ok",
            protected=[tone],
            task=maths,
        )

        self.assertEqual(certificate["decision"], REJECT)
        self.assertEqual(certificate["stopped_early_on"], ["tone"])
        self.assertLess(certificate["samples_consumed"]["tone"], 60)

        patient = EGate(["tone"], max_samples=150, futility=None)
        slow = certify(
            patient,
            candidate=lambda p, r=None: solver(p) if "+" in p else "no",
            baseline=lambda p, r=None: weak(p) if "+" in p else "ok",
            protected=[tone],
            task=maths,
        )
        self.assertEqual(slow["decision"], REJECT)
        self.assertEqual(slow["samples_consumed"]["tone"], 150)

    def test_behaviours_a_patch_cannot_affect_are_not_retested(self) -> None:
        tone, maths = self._behaviours()
        gate = EGate(["tone", "other_route"])

        certificate = certify(
            gate,
            candidate=lambda p, r=None: solver(p) if "+" in p else "ok",
            baseline=lambda p, r=None: weak(p) if "+" in p else "ok",
            protected=[tone],
            task=maths,
        )
        self.assertEqual(certificate["decision"], PROMOTE)
        self.assertEqual(certificate["not_retested"], ["other_route"])
        self.assertEqual(list(certificate["protected"]), ["tone"])


if __name__ == "__main__":
    unittest.main()
