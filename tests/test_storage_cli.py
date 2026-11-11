from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from erratica import Erratica, HybridStringSimilarityScorer, OllamaEmbeddingScorer
from erratica.cli.main import main as cli_main
from erratica.engine.ollama_adapter import OllamaModel
from erratica.learning.similarity import _EmbeddingScorer

from tests.helpers import REFUND_IDEAL, REFUND_PROMPT, EngineTestCase, report_failure

# The tables exactly as version 0.4.0 created them.
_LEGACY_SCHEMA = """
CREATE TABLE interactions (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL,
    response TEXT NOT NULL, route TEXT NOT NULL, policy_version INTEGER NOT NULL,
    metadata_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    rating INTEGER, issue_type TEXT NOT NULL, required_terms_json TEXT, preferred_format TEXT,
    ideal_response TEXT, correction TEXT, notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE patches (id INTEGER PRIMARY KEY AUTOINCREMENT, patch_type TEXT NOT NULL,
    target TEXT NOT NULL, content_json TEXT NOT NULL, rationale TEXT NOT NULL,
    source_issue_type TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'proposed',
    score_before REAL, score_after REAL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    promoted_at TEXT);
CREATE TABLE policy_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, rule_text TEXT NOT NULL,
    source_patch_id INTEGER REFERENCES patches(id) ON DELETE SET NULL,
    active INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    activated_at TEXT);
CREATE TABLE few_shot_examples (id INTEGER PRIMARY KEY AUTOINCREMENT, route TEXT NOT NULL,
    prompt TEXT NOT NULL, response TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE mistake_triage (id INTEGER PRIMARY KEY AUTOINCREMENT, issue_type TEXT NOT NULL,
    verdict TEXT NOT NULL, score REAL NOT NULL DEFAULT 0, reasons_json TEXT,
    case_count INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""


class StorageTests(EngineTestCase):
    def test_database_from_0_4_0_is_upgraded_in_place(self) -> None:
        path = self.tmp / "legacy.db"
        connection = sqlite3.connect(path)
        connection.executescript(_LEGACY_SCHEMA)
        connection.execute(
            "INSERT INTO interactions (prompt, response, route, policy_version) VALUES (?, ?, ?, 0)",
            (REFUND_PROMPT, "old answer", "support"),
        )
        connection.execute(
            "INSERT INTO feedback (interaction_id, rating, issue_type, required_terms_json, ideal_response) "
            "VALUES (1, -1, 'missing_resolution', '[]', ?)",
            (REFUND_IDEAL,),
        )
        connection.execute(
            "INSERT INTO patches (patch_type, target, content_json, rationale, source_issue_type, status) "
            "VALUES ('prompt_rule', 'system_prompt', '{\"rule_text\": \"Old rule.\"}', 'r', 'x', 'promoted')"
        )
        connection.execute(
            "INSERT INTO policy_rules (rule_text, source_patch_id, active) VALUES ('Old rule.', 1, 1)"
        )
        connection.commit()
        connection.close()

        ai = self.engine(db_name="legacy.db")

        failure = ai.memory.failed_interactions()[0]
        self.assertEqual(failure["source"], "human")
        # A rule from before routes existed applies everywhere.
        self.assertIn("Old rule.", ai.policy.render_system_prompt(route="support"))
        self.assertIn("Old rule.", ai.policy.render_system_prompt(route="finance"))
        self.assertIn("Resolution:", ai.generate(REFUND_PROMPT).text)
        ai.run_learning_cycle()
        self.assertIn("Old rule.", ai.model_card())

    def test_policy_version_counts_promotions_and_rollbacks(self) -> None:
        ai = self.engine()
        self.assertEqual(ai.policy.version(), 0)
        self.seed_support_failures(ai)
        patch_id = ai.run_learning_cycle()[0].patch_id
        self.assertEqual(ai.policy.version(), 1)
        self.assertEqual(ai.generate("Customer asks about a refund.").policy_version, 1)
        ai.rollback(patch_id)
        self.assertEqual(ai.policy.version(), 2)

    def test_export_writes_chat_format_and_respects_exclusions(self) -> None:
        ai = self.engine()
        report_failure(ai, REFUND_PROMPT, issue_type="missing_resolution", ideal_response=REFUND_IDEAL)
        report_failure(ai, "Customer wants a refund for order #9.", issue_type="x", ideal_response="Done.")

        report = ai.export_training_data(self.tmp / "sft.jsonl", exclude_prompts=[REFUND_PROMPT.upper()])

        self.assertEqual((report.examples_written, report.held_out), (1, 1))
        record = json.loads((self.tmp / "sft.jsonl").read_text(encoding="utf-8"))
        self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])
        self.assertEqual(record["messages"][2]["content"], "Done.")

    def test_model_card_reports_how_each_patch_was_validated(self) -> None:
        ai = self.engine()
        self.seed_support_failures(ai)
        ai.run_learning_cycle()
        ai.propose_rule("Respond in a single short sentence, avoid detail.", route="support")

        card = ai.model_card(model_name="test-model")

        self.assertIn("# Model Card: test-model", card)
        self.assertIn("Behavior updates promoted: 1", card)
        self.assertIn("Behavior updates rejected: 1", card)
        self.assertIn("Live patches validated on held-out cases: 1/1", card)
        self.assertIn("held-out", card)
        self.assertIn("insufficient improvement", card)
        self.assertTrue(ai.write_model_card(self.tmp / "CARD.md").exists())

    def test_engine_is_a_context_manager(self) -> None:
        with Erratica(db_path=self.tmp / "ctx.db") as ai:
            ai.generate("hello")
        with self.assertRaises(sqlite3.ProgrammingError):
            ai.db.connection.execute("SELECT 1")


class CliTests(EngineTestCase):
    def _run(self, *argv: str) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli_main(list(argv))
        return buffer.getvalue()

    def test_commands_work_on_a_real_database(self) -> None:
        db = str(self.tmp / "cli.db")
        with Erratica(db_path=db) as ai:
            self.seed_support_failures(ai)

        self.assertIn("erratica 0.", self._run("--version"))
        self.assertIn("Commands:", self._run("--help"))
        self.assertIn("PROMOTED", self._run("cycle", "--db", db))
        self.assertIn("No patch was proposed", self._run("cycle", "--db", db))
        self.assertIn("-> kept", self._run("audit", "--db", db))
        self.assertIn("Audit:", self._run("consolidate", "--db", db))
        self.assertIn("Policy version: 1", self._run("dashboard", "--db", db))
        self.assertIn("# Model Card: m", self._run("modelcard", "--db", db, "--name", "m"))
        self.assertIn("Examples written: 2", self._run("export", "--db", db, "--out", str(self.tmp / "o.jsonl")))
        self.assertIn("rolled back", self._run("rollback", "1", "--db", db))
        self.assertIn("Policy version: 2", self._run("dashboard", "--db", db))

    def test_unknown_command_exits_with_usage(self) -> None:
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            cli_main(["nonsense"])
        self.assertEqual(raised.exception.code, 2)


class _FakeOllama(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        type(self).requests.append((self.path, body))
        if self.path == "/api/chat":
            payload = {"message": {"role": "assistant", "content": "<think>hmm</think>\n hello there "}}
        else:
            payload = {
                "embeddings": [[1.0, 0.0] if "cat" in text else [0.0, 1.0] for text in body["input"]]
            }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


class OllamaAdapterTests(unittest.TestCase):
    """Checks the request and response handling against a local stand-in server."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
        cls.host = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _FakeOllama.requests.clear()

    def test_chat_request_shape_and_think_stripping(self) -> None:
        model = OllamaModel("tiny", host=self.host, max_tokens=99)

        self.assertEqual(model.generate("hi", "be brief", "general"), "hello there")

        path, body = _FakeOllama.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["model"], "tiny")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"], {"temperature": 0.0, "num_predict": 99})
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_unreachable_server_gives_a_helpful_error(self) -> None:
        model = OllamaModel("tiny", host="http://127.0.0.1:9", timeout=2)
        with self.assertRaises(RuntimeError) as raised:
            model.generate("hi", "", "general")
        self.assertIn("ollama serve", str(raised.exception))

    def test_embedding_scorer_scores_and_caches(self) -> None:
        scorer = OllamaEmbeddingScorer("embedder", host=self.host)

        self.assertEqual(scorer.score("a cat", "another cat"), 1.0)
        self.assertEqual(scorer.score("a cat", "a dog"), 0.0)
        # "a cat" was embedded once and reused.
        self.assertEqual(len(_FakeOllama.requests), 3)
        self.assertEqual(_FakeOllama.requests[0][0], "/api/embed")


class SimilarityTests(unittest.TestCase):
    def test_hybrid_string_scorer_bounds(self) -> None:
        scorer = HybridStringSimilarityScorer()
        self.assertEqual(scorer.score("Refund approved in 3-5 days", "Refund approved in 3-5 days"), 1.0)
        self.assertLess(scorer.score("Refund approved in 3-5 days", "The weather is sunny"), 0.3)
        self.assertGreater(scorer.score("Timeline: 3–5 business days", "timeline: 3-5 business days"), 0.95)

    def test_embedding_base_clamps_negative_cosine_to_zero(self) -> None:
        class Opposite(_EmbeddingScorer):
            def _embed(self, texts):
                return [[1.0, 0.0] if text == "a" else [-1.0, 0.0] for text in texts]

        self.assertEqual(Opposite().score("a", "b"), 0.0)


if __name__ == "__main__":
    unittest.main()
