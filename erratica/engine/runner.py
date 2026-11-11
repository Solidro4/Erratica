from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Protocol, Sequence

from erratica.domains import Domain, DomainRegistry
from erratica.engine.demo_model import RuleAwareDemoModel
from erratica.engine.feedback import FeedbackIngestor
from erratica.engine.memory import MemoryStore
from erratica.learning.clustering import FailureCluster, FailureClusterer, is_failing
from erratica.learning.critic import SelfReviewer, SelfReviewReport
from erratica.learning.evaluator import PatchEvaluator
from erratica.learning.exporter import ExportReport, TrainingDataExporter
from erratica.learning.modelcard import ModelCardBuilder
from erratica.learning.patcher import (
    FewShotPatcher,
    ModelRulePatcher,
    PatchProposal,
    PromptRulePatcher,
    ToolPatcher,
)
from erratica.learning.promoter import PatchPromoter, PromotionDecision
from erratica.learning.replay import ReplayCase, ReplayDatasetBuilder, ReplayRunner
from erratica.learning.similarity import SimilarityScorer
from erratica.learning.splits import split_cases
from erratica.learning.triage import MistakeTriage
from erratica.learning.weights import ModelTrainer, WeightCycleReport, WeightLoop
from erratica.policy.prompt_rules import PromptPolicy
from erratica.policy.routing import RoutingPolicy
from erratica.storage.db import Database
from erratica.tools.sandbox import ToolValidationError, run_tools, validate_tool_source

__all__ = [
    "AuditFinding",
    "ConsolidationReport",
    "Erratica",
    "GenerationResult",
    "ModelAdapter",
    "RuleAwareDemoModel",
]

_AUDITABLE = ("prompt_rule", "few_shot", "tool")


class ModelAdapter(Protocol):
    def generate(self, prompt: str, system_prompt: str, route: str) -> str:
        ...


@dataclass
class GenerationResult:
    interaction_id: int
    text: str
    route: str
    policy_version: int


class _ReplaySet:
    """Cases plus their replay under the current policy, computed once on demand."""

    def __init__(self, runner: ReplayRunner, cases: list[ReplayCase]) -> None:
        self.runner = runner
        self.cases = cases
        self._baseline: Optional[list[dict[str, Any]]] = None

    def baseline(self) -> list[dict[str, Any]]:
        if self._baseline is None:
            self._baseline = self.runner.run(self.cases)
        return self._baseline


@dataclass
class AuditFinding:
    patch_id: int
    patch_type: str
    route: Optional[str]
    # Mean score with the patch minus mean score without it.
    contribution: float
    cases: int
    action: str


@dataclass
class ConsolidationReport:
    decisions: list[PromotionDecision] = field(default_factory=list)
    audit: list[AuditFinding] = field(default_factory=list)
    self_review: Optional[SelfReviewReport] = None
    weights: Optional[WeightCycleReport] = None


class Erratica:
    """A model wrapper that learns from its mistakes, and proves each lesson.

    Wrap any model that has ``generate(prompt, system_prompt, route)``.
    Record feedback on its answers, then call ``run_learning_cycle``.

    Arguments:
        model: the answering model. Defaults to a deterministic simulator.
        similarity_scorer: how answers are compared with reference answers.
        domains: domain packs to use instead of the built-in ones.
        proposer: a model that writes candidate rules (and tools) from
            failures. Without one, only template rules and few-shot
            examples are proposed.
        critics: extra graders used when scoring replayed answers.
        enable_tools: allow the tool rung and run promoted tools at answer
            time. Off by default because tools execute generated code.
        enable_recall: inject similar past corrections at answer time.
        require_holdout: refuse to promote anything that could only be
            validated in-sample.
        trust_self_review: let critic-written corrections be recalled at
            answer time, as if a person had written them.
        tool_attempts: how many drafts of a tool the proposer gets. After a
            draft fails on the training cases it is shown what went wrong.
    """

    def __init__(
        self,
        db_path: str | Path = Path("data") / "erratica.db",
        *,
        model: Optional[ModelAdapter] = None,
        similarity_scorer: Optional[SimilarityScorer] = None,
        enable_recall: bool = True,
        domains: Optional[Sequence[Domain]] = None,
        proposer: Optional[ModelAdapter] = None,
        critics: Optional[Sequence[Any]] = None,
        enable_tools: bool = False,
        require_holdout: bool = False,
        trust_self_review: bool = False,
        min_delta: float = 0.05,
        regression_tolerance: float = 0.02,
        tool_attempts: int = 3,
    ) -> None:
        self.db = Database(db_path)
        self.tool_attempts = max(1, tool_attempts)
        self.registry = DomainRegistry(list(domains) if domains is not None else None)
        self.memory = MemoryStore(self.db)
        self.feedback = FeedbackIngestor(self.db)
        self.policy = PromptPolicy(self.db, registry=self.registry, enable_tools=enable_tools)
        self.router = RoutingPolicy(self.registry)
        self.model = model or RuleAwareDemoModel()
        self.proposer = proposer
        self.enable_tools = enable_tools
        self.enable_recall = enable_recall
        self.require_holdout = require_holdout
        self.trust_self_review = trust_self_review
        self.min_delta = min_delta
        self.regression_tolerance = regression_tolerance

        self.clusterer = FailureClusterer()
        self.triage = MistakeTriage()
        self.patcher = PromptRulePatcher(self.registry)
        self.few_shot_patcher = FewShotPatcher()
        self.dataset_builder = ReplayDatasetBuilder()
        self.evaluator = PatchEvaluator(
            similarity_scorer=similarity_scorer,
            registry=self.registry,
            critics=critics,
        )
        self.replay_runner = ReplayRunner(
            self.model,
            self.policy,
            self.router,
            recall=self._replay_recall if enable_recall else None,
        )
        self.promoter = PatchPromoter(
            self.db,
            self.policy,
            registry=self.registry,
            min_delta=min_delta,
            regression_tolerance=regression_tolerance,
        )
        self.exporter = TrainingDataExporter(self.db, self.policy)

    def __enter__(self) -> "Erratica":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        route: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Answer a prompt with everything learned so far.

        Pass ``route`` when you know what kind of task this is; otherwise
        the router picks a domain from the prompt.
        """
        route = route or self.router.select_route(prompt)
        policy_version = self.policy.version()
        recalled = self._recall(prompt, route) if self.enable_recall else []
        system_prompt = self.policy.render_system_prompt(
            route=route,
            prompt=prompt,
            extra_examples=recalled or None,
        )
        response = self.model.generate(prompt, system_prompt, route)
        interaction_id = self.memory.record_interaction(
            prompt=prompt,
            response=response,
            route=route,
            policy_version=policy_version,
            metadata=metadata,
        )
        return GenerationResult(
            interaction_id=interaction_id,
            text=response,
            route=route,
            policy_version=policy_version,
        )

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def learn_from_feedback(
        self,
        *,
        interaction_id: int,
        rating: int = -1,
        issue_type: Optional[str] = None,
        required_terms: Optional[list[str]] = None,
        preferred_format: Optional[str] = None,
        ideal_response: Optional[str] = None,
        correction: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Record feedback on an answer.

        ``rating`` below zero reports a problem; above zero approves the
        answer. ``issue_type`` is a short label shared by failures of the
        same kind (for example ``missing_resolution``); failures are grouped
        by it. ``ideal_response`` is the answer that should have been given.
        ``correction`` is accepted as an older name for the same thing.
        ``notes`` is free text for people and is never parsed.
        """
        return self.feedback.record(
            interaction_id=interaction_id,
            rating=rating,
            issue_type=issue_type,
            required_terms=required_terms,
            preferred_format=preferred_format,
            ideal_response=ideal_response,
            correction=correction,
            notes=notes,
        )

    def approve(self, interaction_id: int, *, notes: Optional[str] = None) -> int:
        """Mark an answer as good. Approved answers are what the regression
        guard protects: no later patch or fine-tune may break them."""
        return self.feedback.record(interaction_id=interaction_id, rating=1, notes=notes)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def run_learning_cycle(self, limit: int = 500, *, force: bool = False) -> list[PromotionDecision]:
        """Triage recorded failures and try to fix the learnable ones.

        A cluster is only worked on when it has feedback this method has
        not seen before, so calling it repeatedly is cheap and does not
        fill the audit trail with duplicates. ``force`` re-examines
        everything, which is worth doing after changing the model.
        """
        failures = self.memory.failed_interactions(limit=limit)
        clusters = self.clusterer.cluster(failures)
        decisions: list[PromotionDecision] = []
        for cluster in clusters:
            newest = max((int(case.get("feedback_id") or 0) for case in cluster.cases), default=0)
            if not force and newest <= self.db.last_reviewed_feedback_id(
                route=cluster.route, issue_type=cluster.issue_type
            ):
                continue

            verdict = self.triage.review(cluster)
            if verdict.learnable:
                decisions.extend(self._learn_cluster(cluster, clusters))
            # Recorded only once the cluster has been fully handled, so a
            # cycle that dies halfway (model server down, say) is retried.
            self.db.insert_triage(
                issue_type=cluster.issue_type,
                verdict=verdict.verdict,
                score=verdict.score,
                reasons=verdict.reasons,
                case_count=cluster.size,
                route=cluster.route,
                max_feedback_id=newest,
            )
        return decisions

    def _learn_cluster(
        self,
        cluster: FailureCluster,
        all_clusters: list[FailureCluster],
    ) -> list[PromotionDecision]:
        failing = cluster.failing_cases
        if not failing:
            return []

        # Patches are written from the train side and judged on the holdout
        # side. With too few distinct prompts both sides are the whole
        # cluster and the result is recorded as in-sample.
        split = split_cases(failing)
        if not split.is_holdout and self.require_holdout:
            return []
        train_cluster = cluster.subset(split.train)
        validation_cases = self.dataset_builder.build(split.holdout)
        if not validation_cases:
            return []

        validation = _ReplaySet(self.replay_runner, validation_cases)
        regression = _ReplaySet(self.replay_runner, self._regression_cases(all_clusters, cluster))
        wins = _ReplaySet(
            self.replay_runner,
            self.dataset_builder.build(self.db.list_wins(route=cluster.route, limit=100)),
        )

        human = sum(1 for case in validation_cases if case.source == "human")
        extra_details = {
            "train_cases": len(split.train) if split.is_holdout else 0,
            "human_evidence_fraction": round(human / len(validation_cases), 3),
        }

        decisions: list[PromotionDecision] = []
        for proposal in self._proposal_ladder(train_cluster):
            if proposal.patch_type == "tool":
                proposal, problem = self._refine_tool(proposal, train_cluster)
                if problem:
                    decisions.append(self.promoter.reject(proposal, problem))
                    continue
            decision = self._judge(
                proposal,
                validation,
                regression,
                wins,
                mode=split.mode,
                extra_details=extra_details,
            )
            decisions.append(decision)
            if decision.accepted:
                break
        return decisions

    def _judge(
        self,
        proposal: PatchProposal,
        validation: "_ReplaySet",
        regression: "_ReplaySet",
        wins: "_ReplaySet",
        *,
        mode: str,
        extra_details: Optional[dict[str, Any]] = None,
    ) -> PromotionDecision:
        """Replay one proposal and put it through the promotion gate."""
        evaluation = self.evaluator.evaluate(
            validation.cases,
            validation.baseline(),
            self._replay_with_proposal(validation.cases, proposal),
        )
        min_delta, _ = self.promoter.gates_for(proposal.route)

        regression_summary = None
        wins_summary = None
        if evaluation.delta > min_delta:
            # Only pay for the guard replays once a patch is worth guarding.
            if regression.cases:
                regression_summary = self.evaluator.evaluate(
                    regression.cases,
                    regression.baseline(),
                    self._replay_with_proposal(regression.cases, proposal),
                )
            if wins.cases:
                wins_summary = self.evaluator.evaluate(
                    wins.cases,
                    wins.baseline(),
                    self._replay_with_proposal(wins.cases, proposal),
                )

        return self.promoter.promote(
            proposal,
            evaluation,
            regression=regression_summary,
            wins=wins_summary,
            validation=mode,
            extra_details=extra_details,
        )

    def propose_rule(
        self,
        rule_text: str,
        *,
        route: str,
        issue_type: Optional[str] = None,
        rationale: str = "Submitted by hand.",
        limit: int = 200,
    ) -> PromotionDecision:
        """Put a rule you wrote yourself through the same gate as learned ones.

        It is replayed on the recorded failures of ``route`` (only those
        tagged ``issue_type`` if given) and on the route's approved answers,
        and installed only if it passes. Because the rule was not derived
        from those cases by the system, all of them are used to judge it.
        """
        failures = [
            row
            for row in self.memory.failed_interactions(limit=limit * 5)
            if is_failing(row)
            and row["route"] == route
            and (issue_type is None or row["issue_type"] == issue_type)
        ][:limit]
        proposal = PatchProposal(
            patch_type="prompt_rule",
            target="system_prompt",
            rule_text=rule_text.strip(),
            rationale=rationale,
            source_issue_type=issue_type or "manual",
            route=route,
            origin="manual",
        )
        if not failures:
            return self.promoter.reject(proposal, "no recorded failures on this route to judge it on")
        return self._judge(
            proposal,
            _ReplaySet(self.replay_runner, self.dataset_builder.build(failures)),
            _ReplaySet(self.replay_runner, []),
            _ReplaySet(
                self.replay_runner,
                self.dataset_builder.build(self.db.list_wins(route=route, limit=limit)),
            ),
            mode="manual",
        )

    def _proposal_ladder(self, cluster: FailureCluster) -> Iterator[PatchProposal]:
        """Cheapest fix first. Later rungs are only built if earlier ones lose."""
        active_rules = set(self.policy.active_rules(cluster.route))

        template = self.patcher.propose(cluster)
        if template is not None and template.rule_text not in active_rules:
            yield template

        if self.proposer is not None:
            written = ModelRulePatcher(self.proposer).propose(cluster)
            if written is not None and written.rule_text not in active_rules:
                yield written

        few_shot = self.few_shot_patcher.propose(cluster)
        if few_shot is not None:
            installed = {
                example["prompt"] for example in self.db.list_all_few_shot_examples()
            }
            if few_shot.example_prompt not in installed:
                yield few_shot

        if self.proposer is not None and self.enable_tools:
            tool = ToolPatcher(self.proposer).propose(cluster)
            if tool is not None:
                yield tool

    def _refine_tool(
        self,
        proposal: PatchProposal,
        train_cluster: FailureCluster,
    ) -> tuple[PatchProposal, str]:
        """Try a proposed tool on the training cases and let the proposer fix it.

        First drafts of generated code are often wrong in small ways (a
        parser that trips on thousands separators, say). The tool is run on
        the training side only; if it does not help there, the proposer is
        shown what it returned and asked for a corrected version, up to
        ``tool_attempts`` drafts. Held-out cases are not touched here, so
        the final verdict on them stays honest.

        Returns the last draft and, if it cannot even be tried, the reason.
        """
        rows = train_cluster.failing_cases
        cases = self.dataset_builder.build(rows)[:4]
        training = _ReplaySet(self.replay_runner, cases)
        min_delta, _ = self.promoter.gates_for(proposal.route)
        patcher = ToolPatcher(self.proposer)
        problem = ""
        for attempt in range(self.tool_attempts):
            problem = self._screen_tool(proposal, rows)
            if problem:
                observations = problem
            else:
                results = self._replay_with_proposal(cases, proposal)
                evaluation = self.evaluator.evaluate(cases, training.baseline(), results)
                if evaluation.delta > min_delta:
                    return proposal, ""
                observations = self._describe_tool_trial(proposal, cases, results)
            if attempt == self.tool_attempts - 1:
                break
            repaired = patcher.repair(train_cluster, proposal, observations)
            if repaired is None:
                break
            proposal = repaired
        return proposal, problem

    def _describe_tool_trial(
        self,
        proposal: PatchProposal,
        cases: list[ReplayCase],
        results: list[dict[str, Any]],
    ) -> str:
        tool = {"name": proposal.tool_name or "tool", "source": proposal.tool_source or ""}
        blocks = []
        for case, result in list(zip(cases, results))[:3]:
            outcome = run_tools([tool], case.prompt, timeout=self.policy.tool_timeout)[0]
            lines = [
                f"Request: {case.prompt}",
                f"Your function returned: {outcome.get('output') or '(nothing)'}"
                + (f" [error: {outcome['error']}]" if outcome.get("error") else ""),
                f"The assistant then answered: {result['response'][:300]}",
            ]
            if case.reference:
                lines.append(f"The answer that was wanted: {case.reference[:300]}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) + "\n\nThe answers were still wrong."

    def _screen_tool(self, proposal: PatchProposal, train_cases: list[dict[str, Any]]) -> str:
        """Why a proposed tool cannot even be tried, or '' if it can."""
        source = proposal.tool_source or ""
        try:
            validate_tool_source(source)
        except ToolValidationError as exc:
            return f"tool failed static screening: {exc}"
        tool = {"name": proposal.tool_name or "tool", "source": source}
        outputs = [
            run_tools([tool], str(case.get("prompt", "")), timeout=self.policy.tool_timeout)[0]
            for case in train_cases[:3]
        ]
        if outputs and all(entry.get("error") for entry in outputs):
            return f"tool failed on every training prompt: {outputs[0]['error']}"
        if outputs and not any(str(entry.get("output") or "").strip() for entry in outputs):
            return "tool returned nothing for every training prompt"
        return ""

    def _replay_with_proposal(
        self,
        cases: list[ReplayCase],
        proposal: PatchProposal,
    ) -> list[dict[str, Any]]:
        if proposal.patch_type == "few_shot":
            example = {
                "prompt": proposal.example_prompt or "",
                "response": proposal.example_response or "",
            }
            return self.replay_runner.run(cases, extra_examples=[example])
        if proposal.patch_type == "tool":
            tool = {"name": proposal.tool_name or "tool", "source": proposal.tool_source or ""}
            return self.replay_runner.run(cases, extra_tools=[tool])
        return self.replay_runner.run(cases, extra_rules=[proposal.rule_text])

    def _regression_cases(
        self,
        clusters: list[FailureCluster],
        current: FailureCluster,
        *,
        limit: int = 100,
    ) -> list[ReplayCase]:
        """Other failures on the same route. A fix for one issue must not
        make a different issue on that route worse."""
        cases: list[ReplayCase] = []
        for cluster in clusters:
            if cluster is current or cluster.route != current.route:
                continue
            cases.extend(self.dataset_builder.build_from_cluster(cluster))
        return cases[:limit]

    # ------------------------------------------------------------------
    # Audit and rollback
    # ------------------------------------------------------------------

    def run_audit(
        self,
        *,
        prune: bool = True,
        prune_below: float = 0.0,
        limit: int = 200,
    ) -> list[AuditFinding]:
        """Measure what each live patch still contributes, by switching it off.

        Every promoted rule, example, and tool is replayed against the
        failures and approved answers of its route twice: with it and
        without it. A patch whose removal does not lower the score is no
        longer earning its place in the prompt (later patches or a
        fine-tune may have made it redundant). With ``prune`` it is rolled
        back, so the prompt does not grow forever.
        """
        failures = [
            row for row in self.memory.failed_interactions(limit=limit * 5) if is_failing(row)
        ]
        patches = [
            patch
            for patch in self.db.list_patch_history(limit=100000, status="promoted")
            if patch["patch_type"] in _AUDITABLE
        ]
        # Most expensive kind first, newest first within a kind, so when two
        # patches are redundant the cheaper and older one is the survivor.
        patches.sort(key=lambda patch: (-_AUDITABLE.index(patch["patch_type"]), -patch["id"]))

        findings: list[AuditFinding] = []
        with_cache: dict[Optional[str], tuple[list[ReplayCase], list[dict[str, Any]]]] = {}
        for patch in patches:
            route = patch["route"]
            if route not in with_cache:
                rows = [row for row in failures if route is None or row["route"] == route][:limit]
                cases = self.dataset_builder.build(rows) + self.dataset_builder.build(
                    self.db.list_wins(route=route, limit=limit)
                )
                with_cache[route] = (cases, self.replay_runner.run(cases))
            cases, with_patch = with_cache[route]

            without_patch = self.replay_runner.run(cases, exclude_patch_ids={patch["id"]})
            evaluation = self.evaluator.evaluate(cases, without_patch, with_patch)
            contribution = evaluation.delta

            if evaluation.scored == 0:
                action = "kept (nothing to measure it on)"
            elif contribution <= prune_below and prune:
                self.promoter.rollback(
                    patch["id"],
                    f"audit: contributes {contribution:+.3f} over {evaluation.scored} cases",
                )
                action = "rolled_back"
                with_cache.clear()
            elif contribution <= prune_below:
                action = "flagged (no measurable contribution)"
            else:
                action = "kept"

            self.db.update_patch_status(
                patch_id=patch["id"],
                status="rolled_back" if action == "rolled_back" else "promoted",
                details={"audit_contribution": contribution, "audit_cases": evaluation.scored},
            )
            findings.append(
                AuditFinding(
                    patch_id=patch["id"],
                    patch_type=patch["patch_type"],
                    route=route,
                    contribution=contribution,
                    cases=evaluation.scored,
                    action=action,
                )
            )
        return findings

    def rollback(self, patch_id: int, *, reason: str = "manual rollback") -> bool:
        """Undo one promoted rule, example, or tool. Recorded in the audit trail."""
        return self.promoter.rollback(patch_id, reason)

    # ------------------------------------------------------------------
    # Self-review
    # ------------------------------------------------------------------

    def self_review(
        self,
        critic: Any,
        *,
        limit: int = 20,
        threshold: float = 0.6,
        margin: float = 0.2,
    ) -> SelfReviewReport:
        """Let a critic grade answers nobody rated and record what it finds.

        See ``erratica.learning.critic`` for how far critic-written
        feedback is trusted.
        """
        return SelfReviewer(self.db, self.feedback).run(
            critic,
            limit=limit,
            threshold=threshold,
            margin=margin,
        )

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def export_training_data(
        self,
        path: str | Path = Path("data") / "erratica_sft.jsonl",
        *,
        exclude_prompts: Iterable[str] = (),
        include_self_generated: bool = False,
    ) -> ExportReport:
        return self.exporter.export(
            path,
            exclude_prompts=exclude_prompts,
            include_self_generated=include_self_generated,
        )

    def run_weight_cycle(
        self,
        trainer: ModelTrainer,
        *,
        dataset_path: str | Path = Path("data") / "erratica_sft.jsonl",
        replay_limit: int = 500,
        include_self_generated: bool = False,
    ) -> WeightCycleReport:
        """Fine-tune on recorded corrections and keep the result only if it
        wins on failures that were held out of the training data, without
        getting worse on approved answers."""
        loop = WeightLoop(
            self.db,
            self.exporter,
            self.dataset_builder,
            self.evaluator,
            self.replay_runner,
            min_delta=self.min_delta,
            regression_tolerance=self.regression_tolerance,
        )
        report, next_model = loop.run(
            self.model,
            trainer,
            dataset_path=dataset_path,
            replay_limit=replay_limit,
            include_self_generated=include_self_generated,
        )
        if next_model is not self.model:
            self.model = next_model
            self.replay_runner.model = next_model
        return report

    def consolidate(
        self,
        *,
        critic: Any = None,
        trainer: Optional[ModelTrainer] = None,
        dataset_path: str | Path = Path("data") / "erratica_sft.jsonl",
    ) -> ConsolidationReport:
        """One offline pass over everything learned: self-review (if a
        critic is given), a learning cycle, an audit, and a weight cycle (if
        a trainer is given). Meant to be run while the model is idle, for
        example from a nightly scheduled task."""
        report = ConsolidationReport()
        if critic is not None:
            report.self_review = self.self_review(critic)
        report.decisions = self.run_learning_cycle()
        report.audit = self.run_audit()
        if trainer is not None:
            report.weights = self.run_weight_cycle(trainer, dataset_path=dataset_path)
        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def model_card(self, *, model_name: str = "local model") -> str:
        return ModelCardBuilder(self.db).build(model_name=model_name)

    def write_model_card(self, path: str | Path, *, model_name: str = "local model") -> Path:
        return ModelCardBuilder(self.db).write(path, model_name=model_name)

    def dashboard(self) -> dict[str, Any]:
        return self.db.get_dashboard_snapshot()

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def _recall(
        self,
        prompt: str,
        route: str,
        *,
        exclude_prompts: Iterable[str] = (),
    ) -> list[dict[str, str]]:
        return self.memory.similar_corrections(
            prompt=prompt,
            route=route,
            scorer=self.evaluator.similarity_scorer,
            exclude_prompts=exclude_prompts,
            sources=None if self.trust_self_review else ("human",),
        )

    def _replay_recall(self, prompt: str, route: str) -> list[dict[str, str]]:
        # A replayed case must never see its own correction.
        return self._recall(prompt, route, exclude_prompts=[prompt])
