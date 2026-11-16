from erratica.core import (
    Case,
    CertifiedGate,
    Change,
    Loop,
    MemoryStore,
    ReplayGate,
    Run,
    SqliteStore,
    Trial,
    Verdict,
    all_of,
    checks,
    from_adapter,
    ollama,
    prompted,
    proposers,
)
from erratica.domains import Domain, DomainRegistry
from erratica.engine.runner import (
    AuditFinding,
    ConsolidationReport,
    Erratica,
    GenerationResult,
    ModelAdapter,
    RuleAwareDemoModel,
)
from erratica.learning.critic import Critic, ModelCritic, SelfReviewReport
from erratica.learning.egate import EGate, expected_samples, verify_certificate, verify_chain
from erratica.learning.evalstream import Behaviour, Certifier, GeneratedStream, PoolStream, certify
from erratica.learning.exporter import ExportReport, TrainingDataExporter
from erratica.learning.modelcard import ModelCardBuilder
from erratica.learning.promoter import PromotionDecision
from erratica.learning.similarity import (
    EmbeddingSimilarityScorer,
    HybridStringSimilarityScorer,
    OllamaEmbeddingScorer,
)
from erratica.learning.triage import MistakeTriage, TriageVerdict
from erratica.learning.weights import CommandTrainer, ModelTrainer, WeightCycleReport, WeightLoop

__version__ = "0.6.0"

__all__ = [
    "all_of",
    "AuditFinding",
    "Behaviour",
    "Case",
    "CertifiedGate",
    "Certifier",
    "certify",
    "Change",
    "checks",
    "CommandTrainer",
    "ConsolidationReport",
    "Critic",
    "Domain",
    "DomainRegistry",
    "EGate",
    "EmbeddingSimilarityScorer",
    "Erratica",
    "expected_samples",
    "ExportReport",
    "from_adapter",
    "GeneratedStream",
    "GenerationResult",
    "HybridStringSimilarityScorer",
    "Loop",
    "MemoryStore",
    "MistakeTriage",
    "ModelAdapter",
    "ModelCardBuilder",
    "ModelCritic",
    "ModelTrainer",
    "ollama",
    "OllamaEmbeddingScorer",
    "PoolStream",
    "PromotionDecision",
    "prompted",
    "proposers",
    "ReplayGate",
    "RuleAwareDemoModel",
    "Run",
    "SelfReviewReport",
    "SqliteStore",
    "TrainingDataExporter",
    "TriageVerdict",
    "Trial",
    "Verdict",
    "verify_certificate",
    "verify_chain",
    "WeightCycleReport",
    "WeightLoop",
]
