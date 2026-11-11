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

__version__ = "0.5.0"

__all__ = [
    "AuditFinding",
    "CommandTrainer",
    "ConsolidationReport",
    "Critic",
    "Domain",
    "DomainRegistry",
    "EmbeddingSimilarityScorer",
    "Erratica",
    "ExportReport",
    "GenerationResult",
    "HybridStringSimilarityScorer",
    "MistakeTriage",
    "ModelAdapter",
    "ModelCardBuilder",
    "ModelCritic",
    "ModelTrainer",
    "OllamaEmbeddingScorer",
    "PromotionDecision",
    "RuleAwareDemoModel",
    "SelfReviewReport",
    "TrainingDataExporter",
    "TriageVerdict",
    "WeightCycleReport",
    "WeightLoop",
]
