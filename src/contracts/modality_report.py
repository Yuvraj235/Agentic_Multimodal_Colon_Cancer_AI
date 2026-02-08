# src/contracts/modality_report.py

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ModalityReport:
    """
    Unified output format for ALL agents (vision, text, biospecimen, etc.)
    """

    modality: str  # e.g. "vision", "clinical_text", "biospecimen"

    evidence: Dict[str, Any]
    confidence: float

    explanation: Dict[str, Any] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)

    def is_confident(self, threshold: float = 0.8) -> bool:
        return self.confidence >= threshold

    def is_uncertain(self, threshold: float = 0.8) -> bool:
        return self.confidence < threshold