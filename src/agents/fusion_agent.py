# src/agents/fusion_agent.py

class MultimodalFusionAgent:
    """
    Arbitration layer that fuses multiple ModalityReports
    into a single clinical decision.
    """

    def __init__(self, confidence_threshold=0.80):
        self.confidence_threshold = confidence_threshold

    def fuse(self, reports):
        """
        reports: List[ModalityReport]
        """

        if not reports:
            raise ValueError("No modality reports provided")

        # Sort by confidence (descending)
        reports = sorted(reports, key=lambda r: r.confidence, reverse=True)

        primary = reports[0]
        supporting = reports[1:]

        # Safety logic
        if primary.is_confident(self.confidence_threshold):
            decision = "CONFIDENT"
        else:
            decision = "UNCERTAIN"

        contradictions = self._detect_conflicts(reports)

        return {
            "decision": decision,
            "primary_modality": primary.modality,
            "confidence": primary.confidence,
            "predicted_class": primary.evidence.get("predicted_class"),
            "supporting_modalities": [r.modality for r in supporting],
            "contradictions": contradictions,
            "all_reports": reports
        }

    def _detect_conflicts(self, reports):
        """
        Flags contradictory predictions across modalities.
        """
        labels = set()

        for r in reports:
            label = r.evidence.get("predicted_class")
            if label:
                labels.add(label)

        if len(labels) > 1:
            return {
                "conflict": True,
                "labels": list(labels)
            }

        return {"conflict": False}