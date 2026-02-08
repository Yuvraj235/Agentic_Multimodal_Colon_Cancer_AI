# src/agents/clinical_text_agent.py

class ClinicalTextAgent:
    """
    Converts fused multimodal output into
    clinician-readable medical text.
    """

    def __init__(self, institution="Generic Clinical Setting"):
        self.institution = institution

    def generate(self, fusion_output):
        """
        fusion_output: dict from MultimodalFusionAgent
        """

        decision = fusion_output["decision"]
        predicted_class = fusion_output["predicted_class"]
        confidence = fusion_output["confidence"]

        if decision == "CONFIDENT":
            assessment = (
                f"The multimodal AI system confidently identified "
                f"{predicted_class} features in the colonoscopy image."
            )
            recommendation = (
                "Findings may be used as decision support in conjunction "
                "with clinical expertise."
            )
        else:
            assessment = (
                f"The AI system detected features consistent with "
                f"{predicted_class}, however confidence is limited."
            )
            recommendation = (
                "Manual review by a gastroenterologist is strongly recommended."
            )

        return {
            "institution": self.institution,
            "clinical_assessment": assessment,
            "confidence": round(confidence, 4),
            "recommendation": recommendation,
            "safety_note": (
                "This AI output is intended for clinical decision support "
                "and must not replace professional medical judgment."
            )
        }