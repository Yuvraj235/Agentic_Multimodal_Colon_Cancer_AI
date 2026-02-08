class ClinicalExplanationAgent:
    """
    Converts model predictions into clinician-friendly explanations.
    """

    def __init__(self, safety_threshold=0.80):
        self.safety_threshold = safety_threshold

    def generate(self, reasoning_output):
        """
        reasoning_output: dict from PathologyReasoningAgent
        """

        predicted_class = reasoning_output["predicted_class"]
        confidence = reasoning_output["confidence"]
        probs = reasoning_output["probabilities"]

        # -----------------------------
        # Base explanation templates
        # -----------------------------
        class_explanations = {
            "anatomical-landmarks": (
                "The model identified normal anatomical structures of the colon "
                "without visual evidence of abnormal growth or lesions."
            ),
            "pathological-findings": (
                "The model detected visual patterns consistent with pathological "
                "findings, which may indicate abnormal tissue or lesion presence."
            ),
            "quality-of-mucosal-views": (
                "The model assessed the image quality and found suboptimal mucosal "
                "visualization, which may limit diagnostic reliability."
            ),
            "therapeutic-interventions": (
                "The model identified features consistent with therapeutic "
                "interventions such as clips or surgical tools."
            ),
        }

        explanation = class_explanations.get(
            predicted_class,
            "The model generated a prediction based on visual analysis."
        )

        # -----------------------------
        # Confidence-aware safety logic
        # -----------------------------
        if confidence >= self.safety_threshold:
            decision = "CONFIDENT"
            recommendation = (
                "The confidence level is high. The automated assessment can be "
                "considered reliable within clinical context."
            )
        else:
            decision = "UNCERTAIN"
            recommendation = (
                "The confidence level is moderate. Manual review by a "
                "gastroenterologist is recommended before clinical decision-making."
            )

        # -----------------------------
        # Final structured output
        # -----------------------------
        return {
            "decision": decision,
            "clinical_explanation": explanation,
            "confidence": confidence,
            "recommendation": recommendation,
            "class_probabilities": probs
        }