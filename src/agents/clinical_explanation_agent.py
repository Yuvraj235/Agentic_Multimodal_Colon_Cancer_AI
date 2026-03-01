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
            "polyps": (
                "The model detected visual patterns consistent with colonic polyps. "
                "Polypoid structures with irregular mucosal surface were identified, "
                "indicating increased colorectal cancer risk."
            ),
            "uc-mild": (
                "The model identified mild ulcerative colitis features: mucosal erythema "
                "and loss of vascular pattern without deep ulceration."
            ),
            "uc-moderate-sev": (
                "The model detected moderate-to-severe ulcerative colitis: extensive ulceration, "
                "friability, and mucosal loss — indicating elevated malignancy risk."
            ),
            "barretts-esoph": (
                "The model identified upper GI pathology consistent with Barrett's esophagus "
                "or esophagitis: salmon-pink intestinal metaplasia or mucosal erosions."
            ),
            "therapeutic": (
                "The model identified post-polypectomy therapeutic intervention: dyed resection "
                "sites indicating prior therapeutic endoscopic procedure."
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