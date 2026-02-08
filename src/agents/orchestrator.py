import torch

from src.utils.lesion_semantics import LesionSemanticsEngine


class OrchestratorAgent:
    """
    Central controller that coordinates:
    - Vision perception
    - Pathology reasoning
    - Explainability (Grad-CAM++)
    - Lesion semantics & risk flags
    - Clinical text generation (if present)
    """

    def __init__(
        self,
        image_agent,
        pathology_agent,
        explanation_agent=None,
        clinical_text_agent=None,
        device="cpu"
    ):
        self.image_agent = image_agent
        self.pathology_agent = pathology_agent
        self.explanation_agent = explanation_agent
        self.clinical_text_agent = clinical_text_agent
        self.device = device

        # New: lesion semantics engine
        self.lesion_engine = LesionSemanticsEngine()

    @torch.no_grad()
    def run(self, image_tensor):
        """
        Execute full agentic pipeline on a single image tensor.
        """

        image_tensor = image_tensor.to(self.device)

        # --------------------------------------------------
        # 1. PERCEPTION (Vision Agent)
        # --------------------------------------------------
        perception = self.image_agent.perceive(image_tensor)

        # Expected:
        # perception = {
        #   "features": Tensor[B, 768],
        #   "feature_shape": (...)
        # }

        # --------------------------------------------------
        # 2. REASONING (Pathology Agent)
        # --------------------------------------------------
        reasoning = self.pathology_agent.reason(perception)

        # Expected:
        # reasoning = {
        #   "predicted_class": str,
        #   "class_index": int,
        #   "confidence": float,
        #   "probabilities": dict
        # }

        # --------------------------------------------------
        # 3. EXPLANATION (Grad-CAM++)
        # --------------------------------------------------
        explanation = None
        roi_metrics = None
        localization = None

        if self.explanation_agent is not None:
            explanation = self.explanation_agent.explain(
                image_tensor=image_tensor,
                model=self.image_agent.model,
                class_idx=reasoning["class_index"],
                confidence=reasoning["confidence"]
            )

            # explanation must include ROI outputs
            roi_metrics = explanation.get("roi_metrics")
            localization = explanation.get("localization")

        # --------------------------------------------------
        # 4. LESION SEMANTICS & RISK FLAGS (NEW)
        # --------------------------------------------------
        lesion_analysis = None

        if roi_metrics is not None and localization is not None:
            lesion_analysis = self.lesion_engine.interpret(
                roi_metrics=roi_metrics,
                localization=localization,
                predicted_class=reasoning["predicted_class"]
            )

        # --------------------------------------------------
        # 5. CLINICAL TEXT (Optional)
        # --------------------------------------------------
        clinical_text = None

        if self.clinical_text_agent is not None:
            clinical_text = self.clinical_text_agent.generate(
                reasoning=reasoning,
                lesion_analysis=lesion_analysis
            )

        # --------------------------------------------------
        # 6. FINAL AGENTIC OUTPUT
        # --------------------------------------------------
        return {
            "perception": {
                "feature_shape": perception["features"].shape
            },
            "reasoning": reasoning,
            "explanation": explanation,
            "lesion_semantics": lesion_analysis,
            "clinical_text": clinical_text
        }