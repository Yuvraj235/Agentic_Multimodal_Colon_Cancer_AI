# src/utils/lesion_semantics.py

class LesionSemanticsEngine:
    """
    Interprets Grad-CAM++ ROI metrics into
    clinically meaningful lesion semantics and risk flags.
    """

    def __init__(self):
        # Thresholds are conservative by design (clinical safety)
        self.high_activation_threshold = 0.40
        self.wide_coverage_threshold = 0.30

    def interpret(self, roi_metrics, localization, predicted_class):
        """
        Args:
            roi_metrics (dict):
                activation_coverage, max_activation, mean_activation
            localization (dict):
                dominant_quadrant, quadrant_distribution
            predicted_class (str)

        Returns:
            dict with semantics + risk flags
        """

        semantics = []
        risk_flags = []

        coverage = roi_metrics.get("activation_coverage", 0.0)
        max_act = roi_metrics.get("max_activation", 0.0)
        dominant_quad = localization.get("dominant_quadrant", "unknown")

        # --------------------------------------
        # SEMANTIC INTERPRETATION
        # --------------------------------------
        if max_act >= self.high_activation_threshold:
            semantics.append(
                "Localized high-intensity activation detected"
            )

        if coverage >= self.wide_coverage_threshold:
            semantics.append(
                "Activation spans a relatively wide tissue area"
            )

        semantics.append(
            f"Primary activation localized in {dominant_quad.replace('_', ' ')} region"
        )

        # --------------------------------------
        # RISK FLAGS (Conservative)
        # --------------------------------------
        if predicted_class == "pathological-findings":
            risk_flags.append("POTENTIAL_LESION")

        if coverage >= self.wide_coverage_threshold:
            risk_flags.append("EXTENDED_TISSUE_INVOLVEMENT")

        if not risk_flags:
            risk_flags.append("LOW_RISK_PATTERN")

        return {
            "semantics": semantics,
            "risk_flags": risk_flags,
            "roi_summary": {
                "coverage": round(float(coverage), 3),
                "max_activation": round(float(max_act), 3),
                "dominant_region": dominant_quad
            }
        }