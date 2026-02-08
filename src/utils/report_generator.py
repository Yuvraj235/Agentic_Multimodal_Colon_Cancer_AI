from datetime import datetime


class ClinicalReportGenerator:
    """
    Converts agentic outputs into a formal clinical report.
    """

    def generate(self, output):
        reasoning = output["REASONING"]
        clinical = output["CLINICAL_TEXT"]

        report = f"""
CLINICAL AI ANALYSIS REPORT
----------------------------------
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTED CATEGORY:
- {reasoning["predicted_class"]}

MODEL CONFIDENCE:
- {reasoning["confidence"]:.2f}

CLINICAL INTERPRETATION:
{clinical["clinical_explanation"]}

SAFETY DECISION:
- {clinical["decision"]}

RECOMMENDATION:
{clinical["recommendation"]}

CLASS PROBABILITIES:
"""
        for cls, prob in reasoning["probabilities"].items():
            report += f"- {cls}: {prob:.3f}\n"

        report += """
DISCLAIMER:
This AI-generated report is intended to assist clinical decision-making.
Final diagnosis must be confirmed by a qualified medical professional.
"""

        return report.strip()