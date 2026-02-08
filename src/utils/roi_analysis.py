import numpy as np
import cv2


class ROIAnalyzer:
    """
    Clinical-grade ROI & activation quantification for Grad-CAM / Grad-CAM++
    """

    def __init__(self, image_size=(224, 224), threshold=0.6):
        self.image_size = image_size
        self.threshold = threshold

    def analyze(self, heatmap: np.ndarray):
        """
        Analyze activation heatmap and return ROI metrics + localization
        """

        if heatmap.ndim != 2:
            raise ValueError("Heatmap must be 2D")

        # Normalize heatmap
        heatmap = np.clip(heatmap, 0, 1)

        # Resize to input image size
        heatmap_resized = cv2.resize(
            heatmap,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        # Thresholded ROI mask
        roi_mask = heatmap_resized >= self.threshold

        total_pixels = heatmap_resized.size
        roi_pixels = int(np.sum(roi_mask))

        # -------------------------
        # Metrics (JSON-safe)
        # -------------------------
        activation_coverage = float(round(roi_pixels / total_pixels, 4))
        max_activation = float(round(float(heatmap_resized.max()), 4))
        mean_activation = float(
            round(float(heatmap_resized[roi_mask].mean()), 4)
        ) if roi_pixels > 0 else 0.0

        metrics = {
            "activation_coverage": activation_coverage,
            "max_activation": max_activation,
            "mean_activation": mean_activation
        }

        # -------------------------
        # Localization (Quadrants)
        # -------------------------
        h, w = heatmap_resized.shape
        h_mid, w_mid = h // 2, w // 2

        quadrants = {
            "upper_left": int(np.sum(roi_mask[:h_mid, :w_mid])),
            "upper_right": int(np.sum(roi_mask[:h_mid, w_mid:])),
            "lower_left": int(np.sum(roi_mask[h_mid:, :w_mid])),
            "lower_right": int(np.sum(roi_mask[h_mid:, w_mid:]))
        }

        dominant_quadrant = max(quadrants, key=quadrants.get)

        localization = {
            "dominant_quadrant": dominant_quadrant,
            "quadrant_distribution": quadrants
        }

        return {
            "metrics": metrics,
            "localization": localization,
            "roi_mask": roi_mask  # used internally for overlays
        }