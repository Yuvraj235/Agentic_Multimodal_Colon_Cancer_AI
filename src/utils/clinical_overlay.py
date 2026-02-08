import cv2
import numpy as np
import os

class ClinicalOverlayGenerator:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def overlay(
        self,
        image_tensor,
        heatmap,
        roi_metrics,
        save_path="outputs/gradcam/clinical_overlay.png"
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert image tensor → numpy
        image = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        # Resize heatmap
        heatmap_resized = cv2.resize(
            heatmap, self.image_size, interpolation=cv2.INTER_CUBIC
        )
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

        # Clinical text
        quadrant = roi_metrics["localization"]["dominant_quadrant"]
        coverage = roi_metrics["metrics"]["activation_coverage"] * 100

        text = f"ROI: {quadrant.upper()} | Coverage: {coverage:.1f}%"

        cv2.putText(
            overlay,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imwrite(save_path, overlay)
        return save_path