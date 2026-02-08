import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for probability calibration.
    Device-safe implementation.
    """

    def __init__(self, temperature=1.5, device="cpu"):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.ones(1, device=device) * temperature
        )

    def forward(self, logits):
        # Ensure temperature is on same device as logits
        if self.temperature.device != logits.device:
            self.temperature.data = self.temperature.data.to(logits.device)

        return logits / self.temperature