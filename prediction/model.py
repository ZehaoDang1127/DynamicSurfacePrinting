"""A compact convolutional recurrent model for fixed-grid depth forecasts.

This uses the common ConvLSTM variant without peephole connections, rather
than reproducing every gate term of Shi et al. (2015). Each layer has one
padded convolution for its four gates. A linear 1x1 head reads the final
hidden state. Every forward call starts from zero states, so independent
history windows cannot leak information into one another.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import PredictionConfig


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, frame: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        hidden, memory = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((frame, hidden), dim=1)
        ).chunk(4, dim=1)
        memory = torch.sigmoid(forget_gate) * memory + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(memory)
        return hidden, memory


class ConvLSTMPredictor(nn.Module):
    """Map [batch, time, depth+mask, height, width] to one normalized map."""

    def __init__(self, config: PredictionConfig) -> None:
        super().__init__()
        self.config = config
        channels = (2,) + config.hidden_channels
        self.cells = nn.ModuleList(
            ConvLSTMCell(channels[index], channels[index + 1], config.kernel_size)
            for index in range(len(config.hidden_channels))
        )
        self.head = nn.Conv2d(config.hidden_channels[-1], 1, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[2] != 2:
            raise ValueError('inputs must have shape [B, T, 2, H, W]')
        batch, timesteps, _, height, width = inputs.shape
        if timesteps != self.config.history:
            raise ValueError(f'expected {self.config.history} history frames, received {timesteps}')
        if min(batch, height, width) < 1:
            raise ValueError('batch and spatial dimensions must be nonempty')
        if not torch.is_floating_point(inputs):
            raise ValueError('inputs must be floating point normalized depth and validity masks')
        states = [
            (
                inputs.new_zeros(batch, cell.hidden_channels, height, width),
                inputs.new_zeros(batch, cell.hidden_channels, height, width),
            )
            for cell in self.cells
        ]
        for timestep in range(timesteps):
            features = inputs[:, timestep]
            for index, cell in enumerate(self.cells):
                states[index] = cell(features, states[index])
                features = states[index][0]
        return self.head(features)


def masked_mae(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Average only selected pixels; invalid NaNs never enter arithmetic.

    An empty mask returns a differentiable zero; training additionally skips
    that batch so Adam's momentum cannot move weights without observations.
    Nonfinite predictions or targets at valid pixels are errors.
    """
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError('prediction, target, and valid must have matching shapes')
    selected = valid.bool()
    if not selected.any():
        return prediction.reshape(-1)[:0].sum()
    observed_prediction = prediction[selected]
    observed_target = target[selected]
    if not torch.isfinite(observed_prediction).all():
        raise ValueError('prediction contains nonfinite values at valid target pixels')
    if not torch.isfinite(observed_target).all():
        raise ValueError('target contains nonfinite values at valid pixels')
    return (observed_prediction - observed_target).abs().mean()
