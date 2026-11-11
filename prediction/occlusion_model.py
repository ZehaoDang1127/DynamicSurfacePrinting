"""Causal U-Net and ConvLSTM surface observer with latent-space rollout.

Inputs contain normalized observed heights, a surface-observation mask,
normalized observation age, and a never-observed flag. The decoder's skip
features come from the latest history frame containing observations; no future
surface or generated height is fed back as an observation.

Two ablation switches leave the default model unchanged:

* ``input_channels`` selects which of the four input channels reach the encoder.
  The data still carry all four, and the mask is still used to zero depth
  placeholders and to detect frames without observed cells.
* ``recurrence`` replaces the dynamics/correction pair with one ConvLSTM cell
  (``single``) or two stacked ConvLSTM layers with the same parameter count
  (``stacked``). Both receive the encoded frame when it has observed cells and
  zero input otherwise, as during rollout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ConvLSTMCell


INPUT_CHANNELS = ('depth', 'mask', 'age', 'never')
RECURRENCES = ('dynamics_correction', 'single', 'stacked')


def validate_input_channels(value) -> tuple[str, ...]:
    """Return the selected encoder input channels in canonical order."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError('input_channels must be a list of channel names')
    names = tuple(value)
    if not names or any(name not in INPUT_CHANNELS for name in names) or len(set(names)) != len(names):
        raise ValueError('input_channels must be distinct names from ' + ', '.join(INPUT_CHANNELS))
    if 'depth' not in names:
        raise ValueError('input_channels must include depth')
    return tuple(name for name in INPUT_CHANNELS if name in names)


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
    )


class OcclusionUNetConvLSTM(nn.Module):
    """Estimate the current height and forecast future heights causally.

    ``forward`` returns [B, rollout_steps + 1, 1, H, W]. The first output is
    the current estimate; the remainder are successive latent rollout steps.
    States are local to each call, and fully unobserved frames only propagate
    the state. Spatial skips are held fixed during rollout; consequently this
    is an experimental observer, not an uncertainty estimate.
    """

    def __init__(self, base_channels: int = 16, kernel_size: int = 3,
                 input_channels: tuple[str, ...] = INPUT_CHANNELS,
                 recurrence: str = 'dynamics_correction') -> None:
        super().__init__()
        if type(base_channels) is not int or base_channels < 1:
            raise ValueError('base_channels must be a positive integer')
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('kernel_size must be a positive odd integer')
        input_channels = validate_input_channels(input_channels)
        if recurrence not in RECURRENCES:
            raise ValueError('recurrence must be one of ' + ', '.join(RECURRENCES))
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.recurrence = recurrence
        self._channel_index = [INPUT_CHANNELS.index(name) for name in input_channels]
        channels = base_channels
        self.encoder_high = _block(len(input_channels), channels)
        self.encoder_low = _block(channels, 2 * channels)
        self.encoder_latent = _block(2 * channels, 4 * channels)
        if recurrence == 'dynamics_correction':
            self.dynamics = ConvLSTMCell(4 * channels, 4 * channels, kernel_size)
            self.correction = ConvLSTMCell(4 * channels, 4 * channels, kernel_size)
        elif recurrence == 'single':
            self.recurrent = ConvLSTMCell(4 * channels, 4 * channels, kernel_size)
        else:
            self.recurrent_lower = ConvLSTMCell(4 * channels, 4 * channels, kernel_size)
            self.recurrent_upper = ConvLSTMCell(4 * channels, 4 * channels, kernel_size)
        self.decoder_low = _block(6 * channels, 2 * channels)
        self.decoder_high = _block(3 * channels, channels)
        self.head = nn.Conv2d(channels, 1, 1)

    @staticmethod
    def _downsample(value: Tensor) -> Tensor:
        # Ceil mode keeps even a 1x1 grid valid and preserves odd border cells.
        return F.avg_pool2d(value, kernel_size=2, stride=2, ceil_mode=True)

    def _encode(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # A missing cell's placeholder cannot be interpreted as a height.
        observed = frame[:, 1:2]
        frame = torch.cat((frame[:, :1] * observed, frame[:, 1:]), dim=1)
        if len(self._channel_index) != frame.shape[1]:
            frame = frame[:, self._channel_index]
        high = self.encoder_high(frame)
        low = self.encoder_low(self._downsample(high))
        latent = self.encoder_latent(self._downsample(low))
        return high, low, latent

    def _decode(self, hidden: Tensor, high: Tensor, low: Tensor) -> Tensor:
        x = F.interpolate(hidden, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.decoder_low(torch.cat((x, low), dim=1))
        x = F.interpolate(x, size=high.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.decoder_high(torch.cat((x, high), dim=1)))

    def forward(self, inputs: Tensor, rollout_steps: int) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim != 5 or inputs.shape[2] != 4:
            raise ValueError('inputs must have shape [B, T, 4, H, W]')
        if min(inputs.shape) < 1:
            raise ValueError('batch, history, and spatial dimensions must be nonempty')
        if not torch.is_floating_point(inputs) or not torch.isfinite(inputs).all():
            raise ValueError('inputs must contain finite floating-point values')
        if type(rollout_steps) is not int or rollout_steps < 0:
            raise ValueError('rollout_steps must be a nonnegative integer')
        mask = inputs[:, :, 1:2]
        never = inputs[:, :, 3:4]
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('surface-observation mask must be binary')
        if not torch.all((never == 0) | (never == 1)):
            raise ValueError('never-observed flag must be binary')
        if torch.any(inputs[:, :, 2:3] < 0) or torch.any(inputs[:, :, 2:3] > 1):
            raise ValueError('normalized observation age must be in [0, 1]')

        states = None
        high_skip = low_skip = None
        for frame in inputs.unbind(dim=1):
            high, low, latent = self._encode(frame)
            if states is None:
                hidden = torch.zeros_like(latent)
                states = [(hidden, torch.zeros_like(hidden)) for _ in range(self._state_count)]
                high_skip, low_skip = torch.zeros_like(high), torch.zeros_like(low)
            has_observation = frame[:, 1:2].bool().flatten(1).any(dim=1).view(-1, 1, 1, 1)
            states = self._observe(latent, has_observation, states)
            high_skip = torch.where(has_observation, high, high_skip)
            low_skip = torch.where(has_observation, low, low_skip)

        outputs = [self._decode(states[-1][0], high_skip, low_skip)]
        for _ in range(rollout_steps):
            states = self._propagate(states)
            outputs.append(self._decode(states[-1][0], high_skip, low_skip))
        return torch.stack(outputs, dim=1)

    @property
    def _state_count(self) -> int:
        return 2 if self.recurrence == 'stacked' else 1

    def _observe(self, latent: Tensor, has_observation: Tensor, states: list) -> list:
        """Advance the latent state by one input frame."""
        zeros = torch.zeros_like(latent)
        if self.recurrence == 'dynamics_correction':
            predicted = self.dynamics(zeros, states[0])
            corrected = self.correction(latent, predicted)
            return [tuple(torch.where(has_observation, new, old) for new, old in zip(corrected, predicted))]
        frame_input = torch.where(has_observation, latent, zeros)
        if self.recurrence == 'single':
            return [self.recurrent(frame_input, states[0])]
        lower = self.recurrent_lower(frame_input, states[0])
        return [lower, self.recurrent_upper(lower[0], states[1])]

    def _propagate(self, states: list) -> list:
        """Advance the latent state by one step without an observation."""
        zeros = torch.zeros_like(states[0][0])
        if self.recurrence == 'dynamics_correction':
            return [self.dynamics(zeros, states[0])]
        if self.recurrence == 'single':
            return [self.recurrent(zeros, states[0])]
        lower = self.recurrent_lower(zeros, states[0])
        return [lower, self.recurrent_upper(lower[0], states[1])]
