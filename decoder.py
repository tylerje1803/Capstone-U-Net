from collections.abc import Sequence

import torch
from torch import nn


def upsample_transpose_conv(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
    """Create a transposed convolution that doubles spatial size and halves channels."""
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
    )


def skip_concat(upsampled: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
    """Concatenate decoder and encoder features along the channel dimension."""
    if upsampled.shape[2:] != encoder_features.shape[2:]:
        raise ValueError(
            "Upsampled and encoder features must have matching spatial dimensions; "
            f"got {tuple(upsampled.shape[2:])} and {tuple(encoder_features.shape[2:])}."
        )
    return torch.cat((upsampled, encoder_features), dim=1)


def decoder_stage(
    inputs: torch.Tensor,
    encoder_features: torch.Tensor,
    upsampler: nn.Module,
    convolutions: nn.Module,
) -> torch.Tensor:
    """Upsample, concatenate the matching skip, then refine the features."""
    upsampled = upsampler(inputs)
    combined = skip_concat(upsampled, encoder_features)
    return convolutions(combined)


def decoder_forward(
    bottleneck: torch.Tensor,
    skips: Sequence[torch.Tensor],
    stages: Sequence[nn.Module],
) -> list[torch.Tensor]:
    """Run stages from deepest to shallowest and collect each resolution's output."""
    if len(skips) != len(stages):
        raise ValueError(f"Expected {len(stages)} skip tensors, got {len(skips)}.")

    outputs = []
    features = bottleneck
    for stage, encoder_features in zip(stages, reversed(skips)):
        features = stage(features, encoder_features)
        outputs.append(features)
    return outputs


class DecoderStage(nn.Module):
    """One U-Net upsample, skip-concatenation, and double-convolution stage."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsampler = upsample_transpose_conv(in_channels, out_channels)
        self.convolutions = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        return decoder_stage(inputs, encoder_features, self.upsampler, self.convolutions)


class Decoder(nn.Module):
    """U-Net decoder; channel lists are ordered deepest skip to shallowest skip."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        out_channels: Sequence[int],
    ) -> None:
        super().__init__()
        if len(skip_channels) != len(out_channels):
            raise ValueError("skip_channels and out_channels must have the same length.")

        stages = []
        for stage_in, skip_count, stage_out in zip(
            [in_channels, *out_channels[:-1]], skip_channels, out_channels
        ):
            stages.append(DecoderStage(stage_in, skip_count, stage_out))
        self.stages = nn.ModuleList(stages)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Accept encoder skips in shallow-to-deep order and return stage outputs."""
        return decoder_forward(bottleneck, skips, self.stages)