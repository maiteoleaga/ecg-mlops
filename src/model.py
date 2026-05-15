"""Arquitectura InceptionTime para clasificación de series temporales 1D."""
import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """Un bloque Inception 1D.

    Aplica en paralelo:
      - 3 convoluciones con kernels de distintos tamaños sobre un bottleneck.
      - Una rama MaxPool + Conv 1x1.

    Las 4 salidas se concatenan en la dimensión de canales.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: list = [10, 20, 40],
        bottleneck_channels: int = 32,
    ):
        super().__init__()

        # Bottleneck inicial: reduce el nº de canales antes de las convs paralelas
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )

        # 3 convoluciones en paralelo con kernels distintos
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    n_filters,
                    kernel_size=k,
                    padding="same",
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        # Rama paralela: MaxPool + Conv 1x1
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(bottleneck_channels, n_filters, kernel_size=1, bias=False),
        )

        # BatchNorm sobre la concatenación (4 ramas * n_filters)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        conv_outputs = [conv(x_bottleneck) for conv in self.convs]
        pool_output = self.maxpool_conv(x_bottleneck)
        out = torch.cat([pool_output] + conv_outputs, dim=1)
        out = self.relu(self.bn(out))
        return out


class InceptionTime(nn.Module):
    """Red InceptionTime: una pila de N bloques Inception + GAP + clasificador."""

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 1,
        n_filters: int = 32,
        kernel_sizes: list = [10, 20, 40],
        bottleneck_channels: int = 32,
        n_inception_blocks: int = 6,
    ):
        super().__init__()

        out_channels = n_filters * (len(kernel_sizes) + 1)  # 4 ramas concatenadas

        # Primer bloque recibe `in_channels` (=1 para ECG univariante);
        # los demás reciben `out_channels` (la salida del bloque previo).
        blocks = [
            InceptionModule(in_channels, n_filters, kernel_sizes, bottleneck_channels)
        ]
        for _ in range(n_inception_blocks - 1):
            blocks.append(
                InceptionModule(
                    out_channels, n_filters, kernel_sizes, bottleneck_channels
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Global Average Pooling + clasificador
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


def build_model_from_config(cfg: dict) -> InceptionTime:
    """Construye un InceptionTime usando la sección 'model' del YAML."""
    model_cfg = cfg["model"]
    return InceptionTime(
        n_classes=cfg["data"]["n_classes"],
        in_channels=1,
        n_filters=model_cfg["n_filters"],
        kernel_sizes=model_cfg["kernel_sizes"],
        bottleneck_channels=model_cfg["bottleneck_channels"],
        n_inception_blocks=model_cfg["n_inception_blocks"],
    )