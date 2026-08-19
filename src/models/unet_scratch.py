"""
unet_scratch.py

U-Net clásica (Ronneberger et al., 2015) implementada desde cero en PyTorch,
sin pesos preentrenados -- esta es la red que se entrena completamente desde
inicialización aleatoria, para comparar contra unet_pretrained.py.

Mantenerla simple y estándar a propósito: la comparación que importa para la
tesis es "preentrenada vs. desde cero", no "arquitectura exótica vs. estándar".
Si después se quiere sumar Attention U-Net / DeepLabV3+ / SegFormer como
arquitecturas adicionales, conviene agregarlas como archivos separados en
este mismo paquete `models/`, siguiendo la misma firma de entrada/salida.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x2, el bloque básico de U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool seguido de DoubleConv -- un escalón de bajada del encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample + concatenación con el feature map del encoder (skip connection) + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_encoder_skip):
        x_decoder = self.up(x_decoder)

        # Si las dimensiones no calzan exacto (tamaños de entrada no potencia de 2), padear.
        diff_y = x_encoder_skip.size(2) - x_decoder.size(2)
        diff_x = x_encoder_skip.size(3) - x_decoder.size(3)
        x_decoder = nn.functional.pad(
            x_decoder, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x_encoder_skip, x_decoder], dim=1)
        return self.conv(x)


class UNetFromScratch(nn.Module):
    """
    U-Net estándar de 4 niveles de profundidad.

    in_channels=3 (RGB). out_channels=1 (máscara binaria, logits sin sigmoid
    -- aplicar sigmoid/BCEWithLogitsLoss afuera, ver train.py).
    base_channels controla el ancho de la red (64 es el valor del paper
    original; bajarlo a 32 si el dataset es chico y hay riesgo de overfitting
    o si la GPU se queda sin memoria).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64,
                 bilinear: bool = True):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNetFromScratch()
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print("Output shape:", out.shape)  # esperado: [2, 1, 512, 512]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {n_params:,}")
