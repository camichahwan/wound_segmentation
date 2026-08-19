"""
unet_pretrained.py

U-Net con encoder preentrenado en ImageNet, usando la librería
`segmentation_models_pytorch` (smp) en vez de reimplementar la carga de pesos
a mano -- es la forma estándar de la comunidad de hacerlo y evita errores
sutiles al portar pesos de un backbone preentrenado.

Esta es la contraparte de unet_scratch.py: misma tarea (segmentación binaria
herida/no-herida), misma firma de entrada/salida (logits sin sigmoid), para
que train.py y evaluate.py puedan tratar a los dos modelos de forma
intercambiable y la comparación "preentrenada vs. desde cero" sea limpia.

Instalar con: pip install segmentation-models-pytorch
"""

import segmentation_models_pytorch as smp


def build_pretrained_unet(encoder_name: str = "resnet34", encoder_weights: str = "imagenet",
                           in_channels: int = 3, out_channels: int = 1,
                           freeze_encoder: bool = False):
    """
    encoder_name: cualquier backbone soportado por smp (resnet34, resnet50,
    efficientnet-b0, etc.). resnet34 es un buen punto de partida: liviano,
    rápido de entrenar en una RTX 3060 de 12GB, con pesos ImageNet bien
    probados.

    freeze_encoder: si True, congela los pesos del encoder preentrenado y
    solo entrena el decoder -- útil como primer experimento rápido, o si el
    dataset etiquetado (500 imágenes) es chico y hay riesgo de overfitting
    ajustando todo el encoder. Se puede descongelar después de unas épocas
    (fine-tuning progresivo) modificando el flag y recargando el checkpoint.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,  # logits crudos, igual que UNetFromScratch
    )

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


if __name__ == "__main__":
    import torch

    model = build_pretrained_unet()
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print("Output shape:", out.shape)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {n_total:,} | entrenables: {n_trainable:,}")
