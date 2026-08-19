"""
preprocessing.py

Acá va SOLO preprocesamiento que alimenta a la red (transforma la imagen para
que la CNN la vea mejor), no intentos de segmentar por afuera del modelo.
Vamos a ir agregando funciones de a una, probándolas antes de sumar la
siguiente.

Paso 1: normalización de iluminación con CLAHE.
"""

import cv2
import numpy as np


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8) -> np.ndarray:
    """
    Ecualización de histograma adaptativa con límite de contraste (CLAHE),
    aplicada solo al canal L de LAB (luminosidad), no a los canales de color.

    Por qué esto y no ecualización global: tus fotos tienen iluminación muy
    despareja dentro del mismo cuadro (flash directo sobre el animal, zonas de
    sombra en la jaula, papel casi saturado de blanco). Una ecualización
    global mueve el contraste de toda la imagen por igual y no arregla eso.
    CLAHE trabaja en mosaicos (tiles) locales, así que mejora el contraste
    dentro de zonas oscuras y dentro de zonas claras por separado, sin que el
    papel sobreexpuesto arrastre el ajuste de toda la imagen.

    Se aplica solo sobre L y no sobre a/b para no distorsionar el color (la
    diferencia de tono entre tejido de herida y piel/pelaje es información que
    no queremos tocar, solo queremos mejorar el contraste de brillo).

    clip_limit: cuánto se permite estirar el contraste por tile antes de
    recortar (valores más altos = más contraste, pero más ruido amplificado).
    tile_grid_size: tamaño de la grilla de mosaicos (8x8 es el valor típico).

    Devuelve una imagen BGR del mismo tamaño, lista para seguir el pipeline
    normal (resize, normalize, ToTensor, etc. en dataset.py).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    img_eq_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_eq_bgr
