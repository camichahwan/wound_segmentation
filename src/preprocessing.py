"""
preprocessing.py

Funciones de preprocesamiento clásico para el pipeline de segmentación de heridas.

Estas funciones cumplen DOS roles distintos en la tesis, y conviene tener claro
cuál es cuál cuando se documenta el trabajo:

1. Como ayuda para la anotación manual (semi-automatic pre-labeling):
   generar una máscara candidata rápida que después se corrige a mano en vez de
   dibujar el contorno desde cero. Esto es legítimo y vale la pena mencionarlo
   en la tesis como parte de la metodología de construcción del dataset.

2. Como baseline clásico de comparación contra los modelos de deep learning:
   el propio anteproyecto ya critica los métodos geométricos/manuales
   tradicionales (ej. ajuste de elipses) por su imprecisión. Reportar cómo
   les va a estos métodos clásicos frente al ground truth (y frente a la red)
   es exactamente el tipo de comparación que un jurado espera ver.

No hace falta "hacer de cuenta" que se usaron para otra cosa: corriendo estas
funciones sobre el dataset real y reportando sus métricas ya es un aporte
metodológico legítimo, aunque sea modesto.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR (formato nativo de OpenCV) a espacio de color CIELAB.

    Por qué LAB y no RGB/HSV acá:
    - El canal L (luminosidad) queda separado de la información de color, lo que
      ayuda a que el resto del pipeline sea más robusto a la iluminación despareja
      que se ve en las fotos del bioterio (flash directo, sombras de la mano).
    - El canal 'a' codifica el eje verde-rojo. El tejido de herida/inflamación
      suele tender a tonos más rojizos que el pelaje circundante, así que 'a'
      suele separar mejor herida de fondo que cualquier canal de RGB por separado.

    Devuelve la imagen LAB con los 3 canales en el rango que usa OpenCV
    (L en [0,255], a y b centrados en 128).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def canny_on_red_channel(img_bgr: np.ndarray, low: int = 50, high: int = 150,
                          blur_ksize: int = 5) -> np.ndarray:
    """
    Aplica detección de bordes de Canny sobre el canal rojo de la imagen (no
    sobre la versión en escala de grises, que promedia los 3 canales y diluye
    la señal de color que distingue tejido de herida).

    Se aplica un desenfoque gaussiano leve antes de Canny para no generar
    bordes espurios por el ruido/textura del pelaje.

    Devuelve un mapa binario de bordes (0/255), del mismo tamaño que la imagen
    de entrada. Este mapa de bordes NO es una máscara de segmentación por sí
    solo: típicamente hay que cerrar contornos (morfología) y quedarse con el
    contorno más grande/central para obtener una región candidata.
    """
    red_channel = img_bgr[:, :, 2]  # BGR -> índice 2 es el canal rojo
    red_blur = cv2.GaussianBlur(red_channel, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(red_blur, low, high)
    return edges


def edges_to_candidate_mask(edges: np.ndarray, min_area_frac: float = 0.001) -> np.ndarray:
    """
    Convierte un mapa de bordes (salida de canny_on_red_channel) en una máscara
    binaria rellena, quedándose con el contorno cerrado más grande que supere
    un área mínima (para descartar ruido).

    Útil para pasar de "bordes" a algo comparable con la máscara ground truth
    (que es una región rellena, no un contorno).
    """
    h, w = edges.shape
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)
    if not contours:
        return mask

    min_area = min_area_frac * h * w
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return mask

    largest = max(valid, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def estimate_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aísla al animal del fondo ANTES de buscar la herida por color.

    PRIMER INTENTO (documentado acá a propósito, porque el resultado en sí es
    un hallazgo útil para la tesis): un umbral doble sobre el canal L de LAB
    (descartar muy claro = papel, muy oscuro = jaula/sombra, quedarse con la
    componente conexa más grande de lo intermedio) FALLA en este dataset,
    porque el fondo de jaula oscuro es más grande en área que el animal y
    termina ganando la selección por "componente conexa más grande". Ver
    outputs/figures/preprocessing_demo.png, panel "ROI del animal" en la
    primera corrida: selecciona la jaula, no la rata.

    SEGUNDO INTENTO (el que queda implementado): en casi todas las fotos hay
    una hoja de papel blanca de respaldo sobre la que se apoya al animal
    (protocolo de fotografía del bioterio). En vez de buscar directamente al
    animal, se busca el papel -- que es una región clara, grande y de forma
    aproximadamente convexa -- y se aprovecha que el animal (y la mano que lo
    sostiene) tapan una parte de esa hoja. El "agujero" que el animal genera
    sobre el rectángulo de papel, calculado como
    (envolvente convexa del papel) menos (región de papel detectada),
    aproxima razonablemente al animal sin necesitar un detector entrenado.

    Esta heurística asume que la hoja de papel está presente y mayormente
    visible en la foto. Si en algún subconjunto de imágenes no hay papel de
    fondo, esta función no va a funcionar y hay que decirlo explícitamente en
    vez de reportar un número que no es real.

    Devuelve una máscara binaria (0/255) del animal (aproximada, no perfecta).
    """
    lab = to_lab(img_bgr)
    l_ch = lab[:, :, 0]

    # El papel es la región más clara y grande de la imagen.
    paper_thresh = np.percentile(l_ch, 80)
    paper = np.where(l_ch > paper_thresh, 255, 0).astype(np.uint8)
    paper = clean_mask_morphology(paper, kernel_size=9)
    paper = largest_connected_component(paper)

    if paper.sum() == 0:
        # No se detectó nada parecido a una hoja de papel: no forzar un
        # resultado inventado, devolver todo el frame como ROI (equivale a
        # "no se pudo aislar el fondo, usar la imagen completa").
        return np.full(l_ch.shape, 255, dtype=np.uint8)

    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    hull_mask = np.zeros_like(paper)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Animal (+ mano) = lo que "falta" del papel dentro de su propia envolvente convexa.
    fg = cv2.bitwise_and(hull_mask, cv2.bitwise_not(paper))
    fg = clean_mask_morphology(fg, kernel_size=9)
    fg = largest_connected_component(fg)
    return fg


def kmeans_segment(img_bgr: np.ndarray, k: int = 3, color_space: str = "lab",
                    random_state: int = 42, roi_mask: np.ndarray = None) -> np.ndarray:
    """
    Segmentación no supervisada por K-means sobre los píxeles de la imagen,
    agrupando colores en k clusters (k=3 es lo que mencionó el tutor: en la
    práctica suele separar algo como {pelaje/piel sana, herida/tejido rojizo,
    fondo (jaula/papel/guante)}, aunque el mapeo cluster->clase no es fijo y
    hay que decidirlo por imagen o por heurística de color).

    Devuelve un mapa de etiquetas (valores 0..k-1 dentro de la ROI, -1 fuera de
    ella si se pasa roi_mask), NO una máscara binaria. Para convertirlo en
    máscara de herida hace falta elegir qué cluster corresponde a la herida
    (ver `select_wound_cluster`).

    roi_mask: máscara binaria (0/255) opcional -- típicamente la salida de
    `estimate_foreground_mask` -- para clusterizar SOLO los píxeles del animal
    y no dejar que el fondo (jaula, papel, sombra) compita por un cluster.
    Muy recomendable usarla: sin ROI, k=3 tiende a separar fondo-claro /
    fondo-oscuro / animal en vez de separar zonas dentro del animal.
    """
    if color_space == "lab":
        img_conv = to_lab(img_bgr)
    elif color_space == "rgb":
        img_conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_conv = img_bgr

    h, w = img_conv.shape[:2]
    flat = img_conv.reshape(-1, 3).astype(np.float32)

    if roi_mask is not None:
        roi_flat = (roi_mask.reshape(-1) > 0)
        pixels = flat[roi_flat]
    else:
        roi_flat = np.ones(h * w, dtype=bool)
        pixels = flat

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    fitted_labels = km.fit_predict(pixels)

    labels = np.full(h * w, -1, dtype=np.int32)
    labels[roi_flat] = fitted_labels
    return labels.reshape(h, w), km.cluster_centers_


def select_wound_cluster(labels: np.ndarray, cluster_centers: np.ndarray,
                          color_space: str = "lab") -> np.ndarray:
    """
    Heurística simple para decidir qué cluster de kmeans_segment corresponde a
    la herida: en espacio LAB, se elige el cluster cuyo centro tiene el mayor
    valor en el canal 'a' (más rojo), que es la hipótesis de color detrás de
    "herida = tejido más rojo que el resto".

    Esto es una heurística, no una regla infalible (puede fallar en heridas
    muy pálidas o necróticas oscuras) — vale la pena decir esto explícitamente
    en la tesis en vez de presentarlo como si funcionara siempre.

    Devuelve una máscara binaria (0/255).
    """
    if color_space == "lab":
        a_channel_idx = 1
        wound_cluster = int(np.argmax(cluster_centers[:, a_channel_idx]))
    else:
        # fallback: cluster con mayor componente roja
        wound_cluster = int(np.argmax(cluster_centers[:, 0]))

    mask = np.where(labels == wound_cluster, 255, 0).astype(np.uint8)
    return mask


def otsu_threshold_on_a_channel(img_bgr: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
    """
    Baseline clásico adicional, más simple que k-means: umbralado automático
    de Otsu sobre el canal 'a' de LAB (el mismo canal que motiva el heurístico
    de kmeans). Rápido, determinístico, sin hiperparámetros que ajustar a mano
    — buen "piso" de comparación.

    Igual que en kmeans_segment, conviene pasar roi_mask (salida de
    estimate_foreground_mask) para que Otsu no termine separando fondo-claro
    de fondo-oscuro en vez de piel de herida.
    """
    lab = to_lab(img_bgr)
    a_channel = lab[:, :, 1]

    if roi_mask is not None:
        values = a_channel[roi_mask > 0]
        if values.size == 0:
            return np.zeros(a_channel.shape, dtype=np.uint8)
        thresh_val, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.where((a_channel > thresh_val) & (roi_mask > 0), 255, 0).astype(np.uint8)
        return mask

    _, mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def clean_mask_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Limpieza morfológica estándar (opening + closing) para sacar ruido tipo
    "sal y pimienta" y rellenar huecos chicos, común a aplicar después de
    cualquiera de los métodos clásicos de arriba antes de comparar contra el
    ground truth.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Se queda solo con la componente conexa más grande de una máscara binaria.
    Útil después de kmeans/otsu, que suelen dejar clusters de color dispersos
    por toda la imagen (ej. reflejos en la jaula con el mismo tono que la herida).
    """
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # label 0 es el fondo; buscamos el área más grande entre el resto
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return np.where(labels_im == largest_label, 255, 0).astype(np.uint8)
