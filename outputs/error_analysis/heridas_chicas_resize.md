# Hallazgo: heridas chicas/tempranas se pierden por el resize a 512x512

Fecha: 2026-09-27
Datos: evaluación de `unet_pretrained.pt` (40 épocas) y `unet_scratch.pt` (80 épocas)
sobre las 110 imágenes del set de test (split fijo, seed=42).

## Cómo se detectó

Al comparar métrica por imagen entre los dos modelos (`outputs/checkpoints/evaluation_results.csv`
en Drive), se identificaron 4 imágenes donde AMBOS modelos -- arquitecturas distintas,
entrenadas de formas distintas -- fallan de forma marcada (Dice entre 0.08 y 0.57):

| Imagen | Dice pretrained | Dice scratch |
|---|---|---|
| DSC01428.JPG | 0.130 | 0.083 |
| DSC01017.JPG | 0.409 | 0.488 |
| DSC01995.JPG | 0.495 | 0.247 |
| DSC01234.JPG | 0.570 | 0.178 |

Que dos modelos independientes fallen en las mismas imágenes es indicio de que el
problema no es de capacidad del modelo, sino de la información disponible en la
entrada. Se inspeccionaron las 4 imágenes y sus máscaras a mano (correctamente
etiquetadas, sin corrupción) y se midió el tamaño real de la herida en cada una.

## Medición

| Imagen | Píxeles de herida (imagen original, ~4320x3240) | % de la imagen | Equivalente tras resize a 512x512 |
|---|---|---|---|
| DSC01428 | 79.778 | 0.570% | ~39x39 px |
| DSC01234 | 31.576 | 0.226% | ~24x24 px |
| DSC01995 | 2.448 | 0.017% | ~7x7 px |
| DSC01017 | 2.099 | 0.015% | ~6x6 px |

`train.py` y `evaluate.py` redimensionan toda imagen a 512x512 antes de pasarla a
la red (parámetro `--image_size`, default 512). En las dos imágenes más chicas, la
herida completa queda reducida a un parche de ~6-7 píxeles de lado -- por debajo de
lo que cualquier red convolucional puede segmentar de forma confiable. No es que el
modelo "no aprendió a ver" estas heridas: la información se pierde en el
preprocesamiento, antes de que la red la reciba.

## Por qué esto importa para la tesis (no es un caso a descartar)

El anteproyecto plantea como motivación central que las técnicas tradicionales "no
son capaces de segmentar adecuadamente las heridas en los estadios iniciales" de la
infección. Las heridas tempranas son, casi por definición, chicas y de bajo
contraste contra el pelaje -- exactamente el perfil de estas 4 imágenes. Excluirlas
del set de evaluación ocultaría la evidencia del problema que el proyecto busca
resolver, además de ser una exclusión de datos no justificable metodológicamente
(las máscaras son correctas, no hay ningún defecto en los datos). Se decide
mantenerlas en el dataset y documentar esta limitación explícitamente.

## Decisión

Se adopta un enfoque en dos etapas para las heridas chicas: una localización
gruesa (con más contexto de imagen) seguida de un recorte + segmentación fina sobre
esa región (ver plan en `docs/` o el commit correspondiente cuando se implemente).
El cambio de loss function (Focal Tversky, evaluado para las fallas que no son de
tamaño) no ataca esta causa raíz -- no puede recuperar información que el resize ya
destruyó -- por lo que se lo evalúa por separado, no como solución a este hallazgo
puntual.
