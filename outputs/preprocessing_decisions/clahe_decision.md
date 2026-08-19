# Decisión de preprocesamiento: CLAHE sobre canal L (LAB)

Fecha de esta corrida: 2026-08-19 (ver también fecha del commit de este archivo en git log)
Script: `src/evaluate_preprocessing_effect.py`
Imágenes analizadas: 25 (muestra aleatoria, seed=42, del subconjunto con máscara manual)

## Método

Para cada imagen, se midió la separación (d de Cohen) entre los píxeles de la
región de herida (máscara manual) y un anillo de piel/pelaje sano inmediatamente
circundante, en los canales L, a y b de LAB, antes y después de aplicar CLAHE
(clip_limit=2.5, tile=8x8) sobre el canal L. La comparación entre condiciones
(con/sin CLAHE) se hizo con un test pareado de Wilcoxon sobre el |d| de cada imagen
(cada imagen es una unidad de observación independiente).

## Resultados

| Canal | \|d\| medio sin preprocesar | \|d\| medio con CLAHE | Mejora / Empeora | p (Wilcoxon) |
|---|---|---|---|---|
| L | 1.052 | 1.137 | 14 mejoran / 11 empeoran (n=25) | 0.1199 (sin diferencia significativa) |
| a | 1.045 | 1.036 | 4 mejoran / 21 empeoran (n=25) | 0.0001 (empeora) |
| b | 1.440 | 1.443 | 17 mejoran / 8 empeoran (n=25) | 0.1817 (sin diferencia significativa) |

## Conclusión

El canal **a** (rojo-verde), que es el que más separa herida de piel sana en este dataset incluso sin ningún preprocesamiento (mayor |d| que L y comparable a b), empeora de forma estadísticamente significativa con CLAHE, en la gran mayoría de las imágenes analizadas. El canal L no mejora de forma significativa. **Decisión: no incorporar este CLAHE (aplicado sobre el canal L y reconvertido a BGR) al pipeline de entrenamiento por default.** El daño al canal a parece ser un efecto colateral del ida y vuelta de espacio de color LAB→BGR (no completamente reversible), no una mejora real de ningún canal.

## Nota metodológica

Este análisis es un filtro barato antes de comprometer tiempo de entrenamiento, no un reemplazo del test definitivo. La prueba concluyente es un ablation real: entrenar el mismo modelo con y sin el paso de preprocesamiento, mismo split y semilla, y comparar Dice de validación.
