# Segmentación de heridas infecciosas subcutáneas en ratas — PFC Bioingeniería

Código base para el proyecto final de carrera. Estructura pensada para que el
mismo código corra sin cambios ni en Google Colab ni en VS Code local, y para
que las imágenes (en Google Drive) nunca tengan que subirse a GitHub.

Repo: https://github.com/camichahwan/wound_segmentation

## Estructura del repo

```
wound_segmentation/
├── data/
│   └── samples/                     <- 1 imagen + 1 máscara de ejemplo (SÍ va al repo,
│                                        para poder probar el código sin acceso a Drive)
├── src/
│   ├── config.py                     <- resuelve la ruta a "Tesis Imagenes" según el entorno
│   ├── dataset.py                    <- empareja Images/ con Masks/, separa etiquetadas/sin etiquetar
│   ├── preprocessing.py              <- preprocesamiento que alimenta a la red (por ahora: CLAHE)
│   ├── demo_preprocessing.py         <- visualiza el efecto de preprocessing.py en 1 imagen
│   ├── evaluate_preprocessing_effect.py <- mide con evidencia (d de Cohen + Wilcoxon) si un paso
│   │                                     de preprocesamiento ayuda o no, antes de sumarlo al pipeline
│   ├── metrics.py                    <- Dice, IoU, Precision/Recall, Hausdorff, error de área
│   ├── models/
│   │   ├── unet_scratch.py            <- U-Net implementada desde cero
│   │   └── unet_pretrained.py         <- U-Net con encoder preentrenado (segmentation_models_pytorch)
│   ├── train.py                       <- entrena cualquiera de los dos modelos (o hace fine-tuning
│   │                                     de un checkpoint existente, con --init_checkpoint)
│   ├── evaluate.py                    <- compara checkpoints entrenados sobre el set de test
│   ├── postprocess_and_ensemble.py    <- post-procesamiento + ensemble sobre checkpoints ya entrenados
│   ├── roi_utils.py                   <- calculo compartido de la region de recorte (dos etapas)
│   ├── make_crop_dataset.py           <- genera el dataset de recortes para la etapa 2
│   └── evaluate_two_stage.py          <- evaluacion de punta a punta del enfoque en dos etapas
├── notebooks/
│   └── colab_starter.ipynb           <- notebook listo para correr todo en Colab
├── tools/
│   └── segment_app.py                 <- app Streamlit para segmentar heridas a mano (dibujar
│                                          contorno -> se rellena solo -> guarda máscara en Drive)
├── outputs/
│   └── preprocessing_decisions/      <- justificación citable de cada decisión de preprocesamiento
├── requirements.txt
└── requirements-optional.txt
```

## Tus datos en Drive

```
Tesis Imagenes/
├── Images/   <- todas las fotos (1301 al día de hoy)
└── Masks/    <- solo las que ya segmentaste a mano (739 por ahora, va creciendo)
```

`dataset.py` empareja cada imagen con su máscara por nombre de archivo (acepta
tanto `DSC00598.png` como `DSC00598_mask.png` como nombre de máscara). Las
imágenes que todavía no tienen máscara quedan separadas aparte, no se
descartan — sirven para las ideas de semi-supervised / active learning que
quedaron en el documento de ideas de tesis.

El split de train/val/test es aleatorio por imagen (no por animal): se decidió
así porque organizar qué fotos pertenecen al mismo animal a lo largo del
tiempo no es viable con la información disponible. Queda documentado como
limitación conocida, no como un descuido — ver la nota al final de este README.

## Dónde correr esto: Colab (sin GPU local por ahora)

Como no hay GPU propia todavía, **todo el entrenamiento se corre en Google
Colab**. VS Code queda para escribir/editar código y para operaciones de git
(con GitHub Desktop, en tu caso). El flujo:

1. Editás código en VS Code.
2. Commit + push con GitHub Desktop (sube los cambios a este repo).
3. En Colab, abrís `notebooks/colab_starter.ipynb` (`Archivo -> Abrir notebook
   -> GitHub`, pegando la URL del repo de arriba) y corrés las celdas en
   orden: clona el repo, monta tu Drive, instala dependencias, entrena.
4. Como Colab no guarda nada entre sesiones, cada vez que abrís una sesión
   nueva el notebook clona el repo de cero — por eso el código en sí (no los
   datos) siempre tiene que estar subido a GitHub para que Colab lo vea.

No edites código directo en las celdas de Colab: se pierde al cerrar la
sesión. Los cambios de código van en VS Code -> GitHub Desktop -> GitHub.

## Setup rápido en Colab

Abrir `notebooks/colab_starter.ipynb` desde GitHub (`Archivo -> Abrir notebook
-> GitHub`, pegar la URL del repo) y correr las celdas en orden. Activar GPU
antes: `Entorno de ejecución -> Cambiar tipo de entorno de ejecución -> GPU`.

## Setup rápido en VS Code (solo para editar, no para entrenar)

```powershell
git clone https://github.com/camichahwan/wound_segmentation.git
cd wound_segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Verificar que el emparejamiento de datos funciona antes de tocar nada más:

```powershell
cd src
python dataset.py
```

Debería imprimir cuántas imágenes tenés con máscara y cuántas sin ella. Si
tira error de que no encuentra la carpeta, revisar `DRIVE_LOCAL_WINDOWS_PATH`
en `src/config.py` (puede que tu letra de unidad de Drive no sea `G:`).

## Dónde se guardan los checkpoints entrenados (importante)

`train.py` guarda los checkpoints (`.pt`) y el historial (`.csv`) usando
`config.get_checkpoints_dir()`:

- **En Colab: dentro de Google Drive** (`Tesis Imagenes/model_checkpoints/`),
  nunca en `/content`. Esto se corrigió después de perder por un rato el
  acceso a un entrenamiento completo (pretrained 40 épocas + scratch 80
  épocas) que se había guardado solo en `/content`, disco temporal de la VM
  de Colab que se borra si se desconecta la sesión o Colab recicla el
  entorno de ejecución.
- **Local (VS Code): sigue relativo al repo** (`outputs/checkpoints/`), como
  antes, porque ahí el disco no es efímero.

Si alguna vez hace falta forzar otra ubicación, se puede setear la variable
de entorno `WOUND_CHECKPOINTS_DIR`.

## Herramienta de segmentación manual (`tools/segment_app.py`)

App local en Streamlit para ir generando las máscaras que todavía faltan
(las imágenes sin par en `Masks/`). Se corre aparte del resto del pipeline,
no necesita PyTorch:

```powershell
cd tools
pip install streamlit streamlit-drawable-canvas pillow opencv-python numpy
streamlit run segment_app.py
```

Abre el navegador en `localhost:8501`, muestra directo la primera imagen sin
máscara (salteando las que ya están hechas), y funciona así: se traza a mano
alzada el contorno de la herida y, al guardar, la app rellena sola el
interior del contorno cerrado y lo guarda como `Masks/<nombre>_mask.png` —
mismo formato y convención de nombre que ya usa `dataset.py`, así que las
máscaras nuevas se integran solas al dataset sin tocar nada más.

**Importante:** el trazo tiene que quedar cerrado (el punto final cerca del
inicial) para que el relleno automático funcione bien — si el contorno queda
abierto, no se detecta un interior para rellenar.

## Decisiones de preprocesamiento (justificación para la tesis)

Cada vez que se evalúa si un paso de preprocesamiento entra o no al pipeline,
el resultado se guarda en `outputs/preprocessing_decisions/` (CSV con el
detalle por imagen + un `.md` con método, tabla de resultados y conclusión),
para poder citarlo directamente al redactar la tesis en vez de tener que
reconstruir el análisis de memoria.

- `clahe_decision.md` — CLAHE sobre el canal L (LAB) para normalizar
  iluminación despareja. Resultado sobre 25 imágenes reales: no mejora la
  separación herida/piel de forma significativa en L, y empeora de forma
  significativa (p=0.0001) la separación en el canal a (rojo-verde), que es
  el canal más informativo del dataset. **Decisión: no se incorpora al
  pipeline de entrenamiento por default.** Correr
  `src/evaluate_preprocessing_effect.py` sobre más imágenes (o sobre las 739
  completas) si se quiere una conclusión más robusta antes de la redacción
  final.

## Análisis de errores (resultados del entrenamiento, justificación para la tesis)

Después de comparar `unet_pretrained` vs. `unet_scratch` sobre el set de test
(110 imágenes, resultados en `Drive/model_checkpoints/evaluation_results.csv`),
cada hallazgo relevante se documenta en `outputs/error_analysis/`:

- `heridas_chicas_resize.md` — las 4 imágenes donde ambos modelos fallan más
  fuerte resultaron ser heridas chicas/tempranas (hasta 0.015% de la imagen)
  que el resize fijo a 512x512 reduce a ~6x6 píxeles, por debajo de lo que la
  red puede segmentar. No son máscaras mal hechas ni casos a descartar --
  son justo el tipo de caso (herida en estadio inicial) que el anteproyecto
  plantea como motivación. Decisión: mantenerlas en el dataset, documentar la
  limitación, y atacarla con un enfoque en dos etapas (localización gruesa +
  recorte + segmentación fina) en vez de con cambios de loss function.

## Mejorar resultados sin reentrenar (`src/postprocess_and_ensemble.py`)

Evalúa, sobre los checkpoints ya entrenados y con la misma metodología de
evidencia (Wilcoxon pareado por imagen) que el resto del repo, si conviene
sumar:

- **Post-procesamiento**: cierre morfológico + quedarse con el componente
  conectado más grande (elimina blobs falsos positivos sueltos, que son los
  que más inflan el Hausdorff distance).
- **Ensemble**: promediar las probabilidades de `unet_pretrained` +
  `unet_scratch` antes de binarizar.

No ataca el hallazgo de `outputs/error_analysis/heridas_chicas_resize.md`
(eso es pérdida de información en el resize, no algo que postprocesamiento
o ensemble puedan arreglar) — apunta a las demás fallas: bordes ruidosos,
blobs sueltos, casos donde un modelo se equivoca y el otro no.

```
python src/postprocess_and_ensemble.py
```

Guarda `postprocess_ensemble_results.csv` y `postprocess_ensemble_decision.md`
en la misma carpeta de Drive que los checkpoints.

## Enfoque en dos etapas (heridas chicas/tempranas)

Ataca el hallazgo de `outputs/error_analysis/heridas_chicas_resize.md`: el
resize fijo a 512x512 destruye la información de las heridas más chicas
(~6x6 px después del resize). En vez de cambiar la resolución de todo el
pipeline (caro en memoria/tiempo), se agrega una segunda etapa:

1. **Etapa 1 (localización gruesa)**: `unet_pretrained`, pero entrenado a
   1024x1024 en vez de 512x512 (`--run_name unet_localizer_1024`), para que
   pueda detectar aunque sea aproximadamente heridas chicas que a 512 ya
   desaparecen. Da una máscara aproximada, no el contorno final.
2. **Generar recortes de entrenamiento** (`make_crop_dataset.py`): a partir
   de la máscara REAL de cada imagen de train, recorta la región alrededor
   de la herida (con margen + aleatoriedad, `roi_utils.py`) y guarda el
   resultado en una carpeta nueva de Drive (`Tesis Imagenes Crops/`,
   hermana de `Tesis Imagenes/`). Como los nombres de archivo no cambian,
   el split train/val/test es exactamente el mismo que en el dataset
   original (misma seed).
3. **Etapa 2 (segmentación fina)**: fine-tuning de `unet_pretrained` (no
   parte de cero, parte de los pesos ya entrenados con `--init_checkpoint`)
   sobre esos recortes a 512x512 -- ahora la herida ocupa una fracción
   mucho más grande de la imagen que le llega a la red.
4. **Evaluación de punta a punta** (`evaluate_two_stage.py`): sobre las
   imágenes de test ORIGINALES completas (no los recortes), corre la etapa
   1 para localizar, recorta la imagen real con esa predicción (no con la
   máscara real -- así es honesto), corre la etapa 2 sobre el recorte, y
   compara el resultado final contra `unet_pretrained` solo con test
   pareado de Wilcoxon. Reporta también especialmente cómo le va en las 4
   imágenes de heridas chicas ya documentadas.

Las celdas correspondientes están en `colab_starter.ipynb` (después de la
celda 8, marcadas "Enfoque en dos etapas"). Tiempo estimado adicional:
~3-4hs para la etapa 1 a 1024px, ~1h para la etapa 2, ~20min para la
evaluación final -- bastante más que las corridas anteriores, tenerlo en
cuenta para planificar la sesión de Colab (los checkpoints se siguen
guardando en Drive automáticamente a medida que mejoran, así que un corte
de sesión no hace perder todo el progreso).

## Orden sugerido de trabajo desde acá

1. ~~Confirmar en Colab que `train.py` corre de punta a punta~~ — hecho.
2. ~~Entrenar `unet_pretrained` (40 épocas) y `unet_scratch` (80 épocas),
   comparar con `evaluate.py`~~ — hecho, ver
   `outputs/error_analysis/heridas_chicas_resize.md` para el resultado y el
   hallazgo de heridas chicas.
3. `python src/postprocess_and_ensemble.py` — post-procesamiento + ensemble
   sobre lo ya entrenado, sin reentrenar (celda 8 del notebook).
4. Enfoque en dos etapas para heridas chicas (celdas 9 a 12 del notebook,
   ver sección de arriba): entrenar localizador a 1024px, generar recortes,
   fine-tuning de la etapa 2, evaluación de punta a punta.
5. Más adelante: baselines clásicos (región growing y los más usados para
   este tipo de segmentación) y comparación contra herramientas actuales
   (SAM) — todavía no están implementados en este repo, se agregan cuando
   lleguemos a esa parte.

## Notas honestas sobre el estado actual del código

- El split train/val/test en `dataset.py` es por imagen, no por animal
  (decisión tomada explícitamente: no es viable organizar qué fotos son del
  mismo animal). Documentado en el comentario de `train_val_test_split`.
- Los modelos no se probaron corriendo un entrenamiento real todavía (se
  escribieron y revisaron a mano en un entorno sin GPU) — el primer paso en
  Colab debería ser correr unas pocas épocas de prueba para confirmar que no
  hay ningún error de forma/dimensión antes de lanzar un entrenamiento largo.
- **El enfoque en dos etapas (`roi_utils.py`, `make_crop_dataset.py`,
  `evaluate_two_stage.py`, y el flag `--init_checkpoint` de `train.py`) está
  escrito y revisado a mano (compila, sin errores de sintaxis) pero
  TODAVÍA NO SE CORRIÓ de punta a punta con GPU real.** Es código nuevo y
  más complejo que el resto del repo (recorte, coordenadas, pegado de
  vuelta a la imagen completa) — corrida la primera vez, revisar con
  atención que los números den razonables (por ejemplo, que el `n_stage1_empty`
  que imprime `evaluate_two_stage.py` no sea sospechosamente alto) antes de
  asumir que el resultado es correcto.
