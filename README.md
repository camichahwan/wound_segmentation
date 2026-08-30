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
│   ├── train.py                       <- entrena cualquiera de los dos modelos
│   └── evaluate.py                    <- compara checkpoints entrenados sobre el set de test
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

## Orden sugerido de trabajo desde acá

1. Confirmar en Colab que `train.py` corre de punta a punta con pocas épocas
   (chequeo de que no hay errores de código, antes de un entrenamiento largo).
2. `python src/train.py --model pretrained --encoder resnet34 --epochs 40` —
   primer modelo entrenable.
3. `python src/train.py --model scratch --epochs 80` — la U-Net desde cero
   que pidió tu tutor, para comparar contra la anterior.
4. `python src/evaluate.py --checkpoints ...` — tabla comparativa final entre
   los dos modelos, sobre el mismo set de test.
5. Más adelante: baselines clásicos y comparación contra herramientas
   actuales (SAM) como punto de comparación adicional — todavía no están
   implementados en este repo, se agregan cuando lleguemos a esa parte.

## Notas honestas sobre el estado actual del código

- El split train/val/test en `dataset.py` es por imagen, no por animal
  (decisión tomada explícitamente: no es viable organizar qué fotos son del
  mismo animal). Documentado en el comentario de `train_val_test_split`.
- Los modelos no se probaron corriendo un entrenamiento real todavía (se
  escribieron y revisaron a mano en un entorno sin GPU) — el primer paso en
  Colab debería ser correr unas pocas épocas de prueba para confirmar que no
  hay ningún error de forma/dimensión antes de lanzar un entrenamiento largo.
