# Segmentación de heridas infecciosas subcutáneas en ratas — PFC Bioingeniería

Código base para el proyecto final de carrera. Estructura pensada para que el
mismo código corra sin cambios ni en Google Colab ni en VS Code local, y para
que las 1000 imágenes (en Google Drive) nunca tengan que subirse a GitHub.

## Estructura del repo

```
wound_segmentation/
├── data/
│   └── samples/            <- 1 imagen + 1 máscara de ejemplo (SÍ va al repo, para poder
│                               probar el código sin acceso a Drive)
├── src/
│   ├── config.py            <- resuelve la ruta a "Tesis Imagenes" según el entorno
│   ├── dataset.py            <- empareja Images/ con Masks/, separa etiquetadas/sin etiquetar
│   ├── preprocessing.py      <- LAB, Canny sobre canal rojo, K-means, Otsu (métodos clásicos)
│   ├── demo_preprocessing.py <- corre lo anterior sobre 1 imagen y genera una figura
│   ├── baselines_classical.py<- corre los métodos clásicos sobre TODO el test set
│   ├── sam_baseline.py       <- comparación contra Segment Anything (opcional, Idea 4)
│   ├── metrics.py            <- Dice, IoU, Precision/Recall, Hausdorff, error de área
│   ├── models/
│   │   ├── unet_scratch.py    <- U-Net implementada desde cero
│   │   └── unet_pretrained.py <- U-Net con encoder preentrenado (segmentation_models_pytorch)
│   ├── train.py               <- entrena cualquiera de los dos modelos
│   └── evaluate.py            <- compara checkpoints entrenados sobre el set de test
├── notebooks/
│   └── colab_starter.ipynb   <- notebook listo para correr todo en Colab
├── outputs/                  <- checkpoints, csvs, figuras (NO se versiona el contenido)
├── requirements.txt
└── requirements-optional.txt <- solo si vas a usar sam_baseline.py
```

## Tus datos en Drive

```
Tesis Imagenes/
├── Images/   <- TODAS las fotos (1000)
└── Masks/    <- solo las que ya segmentaste a mano (~500 por ahora, va creciendo)
```

`dataset.py` empareja cada imagen con su máscara por nombre de archivo (acepta
tanto `DSC00598.png` como `DSC00598_mask.png` como nombre de máscara). Las
imágenes que todavía no tienen máscara quedan separadas aparte, no se
descartan — sirven para las ideas de semi-supervised / active learning que
quedaron en el documento de ideas de tesis.

No hace falta que hoy tengas 1000/1000 segmentadas para empezar a entrenar:
`train.py` arranca apenas haya un puñado de pares imagen-máscara.

## Dónde correr esto: Colab vs. VS Code

**Recomendación: los dos, para cosas distintas, apuntando siempre al mismo repo de GitHub.**

- **VS Code local**, para escribir y depurar código. Ya tenés `Tesis Imagenes`
  sincronizada en `G:\My Drive\Tesis Imagenes` vía Google Drive for Desktop,
  así que desde VS Code el dataset se ve como una carpeta local común y
  corriente — no hay nada especial que configurar más que dejar `config.py`
  apuntando a esa ruta (ya está seteada por default). Es el mejor lugar para
  editar, revisar diffs de git, y correr scripts cortos (`dataset.py`,
  `demo_preprocessing.py`, `baselines_classical.py`) que no necesitan GPU.

- **Google Colab**, para los entrenamientos (`train.py`) mientras no tengas
  la GPU propia (la RTX 3060 que está en el análisis de costos del
  anteproyecto). Colab monta tu Google Drive directamente (misma carpeta
  `Tesis Imagenes`, sin tener que subir nada a mano) y te da GPU gratis (con
  límites de uso) o con Colab Pro si hace falta más tiempo/memoria. Usá
  `notebooks/colab_starter.ipynb` como punto de partida: clona el repo,
  monta Drive, instala dependencias y corre todo en orden.

  Cuando tengas la GPU local, el mismo `train.py` corre igual desde VS Code
  (una terminal integrada) sin cambiar una línea — `config.py` ya detecta
  que no estás en Colab y usa la ruta de Windows.

No hace falta elegir uno para siempre: el código vive en GitHub, así que
"dónde lo corro" es una decisión de cada sesión de trabajo, no del proyecto.

## Setup en VS Code (local, Windows)

```powershell
git clone https://github.com/TU_USUARIO/wound-segmentation.git
cd wound-segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Verificar que el emparejamiento de datos funciona antes de entrenar nada:

```powershell
cd src
python dataset.py
```

Debería imprimir cuántas imágenes tenés con máscara y cuántas sin ella. Si
tira error de que no encuentra la carpeta, revisar `DRIVE_LOCAL_WINDOWS_PATH`
en `src/config.py` (puede que tu letra de unidad de Drive no sea `G:`).

Correr el demo de preprocesamiento (no necesita GPU ni datos de Drive, usa
las 2 imágenes de muestra del repo):

```powershell
python demo_preprocessing.py
```

## Setup en Colab

Abrir `notebooks/colab_starter.ipynb` en Colab (subiéndolo directo, o desde
GitHub con `Archivo -> Abrir notebook -> GitHub` una vez que el repo esté
publicado) y correr las celdas en orden. Activar GPU antes: `Entorno de
ejecución -> Cambiar tipo de entorno de ejecución -> GPU`.

## Subir este proyecto a GitHub (primera vez)

1. Crear un repo vacío en https://github.com/new (ej. `wound-segmentation`),
   **sin** marcar "Add a README" (para no generar conflictos con el que ya
   tenés acá).

2. Desde la carpeta del proyecto (en VS Code o en una terminal):

   ```bash
   git init
   git add .
   git commit -m "Estructura inicial: preprocesamiento, U-Net (scratch y preentrenada), baselines"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/wound-segmentation.git
   git push -u origin main
   ```

3. De ahí en más, el flujo normal es: editás en VS Code, `git add`,
   `git commit`, `git push`; y en Colab, al principio de cada sesión,
   `git clone` (o `git pull` si ya lo tenías clonado en ese runtime, aunque
   en Colab lo más simple suele ser clonar de nuevo cada vez que se reinicia
   el entorno de ejecución, ya que no persiste entre sesiones).

Importante: como `data/raw_images/`, `data/raw_masks/` y `outputs/` están en
`.gitignore`, ni las fotos ni los checkpoints entrenados se van a subir a
GitHub — eso es intencional (GitHub no es el lugar para 1000 fotos en alta
resolución ni para pesos de modelos de cientos de MB). El dataset real vive
solo en Google Drive; el repo es solo el código.

## Orden sugerido de trabajo

1. `python src/dataset.py` — confirmar que el emparejamiento de imágenes y
   máscaras funciona con tu estructura real de Drive.
2. `python src/demo_preprocessing.py` — mirar la figura en
   `outputs/figures/preprocessing_demo.png`. Ya se corrió sobre la imagen de
   muestra de este repo: los métodos clásicos (LAB / Canny sobre canal rojo /
   K-means k=3) dan resultados pobres contra la máscara manual en esa imagen,
   lo cual es un hallazgo real y vale la pena reportarlo tal cual en la
   sección de justificación de la tesis, no ocultarlo.
3. `python src/baselines_classical.py` — lo mismo pero sobre TODO el set de
   test, para tener un número (no una sola imagen) que comparar contra los
   modelos de deep learning.
4. `python src/train.py --model pretrained --encoder resnet34 --epochs 40` —
   primer modelo entrenable, el más rápido de poner andando.
5. `python src/train.py --model scratch --epochs 80` — la U-Net desde cero
   que pidió tu tutor, para comparar contra la anterior.
6. `python src/evaluate.py --checkpoints ...` — tabla comparativa final entre
   los dos modelos, sobre el mismo set de test.
7. (Opcional, cuando llegues ahí) `src/sam_baseline.py` — comparación contra
   Segment Anything como "herramienta actual" además de los métodos clásicos.

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
  pipeline de entrenamiento.** Correr `src/evaluate_preprocessing_effect.py`
  sobre más imágenes (o sobre las 739 completas) si se quiere una conclusión
  más robusta antes de la redacción final.

## Notas honestas sobre el estado actual del código

- `estimate_foreground_mask` en `preprocessing.py` (aislar al animal del
  fondo antes de buscar la herida por color) dio resultados inestables en las
  pruebas iniciales — está documentada como experimental en el propio código
  y no se usa por default en `demo_preprocessing.py` ni en
  `baselines_classical.py`. Si la mejorás, actualizá ambos scripts para
  usarla.
- El split train/val/test en `dataset.py` es por imagen, no por animal. Si
  varias fotos del dataset son el mismo animal en distintos días, hay que
  cambiar el split para separar por animal (evitar que el mismo sujeto
  aparezca en train y en test), ver el comentario en
  `train_val_test_split`.
- Los modelos no se probaron corriendo un entrenamiento real (este entorno de
  desarrollo no tiene GPU ni PyTorch instalado) — el código está revisado a
  mano y sigue los patrones estándar de PyTorch/segmentation_models_pytorch,
  pero conviene correr unas pocas épocas de prueba apenas tengas entorno con
  GPU para confirmar que no hay ningún error de forma/dimensión antes de
  lanzar un entrenamiento largo.
