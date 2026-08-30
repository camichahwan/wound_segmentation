"""
config.py

Un solo lugar para resolver DÓNDE están las imágenes, para que el mismo
código corra sin tocar nada ni en Google Colab ni en VS Code local con
Google Drive for Desktop sincronizado.

Estructura de datos esperada (fija, según lo que ya armaste en Drive):

    Tesis Imagenes/
        Images/     <- TODAS las fotos (1000)
        Masks/      <- solo las máscaras ya segmentadas a mano (~500 por ahora)

Los nombres de archivo de imagen y máscara deben compartir el mismo "stem"
(nombre sin extensión) para poder emparejarlos, ej.:
    Images/DSC00598.JPG   <->  Masks/DSC00598_mask.png
o
    Images/DSC00598.JPG   <->  Masks/DSC00598.png
(dataset.py soporta las dos convenciones, ver `_match_mask_filename`).

Cómo se resuelve la ruta, en orden de prioridad:
1. Variable de entorno WOUND_DATA_ROOT, si está seteada (override manual).
2. Si se detecta que se está corriendo en Google Colab -> ruta de Colab
   después de montar Drive.
3. Si no, la ruta local de Windows (ajustar DRIVE_LOCAL_WINDOWS si tu letra
   de unidad o el nombre de carpeta cambian).
"""

import os


def _running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# --- Ajustar estas dos rutas si hace falta ---
DRIVE_COLAB_PATH = "/content/drive/MyDrive/Tesis Imagenes"
DRIVE_LOCAL_WINDOWS_PATH = r"G:\My Drive\Tesis Imagenes"


def get_data_root() -> str:
    """Devuelve la carpeta raíz 'Tesis Imagenes' según el entorno de ejecución."""
    env_override = os.environ.get("WOUND_DATA_ROOT")
    if env_override:
        return env_override

    if _running_in_colab():
        return DRIVE_COLAB_PATH

    return DRIVE_LOCAL_WINDOWS_PATH


def get_images_dir() -> str:
    return os.path.join(get_data_root(), "Images")


def get_masks_dir() -> str:
    return os.path.join(get_data_root(), "Masks")


CHECKPOINTS_SUBDIR = "model_checkpoints"


def get_checkpoints_dir() -> str:
    """
    Dónde se guardan los checkpoints entrenados (.pt) y su historial (.csv).

    En Colab esto TIENE que vivir en Drive, no en /content (disco temporal de
    la VM: se borra si se desconecta la sesión o Colab recicla el entorno de
    ejecución -- ya perdimos un entrenamiento completo de 40+80 épocas por
    guardarlo solo ahí). Localmente se mantiene relativo al repo, como antes,
    porque ahí el disco no es efímero.
    """
    env_override = os.environ.get("WOUND_CHECKPOINTS_DIR")
    if env_override:
        return env_override

    if _running_in_colab():
        return os.path.join(get_data_root(), CHECKPOINTS_SUBDIR)

    return os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints")


def mount_drive_if_colab():
    """
    Llamar esto al principio de cualquier notebook/script en Colab, antes de
    usar get_data_root(). No hace nada si no se está en Colab.
    """
    if _running_in_colab():
        from google.colab import drive
        drive.mount("/content/drive")


if __name__ == "__main__":
    mount_drive_if_colab()
    print("Entorno Colab:", _running_in_colab())
    print("Data root:", get_data_root())
    print("Images dir:", get_images_dir())
    print("Masks dir:", get_masks_dir())
    print("Checkpoints dir:", get_checkpoints_dir())
