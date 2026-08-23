import os
from pathlib import Path
import json
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

# Configuración de página
st.set_page_config(page_title="Segmentador de Heridas Pro: Trazado de Borde Irregular", layout="wide")

# ---------- Configuración de Rutas (Tu Google Drive) ----------
# Usamos 'r' adelante para que Windows lea bien las barras invertidas
ROOT_DIR = Path(r"G:\My Drive\Tesis Imagenes")
IM_DIR = ROOT_DIR / "Images"
MK_DIR = ROOT_DIR / "Masks"

# Creamos la carpeta de máscaras por si no existe
MK_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Menú Lateral (Barra de herramientas) ----------
st.sidebar.title("🛠️ Herramientas de Trazado")

# CONFIGURACIÓN DEL PINCEL - Ahora solo un modo, el de Trazado de Borde
brush = st.sidebar.slider("Grosor del borde (finito es mejor):", 1, 10, 2, 1)
# El relleno temporal es muy transparente para que veas el borde perfectamente
alpha_fill = st.sidebar.slider("Opacidad del relleno automático (temporal):", 0.05, 0.50, 0.10, 0.05)

# Color verde fosforescente para el borde (RGB: 0, 255, 0)
stroke_color = "#00ff00" 
# El relleno temporal es rojo transparente para mostrar el área que se rellenará
fill_color = f"rgba(255, 0, 0, {alpha_fill})"

skip_labeled = st.sidebar.checkbox("Omitir imágenes que ya tienen máscara", True)
max_canvas_width = st.sidebar.number_input("Ancho máximo de pantalla (px)", 600, 1400, 800, 50)

if "idx" not in st.session_state:
    st.session_state.idx = 0

# ---------- Utilidades Matemáticas Pro ----------
def list_images():
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF"}
    ims = sorted([p for p in IM_DIR.glob("*") if p.suffix in exts])
    if skip_labeled:
        ims = [p for p in ims if not (MK_DIR / f"{p.stem}_mask.png").exists()]
    return ims

def generate_filled_binary_mask(rgba_canvas, target_size):
    """
    Función Mágica: Agarra el trazo de borde, encuentra el contorno cerrado y rellena el interior.
    """
    arr = np.asarray(rgba_canvas).astype(np.uint8)   # Matriz de la imagen (R,G,B,Alpha)
    
    # Extraemos solo el canal alpha (donde dibujaste el contorno)
    contour_drawn = arr[:, :, 3] > 0
    contour_drawn_mask = (contour_drawn.astype(np.uint8) * 255) # Máscara binaria del trazo
    
    # Redimensionamos al tamaño original antes del procesamiento para máxima precisión
    if contour_drawn_mask.shape[::-1] != target_size:
        contour_drawn_mask = cv2.resize(contour_drawn_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Usamos OpenCV para encontrar el contorno más grande (el tuyo) y rellenarlo
    filled_mask = np.zeros_like(contour_drawn_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(contour_drawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontramos el contorno con el área máxima (asumiendo que dibujaste uno solo)
        largest_contour = max(contours, key=cv2.contourArea)
        # Dibujamos y RELLENAMOS ese contorno con blanco (255)
        cv2.drawContours(filled_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
        return filled_mask
    else:
        # No se encontró ningún contorno
        return np.zeros_like(contour_drawn_mask, dtype=np.uint8)

def load_image(path: Path):
    im = Image.open(path).convert("RGB")
    return im

def ensure_idx_in_range(n):
    st.session_state.idx = max(0, min(st.session_state.idx, n-1))

# ---------- Carga de Imágenes ----------
images = list_images()
st.sidebar.write("---")
st.sidebar.write(f"📸 Imágenes por anotar: **{len(images)}**")

if len(images) == 0:
    st.success("¡Excelente! No hay fotos pendientes o no se encontraron imágenes en la carpeta 'Images'.")
    st.balloons()
    st.stop()

ensure_idx_in_range(len(images))
img_path = images[st.session_state.idx]
mask_path = MK_DIR / f"{img_path.stem}_mask.png"

# ---------- Procesamiento de la foto actual ----------
img = load_image(img_path)        
W, H = img.size
scale = min(1.0, max_canvas_width / W)
canvas_w, canvas_h = int(W * scale), int(H * scale)

st.markdown(f"### 🐭 Procesando Borde Irregular: `{img_path.name}` (Foto {st.session_state.idx+1} de {len(images)})")

col_canvas, col_ctrl = st.columns([3, 1]) # Canvas más grande

with col_canvas:
    st.info("💡 **Traza el borde irregular de la herida a mano alzada con un trazo fino.** El interior se rellenará automáticamente de blanco sólido al guardar.")
    
    # EL CANVAS PRO: Solo en modo freedraw
    canvas = st_canvas(
        fill_color=fill_color,
        stroke_width=brush,
        stroke_color=stroke_color,
        background_image=img.resize((canvas_w, canvas_h)),
        height=canvas_h,
        width=canvas_w,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.idx}",
        display_toolbar=True,
    )

with col_ctrl:
    st.subheader("Controles")
    st.write("Dibuja el contorno y tocá Guardar y Siguiente.")
    
    c1, c2 = st.columns(2)
    save_btn   = c1.button("💾 Guardar", use_container_width=True)
    next_btn   = c2.button("➡️ Guardar y Siguiente", type="primary", use_container_width=True)
    
    st.write("---")
    prev_btn   = st.button("⬅️ Volver a la foto anterior", use_container_width=True)

    overwrite = st.checkbox("Sobrescribir si ya existe máscara", False)

# ---------- Lógica de Guardado (Modificada para Relleno Pro) ----------
def do_save():
    if canvas is None or canvas.image_data is None:
        st.error("No hay anotación en el lienzo para guardar.")
        return False

    if mask_path.exists() and not overwrite:
        st.error("⚠️ La máscara ya existe. Marca 'Sobrescribir' abajo si querés reemplazarla.")
        return False

    # Extraer la máscara y GENERAR EL RELLENO AUTOMÁTICO
    filled_binary_mask = generate_filled_binary_mask(canvas.image_data, (W, H))
    
    # Chequear si la máscara está completamente vacía (negra)
    if not np.any(filled_binary_mask):
        st.warning("⚠️ No dibujaste un contorno cerrado válido. Dibujá la herida antes de guardar.")
        return False

    # Guardar el PNG sólido (blanco y negro)
    Image.fromarray(filled_binary_mask).save(mask_path)

    st.toast(f"✅ Guardado y RELLENADO: {mask_path.name}")
    return True

# ---------- Acciones de los botones ----------
if save_btn:
    do_save()

if next_btn:
    if do_save():
        st.session_state.idx = min(st.session_state.idx + 1, len(images)-1)
        st.rerun() # Actualizado a la sintaxis moderna

if prev_btn:
    st.session_state.idx = max(0, st.session_state.idx - 1)
    st.rerun() # Actualizado a la sintaxis moderna