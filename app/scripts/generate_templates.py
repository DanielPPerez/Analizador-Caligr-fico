import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
import sys

# Ajuste de ruta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.core.processor import prune_skeleton

FONT_PATH = "app/fonts/KGPrimaryPenmanship.ttf" 
OUTPUT_DIR = "app/templates"
ALPHABET = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyz0123456789"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_safe_filename(char):
    if char.isdigit(): return f"digit_{char}"
    c = "N_tilde" if char.upper() == "Ñ" else char
    suffix = "upper" if char.isupper() else "lower"
    return f"{c}_{suffix}"

def generate_npy_templates():
    try:
        # Aumentamos el tamaño de renderizado para máxima precisión
        render_size = 1200 
        font_size = 1000
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        print("Error: No se encontró la fuente.")
        return

    for char in ALPHABET:
        # 1. RENDERIZADO EN ALTA DEFINICIÓN
        # Usamos un fondo negro y letra blanca para facilitar el procesamiento
        img = Image.new('L', (render_size, render_size), 0)
        draw = ImageDraw.Draw(img)
        
        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top
        draw.text(((render_size - w) / 2 - left, (render_size - h) / 2 - top), char, font=font, fill=255)
        
        # Convertir a numpy
        img_np = np.array(img)

        # 2. SUAVIZADO ANTI-ALIASING (Crucial para evitar líneas "cuadradas")
        # Aplicamos un desenfoque gaussiano antes de binarizar para suavizar bordes
        img_np = cv2.GaussianBlur(img_np, (15, 15), 0)
        _, binary = cv2.threshold(img_np, 100, 255, cv2.THRESH_BINARY)

        # 3. MORFOLOGÍA CIRCULAR (Evita esquinas cuadradas en 'v', 'z')
        # Usamos un kernel elíptico para que las puntas sean redondeadas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 4. RECORTE PROPORCIONAL
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w_b, h_b = cv2.boundingRect(coords)
            # Crear un lienzo cuadrado para mantener el aspecto
            side = max(w_b, h_b) + 100
            square_canvas = np.zeros((side, side), dtype=np.uint8)
            
            # Centrar el recorte en el lienzo cuadrado
            off_y = (side - h_b) // 2
            off_x = (side - w_b) // 2
            square_canvas[off_y:off_y+h_b, off_x:off_x+w_b] = binary[y:y+h_b, x:x+w_b]
            
            # Redimensionar a 256x256 (un poco más grande mejora la calidad)
            # Usar INTER_AREA para reducir sin artefactos
            resized = cv2.resize(square_canvas, (256, 256), interpolation=cv2.INTER_AREA)
            _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

            # 5. ESQUELETIZACIÓN
            skel = skeletonize(resized > 0).astype(np.uint8)
            
            # 6. PODA SELECTIVA
            # Bajamos el valor. 15-20 es ideal para 256px.
            # 'i' y 'l' suelen tener trazos de más de 20px, pero ramas de error tienen < 10.
            skel = prune_skeleton(skel, min_branch_length=15)
            
            name = get_safe_filename(char)
            np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), skel)
            
            # Guardar visualización para debug (opcional)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), skel * 255)
            print(f"Generada: {char}")

if __name__ == "__main__":
    generate_npy_templates()