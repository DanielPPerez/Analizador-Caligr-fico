import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
import sys

# Ajuste de ruta para importar desde app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.core.processor import prune_skeleton

FONT_PATH = "app/fonts/KGPrimaryPenmanship.ttf"
OUTPUT_DIR = "app/templates_debug_png"
ALPHABET = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyz0123456789"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_safe_filename(char):
    if char.isdigit(): return f"digit_{char}"
    c = "N_tilde" if char.upper() == "Ñ" else char
    suffix = "upper" if char.isupper() else "lower"
    return f"{c}_{suffix}"

def generate_png_templates():
    try:
        font = ImageFont.truetype(FONT_PATH, 280)
    except Exception:
        print("Error: No se encontró la fuente en app/fonts/")
        return

    for char in ALPHABET:
        img = Image.new('L', (400, 400), 255)
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top
        draw.text(((400 - w) / 2 - left, (400 - h) / 2 - top), char, font=font, fill=0)

        cv_img = np.array(img)
        _, binary = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        smooth_mask = np.zeros_like(binary)
        if hierarchy is not None:
            for i in range(len(contours)):
                if hierarchy[0][i][3] == -1:
                    epsilon = 0.002 * cv2.arcLength(contours[i], True)
                    approx = cv2.approxPolyDP(contours[i], epsilon, True)
                    cv2.drawContours(smooth_mask, [approx], -1, 255, -1)
                else:
                    cv2.drawContours(smooth_mask, [contours[i]], -1, 0, -1)

        coords = cv2.findNonZero(smooth_mask)
        if coords is not None:
            x, y, w_b, h_b = cv2.boundingRect(coords)
            crop = smooth_mask[y:y+h_b, x:x+w_b]
            resized = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA)

            skel = skeletonize(resized > 0).astype(np.uint8)
            skel = prune_skeleton(skel, min_branch_length=20)

            skel_visible = (skel * 255).astype(np.uint8)
            name = get_safe_filename(char)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), skel_visible)
            print(f"Imagen generada: {name}.png")
        else:
            print(f"Salto: {char}")

if __name__ == "__main__":
    generate_png_templates()