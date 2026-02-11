import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.core.processor import prune_skeleton
from app.core.config import TARGET_SHAPE, MIN_BRANCH_LENGTH

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
        render_size = 1400 
        font_size = 1100
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        print(f"Error con la fuente.")
        return

    for char in ALPHABET:
        img = Image.new('L', (render_size, render_size), 0)
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top
        draw.text(((render_size - w) / 2 - left, (render_size - h) / 2 - top), char, font=font, fill=255)
        
        binary = np.array(img)

        # Pipeline alineado con preprocess_robust (alumno): mismo blur y umbral
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w_b, h_b = cv2.boundingRect(coords)
            side = max(w_b, h_b) + 60
            square_canvas = np.zeros((side, side), dtype=np.uint8)
            off_y, off_x = (side - h_b) // 2, (side - w_b) // 2
            square_canvas[off_y:off_y+h_b, off_x:off_x+w_b] = binary[y:y+h_b, x:x+w_b]

            resized = cv2.resize(square_canvas, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
            resized = cv2.GaussianBlur(resized, (9, 9), 0)
            _, resized = cv2.threshold(resized, 110, 255, cv2.THRESH_BINARY)

            skel = skeletonize(resized > 0, method='lee').astype(np.uint8)
            skel = prune_skeleton(skel, min_branch_length=MIN_BRANCH_LENGTH)
            
            name = get_safe_filename(char)
            np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), skel)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), skel * 255)
            print(f"Éxito: {char}")

if __name__ == "__main__":
    generate_npy_templates()