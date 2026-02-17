import cv2
import numpy as np

def segment_characters(binary_img):
    """
    Recibe una imagen binarizada y devuelve una lista de recortes de letras
    ordenados de izquierda a derecha.
    """
    # 1. Encontrar contornos
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_crops = []
    # 2. Filtrar por tamaño para evitar ruido
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10: # Filtro mínimo
            char_crops.append((x, y, w, h))
            
    # 3. Ordenar de izquierda a derecha (importante para leer palabras)
    char_crops.sort(key=lambda x: x[0])
    
    return char_crops