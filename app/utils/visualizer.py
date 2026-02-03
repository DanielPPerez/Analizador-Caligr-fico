import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import cv2

def generate_comparison_plot(skel_p, skel_a, score):
    # 1. Definir tamaño objetivo
    target_size = (256, 256)
    
    # 2. Asegurar que ambos tengan el mismo tamaño 
    if skel_p.shape != target_size:
        skel_p = cv2.resize(skel_p.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
    if skel_a.shape != target_size:
        skel_a = cv2.resize(skel_a.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

    # 3. Crear el lienzo RGB (Fondo oscuro para mejor contraste)
    h, w = target_size
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Lógica de colores para Debug Profesional:
    # Amarillo (255, 255, 0) -> Coincidencia perfecta (Intersección)
    # Verde (0, 255, 0)      -> Parte de la plantilla que no se tocó
    # Rojo (255, 0, 0)       -> Parte que el alumno dibujó fuera de la plantilla
    
    intersection = np.logical_and(skel_p > 0, skel_a > 0)
    only_p = np.logical_and(skel_p > 0, np.logical_not(skel_a > 0))
    only_a = np.logical_and(skel_a > 0, np.logical_not(skel_p > 0))

    overlay[only_p] = [0, 150, 0]       # Verde oscuro (Guía)
    overlay[only_a] = [255, 50, 50]     # Rojo (Error/Desvío)
    overlay[intersection] = [255, 255, 0] # Amarillo (Acierto)

    # 4. Crear la figura de Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    
    # Añadir texto con el Score y leyenda
    ax.set_title(f"Evaluación: {score:.2f}%", color="white", fontsize=14, fontweight='bold')
    
    ax.text(5, 245, "Verde: Guía | Rojo: Error | Amarillo: Acierto", 
            color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    ax.axis('off')
    fig.patch.set_facecolor('#1e1e1e') 

    # 5. Convertir gráfica a Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    
    return img_str