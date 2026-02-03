import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate_comparison_plot(skel_p, skel_a, score):
    fig, ax = plt.subplots(figsize=(5, 5))
    overlay = np.zeros((256, 256, 3), dtype=np.uint8)
    overlay[skel_p > 0] = [0, 255, 0] # Patrón verde
    overlay[skel_a > 0] = [255, 0, 0] # Alumno rojo
    
    ax.imshow(overlay)
    ax.set_title(f"Resultado Final - Score: {score}")
    ax.axis('off')
    
    # Convertir gráfica a Base64 para enviarla por JSON
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str