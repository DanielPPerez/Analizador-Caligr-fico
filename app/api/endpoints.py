import os
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.processor import preprocess_robust
# from app.core.processor import get_image_for_resnet  # Para ResNet/OCR futuro
from app.metrics.geometric import calculate_geometric
from app.metrics.topologic import get_topology
from app.metrics.trajectory import calculate_trajectory_dist
from app.metrics.quality import calculate_quality_metrics
from app.metrics.segment_cosine import calculate_segment_cosine_similarity
from app.metrics.scorer import calculate_final_score
from app.core.config import TARGET_SHAPE
from app.utils.visualizer import generate_comparison_plot

router = APIRouter()
TEMPLATES_DIR = "app/templates"

# Cache de plantillas por carácter para requests repetidas (reduce I/O y carga)
TEMPLATE_CACHE = {}


def get_template_filename(char):
    """Debe coincidir exactamente con la lógica del script de generación."""
    if char.isdigit():
        return f"digit_{char}.npy"
    c = "N_tilde" if char.upper() == "Ñ" else char
    suffix = "upper" if char.isupper() else "lower"
    return f"{c}_{suffix}.npy"


def get_template(char):
    """Devuelve el esqueleto de la plantilla para el carácter (desde cache o disco)."""
    if char in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[char]
    filename = get_template_filename(char)
    template_path = os.path.join(TEMPLATES_DIR, filename)
    if not os.path.exists(template_path):
        return None
    skel = np.load(template_path)
    if skel.shape != TARGET_SHAPE:
        skel = cv2.resize(
            skel.astype(np.uint8), (TARGET_SHAPE[1], TARGET_SHAPE[0]),
            interpolation=cv2.INTER_NEAREST
        )
    TEMPLATE_CACHE[char] = skel
    return skel


@router.post("/evaluate")
async def evaluate(file: UploadFile = File(...), target_char: str = Form(...)):
    skel_p = get_template(target_char)
    if skel_p is None:
        raise HTTPException(status_code=404, detail=f"No existe plantilla para '{target_char}'")

    img_bytes = await file.read()
    skel_a = preprocess_robust(img_bytes)
    if skel_a is None: return {"error": "Sin trazo detectado"}

    geo = calculate_geometric(skel_p, skel_a)
    topo_p = get_topology(skel_p)
    topo_a = get_topology(skel_a)
    traj_dist = calculate_trajectory_dist(skel_p, skel_a)
    quality = calculate_quality_metrics(skel_a)
    cosine_cos, cosine_score = calculate_segment_cosine_similarity(skel_p, skel_a)
    
    topo_match = (topo_p['loops'] == topo_a['loops'])
    
    score_final = calculate_final_score(geo, topo_match, traj_dist, cosine_segment_score=cosine_score)
    v_img = generate_comparison_plot(skel_p, skel_a, score_final)

    return {
        "char": target_char,
        "score_final": score_final,
        "metrics": {
            "geometric": geo,
            "topology": {"match": topo_match, "student": topo_a, "pattern": topo_p},
            "quality": quality,
            "trajectory_error": traj_dist,
            "segment_cosine": {"cosine": cosine_cos, "score": cosine_score},
        },
        "image_b64": v_img
    }