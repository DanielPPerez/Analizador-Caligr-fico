import os
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.processor import preprocess_robust
from app.metrics.geometric import calculate_geometric
from app.metrics.topologic import get_topology
from app.metrics.trajectory import calculate_trajectory_dist
from app.metrics.quality import calculate_quality_metrics
from app.metrics.scorer import calculate_final_score
from app.utils.visualizer import generate_comparison_plot

router = APIRouter()
TEMPLATES_DIR = "app/templates"

def get_template_filename(char):
    """Debe coincidir exactamente con la lógica del script de generación."""
    if char.isdigit():
        return f"digit_{char}.npy"
    c = "N_tilde" if char.upper() == "Ñ" else char
    suffix = "upper" if char.isupper() else "lower"
    return f"{c}_{suffix}.npy"

@router.post("/evaluate")
async def evaluate(file: UploadFile = File(...), target_char: str = Form(...)):
    filename = get_template_filename(target_char)
    template_path = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail=f"No existe plantilla para '{target_char}'")
    
    skel_p = np.load(template_path)
    img_bytes = await file.read()
    skel_a = preprocess_robust(img_bytes)
    if skel_a is None: return {"error": "Sin trazo detectado"}

    geo = calculate_geometric(skel_p, skel_a)
    topo_p = get_topology(skel_p)
    topo_a = get_topology(skel_a)
    traj_dist = calculate_trajectory_dist(skel_p, skel_a)
    quality = calculate_quality_metrics(skel_a)
    
    topo_match = (topo_p['loops'] == topo_a['loops'])
    
    score_final = calculate_final_score(geo, topo_match, traj_dist)
    v_img = generate_comparison_plot(skel_p, skel_a, score_final)

    return {
        "char": target_char,
        "score_final": score_final,
        "metrics": {
            "geometric": geo,
            "topology": {"match": topo_match, "student": topo_a, "pattern": topo_p},
            "quality": quality,
            "trajectory_error": traj_dist
        },
        "image_b64": v_img
    }