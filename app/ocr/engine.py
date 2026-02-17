import cv2
import numpy as np
import onnxruntime as ort # Ideal para Edge Computing
from app.ocr.segmentation import segment_characters

class OCREngine:
    def __init__(self, model_path="app/models/ocr_model.onnx"):
        # Cargar el modelo ONNX (mucho más ligero que TensorFlow completo)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # Mapeo de índices a letras (Ejemplo para EMNIST)
        self.labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def predict_character(self, crop):
        """Preprocesa el recorte y predice la letra"""
        # 1. Preparar para el modelo (ejemplo: 28x28 o 32x32)
        gray = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        gray = gray.reshape(1, 1, 28, 28) # Ajustar según tu modelo

        # 2. Inferencia
        result = self.session.run(None, {self.input_name: gray})
        idx = np.argmax(result[0])
        return self.labels[idx]

    def process_full_image(self, binary_img):
        """Segmenta y clasifica todas las letras de la imagen"""
        boxes = segment_characters(binary_img)
        results = []

        for (x, y, w, h) in boxes:
            crop = binary_img[y:y+h, x:x+w]
            letter = self.predict_character(crop)
            results.append({
                "letter": letter,
                "bbox": (x, y, w, h),
                "crop": crop # Para pasar a tu evaluador de esqueletos
            })
            
        return results