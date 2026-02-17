import cv2
import numpy as np
import onnxruntime as ort

class ResNetOCR:
    def __init__(self, model_path="app/models/resnet_ocr.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # Mapeo: 0-9, A-Z, a-z, Ñ, ñ
        self.labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÑñ"

    def predict(self, binary_crop):
        # 1. Preprocesar para ResNet (32x32)
        # Padding para mantener relación de aspecto y no deformar la letra
        h, w = binary_crop.shape
        size = max(h, w)
        pad_img = np.zeros((size, size), dtype=np.uint8)
        off_h, off_w = (size-h)//2, (size-w)//2
        pad_img[off_h:off_h+h, off_w:off_w+w] = binary_crop
        
        char_img = cv2.resize(pad_img, (32, 32), interpolation=cv2.INTER_AREA)
        char_img = char_img.astype(np.float32) / 255.0
        char_img = char_img.reshape(1, 32, 32, 1)

        # 2. Inferencia ONNX
        preds = self.session.run(None, {self.input_name: char_img})[0]
        idx = np.argmax(preds)
        return self.labels[idx], float(preds[0][idx])

    def segment_and_classify(self, full_binary):
        contours, _ = cv2.findContours(full_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 6 or h < 6: continue
            
            crop = full_binary[y:y+h, x:x+w]
            char, conf = self.predict(crop)
            
            detections.append({
                "char": char,
                "confidence": conf,
                "bbox": (x, y, w, h),
                "binary_crop": crop
            })
        
        detections.sort(key=lambda d: d["bbox"][0])
        return detections