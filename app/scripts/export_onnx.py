import tensorflow as tf
import tf2onnx
import onnx

def convert_h5_to_onnx(h5_path, output_path):
    # 1. Cargar el modelo de Keras
    model = tf.keras.models.load_model(h5_path)
    
    # 2. Definir la firma de entrada (Input Spec)
    # Asumiendo 28x28 en escala de grises
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    
    # 3. Convertir
    print(f"Convirtiendo {h5_path} a ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    # 4. Guardar
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"Modelo guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    # Ejemplo de uso
    convert_h5_to_onnx("app/models/mi_modelo_letras.h5", "app/models/ocr_letter_model.onnx")