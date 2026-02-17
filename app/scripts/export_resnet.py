import tensorflow as tf
import tf2onnx

def export_resnet(h5_path, onnx_path):
    model = tf.keras.models.load_model(h5_path)
    # Especificamos el input para ResNet: 32x32x1
    spec = (tf.TensorSpec((None, 32, 32, 1), tf.float32, name="input"),)
    
    print("Convirtiendo ResNet a ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Exportado exitosamente a {onnx_path}")

if __name__ == "__main__":
    export_resnet("app/models/resnet_trained.h5", "app/models/resnet_ocr.onnx")