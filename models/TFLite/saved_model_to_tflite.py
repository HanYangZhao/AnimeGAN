# Run with TF 2.0+
import tensorflow as tf
# Convert the model.

model = tf.saved_model.load('./')
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 2000, 1333, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])


# converter = tf.lite.TFLiteConverter.from_saved_model('./')
tflite_model = converter.convert()
open("./converted_model.tflite","wb").write(tflite_model)
