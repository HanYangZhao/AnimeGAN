
import numpy as np
import tensorflow as tf
import sys
sys.stdout.flush()
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="hayao_epoch60.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
