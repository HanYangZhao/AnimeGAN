# Run with TF 2.0+
import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds
import cv2
from glob import glob
import numpy as np
# Convert the model.

input_height = 992
input_width = 1504
h_to_w_ratio = input_height /input_width
num_calibration_steps = 20


def representative_dataset_gen():
    sample_folder = glob('{}/*.*'.format('samples'))
    data = [load_test_data(sample_file) for sample_file in tqdm(sample_folder)]
    # data = tfds.load(data)

    for _ in range(num_calibration_steps):
        image = data.pop()
        yield [image]


def load_test_data(image_path):
    img = cv2.imread(image_path).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop(img)
    img = preprocessing(img)
    img = np.expand_dims(img, axis=0)
    return img

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def crop(img):
    h, w = img.shape[:2]
    # if (h < input_height) or (w < input_width):
    #     print("Error, please make sure the image is at least" + str(input_width) + "x" + str(input_height))
    #     return;
    if (h != input_height) or (w != input_width):
        if(h > w):
            print("Error, please make sure the image is in landscape mode")
            return;
        else:
            crop_height = int(w * h_to_w_ratio)
            height_middle_point = int(h / 2)
            cropped_image = crop_image(img, 0, height_middle_point - int(crop_height / 2), w, crop_height)
            return cropped_image.astype(np.float32)
    else:
        return img
    

def preprocessing(img):
    img = cv2.resize(img, (input_width, input_height))
    print(input_width)
    print(input_height)
    return img/127.5 - 1.0


def save_images(images, image_path):
    # return imsave(inverse_transform(images), size, image_path)
    return imsave(inverse_transform(images.squeeze()).astype(np.uint8),  image_path)

def inverse_transform(images):
    return (images+1.) / 2 * 255


def imsave(images, path):
    # return misc.imsave(path, images)
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


model = tf.saved_model.load('./')
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

'''keep the ratio as close to 3/2 while being divisible by 32'''
concrete_func.inputs[0].set_shape([1, input_height, input_width, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(
    representative_dataset_gen)
tflite_model = converter.convert()
open("./converted_model.tflite","wb").write(tflite_model)
