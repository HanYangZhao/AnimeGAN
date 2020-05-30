
import numpy as np
import tensorflow as tf
import sys
import traceback
import cv2
import argparse
import time
sys.path.append("../../")


input_width = 1504
input_height = 992
h_to_w_ratio = input_height / input_width


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
    if (h < input_height) or (w < input_width):
        print("Error, please make sure the image is at least" + str(input_width) + "x" + str(input_height))
        return;
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


def parse_args():
    desc = "Tensorflow lite model to infer images in landscape mode"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input_filepath', type=str, default='input.jpg', help='input filepath')
    parser.add_argument('--output_filepath', type=str, default='output.jpg', help='output filepath')
    parser.add_argument('--model_path',type=str, default='hayao_epoch60_float16.tflite', help='model path')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    start = time.time()
    args = parse_args()
    if args is None:
        print("error")
        exit()

    try:
        # Load TFLite model and allocate tensors.f
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    except:
        traceback.print_exc()

    test_img = np.asarray(load_test_data(args.input_filepath))
    interpreter.set_tensor(input_details[0]['index'], test_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    save_images(output_data, args.output_filepath)

    # end time
    end = time.time()

    # total time taken
    print(f"Runtime of the program is {end - start}")