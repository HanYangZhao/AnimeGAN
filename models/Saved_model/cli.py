
# Note especially that the image must pass from the client to the server as a Base64 encoded string. This is because JSON has no other way to represent images (besides an array representation of a tensor, and that gets out of hand very quickly).

import base64
import requests
import json
import argparse
import cv2
import numpy as np

def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return img

def preprocessing(img, size):
    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    print(w)
    print(h)
    return img/127.5 - 1.0


def save_images(images, image_path):
    # return imsave(inverse_transform(images), size, image_path)
    return imsave(inverse_transform(images.squeeze()).astype(np.uint8),  image_path)

def inverse_transform(images):
    return (images+1.) / 2 * 255


def imsave(images, path):
    # return misc.imsave(path, images)
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Open and read image np array
input_image = load_test_data("test_image.jpg", [256,256])

data = json.dumps({"instances": input_image},cls=NumpyEncoder)

json_response = requests.post("http://localhost:8501/v1/models/animegan_hayao:predict", data=data)
# Extract text from JSON
response = json.loads(json_response.text)['predictions']

response_array = np.asarray(response)

# Save inferred image
save_images(response_array, "response_image.jpg")





