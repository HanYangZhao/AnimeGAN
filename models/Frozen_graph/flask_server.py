import os
from flask import Flask, flash, request, redirect, url_for, render_template
import tensorflow as tf
import cv2
import time
import numpy as np
import sys
sys.path.append("../../")
from utils import *
from werkzeug.utils import secure_filename
import uuid as unique_id
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
from pathlib import Path
#Initialize Tensorflow 

graph_def = None
print('Loading model...')
graph = tf.Graph()

model_filepath = "./hayao_frozen_graph_epoch60.pb"

with tf.gfile.GFile(model_filepath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print('Check out the input placeholders:')
nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
for node in nodes:
    print(node)

test_real = None
with graph.as_default():
    # Define input tensor
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test_real_A')
    tf.import_graph_def(graph_def, {'test_real_A': test_real})

graph.finalize()

print('Model loading complete!')

sess = tf.Session(graph = graph)

# Know your output node name
output_tensor = graph.get_tensor_by_name("import/generator_1/G_MODEL/Tanh:0")

app = Flask(__name__,static_folder = "converted")
app.config['UPLOADED_FOLDER'] = "uploaded"
app.config['CONVERTED_FOLDER'] = "converted"
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB Max
Path("uploaded").mkdir(parents=True, exist_ok=True)
Path("converted").mkdir(parents=True, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        uuid = uuid.uuid4()
        uploaded_filepath = os.path.join(app.config['UPLOADED_FOLDER'], uuid)
        file.save(uploaded_filepath)
        sample_image = np.asarray(load_test_data(uploaded_filepath, [256,256]))
        fake_img = sess.run(output_tensor, feed_dict = {test_real : sample_image})
    return cv2.cvtColor(inverse_transform(images.squeeze()).astype(np.uint8),cv2.COLOR_BGR2RGB)


@app.route('/uploaded/<converted_image>')
def uploaded(converted_image):
    print(converted_image)
    return render_template('show.html', converted_image=converted_image)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filetype = file.filename.split('.')[1]
            uuid = str(unique_id.uuid4()) + '.' + filetype
            uploaded_filepath = os.path.join(app.config['UPLOADED_FOLDER'], uuid)
            file.save(uploaded_filepath)
            sample_image = np.asarray(load_test_data(uploaded_filepath, [256,256]))
            fake_img = sess.run(output_tensor, feed_dict = {test_real : sample_image})
            print(uuid)
            converted_filepath = os.path.join(app.config['CONVERTED_FOLDER'], uuid)
            save_images(fake_img,converted_filepath)
        return redirect(url_for('uploaded', converted_image=uuid))
    return '''
    <!doctype html>
    <title>AnimeGan Converter</title>
    <h1>Upload new File (2MB Max)</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080)