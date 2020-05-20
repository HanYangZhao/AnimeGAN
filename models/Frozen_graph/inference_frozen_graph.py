import os
import argparse
import tensorflow as tf
import cv2
from tqdm import tqdm
from glob import glob
import time
import numpy as np
import generator
import utils

def infer(model_filepath,test_dir,output_dir,img_size=[256,256]):
    '''
    Lode trained model.
    '''
    graph_def = None
    print('Loading model...')
    graph = tf.Graph()

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

    result_dir = 'results/'+output_dir
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(test_dir))
    # Know your output node name
    output_tensor = graph.get_tensor_by_name("import/generator_1/G_MODEL/Tanh:0")


    for sample_file  in tqdm(test_files) :
        # print('Processing image: ' + sample_file)
        sample_image = np.asarray(load_test_data(sample_file, img_size))
        image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))
        fake_img = sess.run(output_tensor, feed_dict = {test_real : sample_image})
        save_images(fake_img, image_path)


def parse_args():
    desc = "Tensorflow implementation of AnimeGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_path', type=str, default='frozen_model.pb',
                        help='Frozen Model path')
    parser.add_argument('--test_dir', type=str, default='',
                        help='Directory name of test photos')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='what style you want to get')

    """checking arguments"""

    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_args()
    # Initialize the model
    infer(arg.model_path,arg.test_dir,arg.output_dir)



