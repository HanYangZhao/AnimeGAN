

import tensorflow as tf
import pprint

saver = tf.train.import_meta_graph('AnimeGAN.model-42.meta')
sess = tf.Session()
saver.restore(sess, 'AnimeGAN.model-42')
graph = sess.graph
nodes = [node.name for node in graph.as_graph_def().node]
with open("nodes.txt", "w") as fout:
	fout.write(pprint.pformat(nodes))
