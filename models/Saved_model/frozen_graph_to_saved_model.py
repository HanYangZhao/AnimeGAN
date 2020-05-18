import tensorflow as tf
from tensorflow.python import ops

def get_graph_def_from_file(graph_filepath):
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def convert_graph_def_to_saved_model(export_dir, graph_filepath):
  # if tf.gfile.Exists(export_dir):
  #   tf.gfile.DeleteRecursively(export_dir)
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.saved_model.simple_save(
        session,
        export_dir,
        inputs={
            node.name: session.graph.get_tensor_by_name(
                'test_real_A:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={'class_ids': session.graph.get_tensor_by_name(
            'generator_1/G_MODEL/Tanh:0')}
    )
    print('Optimized graph converted to SavedModel!')

convert_graph_def_to_saved_model('./saved_model','./Hayao_frozen_graph_epoch60.pb')