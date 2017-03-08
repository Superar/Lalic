import tensorflow as tf


sess = tf.Session()
new_saver = tf.train.import_meta_graph('modelo-encoder-decoder.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
