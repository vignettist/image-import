from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import time

class Classifier:

    def __init__(self, prefix):
        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        self.PREFIX_DIR = prefix

        # classify_image_graph_def.pb:
        #   Binary representation of the GraphDef protocol buffer.
        # imagenet_synset_to_human_label_map.txt:
        #   Map from synset ID to a human readable string.
        # imagenet_2012_challenge_label_map_proto.pbtxt:
        #   Text representation of a protocol buffer mapping a label to synset ID.
        tf.app.flags.DEFINE_string(
            'model_dir', '/Users/loganw/Documents/models/imagenet',
            """Path to classify_image_graph_def.pb, """
            """imagenet_synset_to_human_label_map.txt, and """
            """imagenet_2012_challenge_label_map_proto.pbtxt.""")

        self.FLAGS = tf.app.flags.FLAGS

        self.maybe_download_and_extract()


    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(os.path.join(self.FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def maybe_download_and_extract(self):
        """Download and extract model tar file.

        If the pretrained model we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a directory.
        """
        dest_directory = self.FLAGS.model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                               (filename,
                                float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                     filepath,
                                                     _progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)


    def run_inference_on_images(self, images):
        # Creates graph from saved GraphDef.
        self.create_graph()

        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            pool_tensor = sess.graph.get_tensor_by_name('pool_3:0')

            # print(softmax_tensor)
            
            t = time.time()
            for i in range(len(images)):
                if (i % 100) == 0:
                    elapsed = time.time() - t
                    t = time.time()
                    
                    print( str(i) + '/' + str(len(images)) + ', ' + str(elapsed/100.0) + ' seconds per image')
                    
                image_data = tf.gfile.FastGFile(self.PREFIX_DIR + images[i]['resized_uris']['1280'], 'rb').read()
                
                (predictions, pools) = sess.run((softmax_tensor, pool_tensor),
                                   {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions).tolist()
                pools = np.squeeze(pools).tolist()

                images[i]['inception_pool'] = pools
                images[i]['inception_classification'] = predictions

        return images