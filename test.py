from model_new import AzzuNet
import tensorflow as tf
import numpy as np
import _pickle as pkl
import gzip
import os


def transform(back):
    for x in range(back.shape[0]-1):
        if x == 0:continue
        temp = back[x]
        back[x] = back[x+1]
        back[x+1] = temp

model_name = "27aug1745/NewAzzuNet-8.803449"
pkl_file = gzip.open('./dataset/data.pkl', 'rb')
data = pkl.load(pkl_file)

test = data["test"]

test_limit= 2704

test_length = test_limit

words_test = np.array(test[0])
deps_test = np.array(test[1])
labels_c_test = np.array(test[2])
labels_f_test = np.array(test[3])
labels_b_test = np.array(test[4])

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
with tf.device('/cpu:0'):
    saver = tf.train.import_meta_graph('./models/%s.meta'%model_name)
    saver.restore(sess, './models/%s'%model_name)
    graph = tf.get_default_graph()
    words = graph.get_tensor_by_name("AzzuNet/inputs/words:0")
    deps = graph.get_tensor_by_name("AzzuNet/inputs/dependencies:0")
    l_c = graph.get_tensor_by_name("AzzuNet/labels/Placeholder:0")
    l_f = graph.get_tensor_by_name("AzzuNet/labels/Placeholder_1:0")
    l_b = graph.get_tensor_by_name("AzzuNet/labels/Placeholder_2:0")
    prob = graph.get_tensor_by_name("AzzuNet/inputs/dropout_prob:0")
    
    acc_c = graph.get_tensor_by_name("AzzuNet/metrics/combined/accuracy:0")
    acc_f = graph.get_tensor_by_name("AzzuNet/metrics/forward/accuracy:0")
    acc_b = graph.get_tensor_by_name("AzzuNet/metrics/backward/accuracy:0")
    logit_f = graph.get_tensor_by_name("AzzuNet/softmax_layers/forward/logit:0")
    logit_b = graph.get_tensor_by_name("AzzuNet/softmax_layers/backward/logit:0")
    logit_c = graph.get_tensor_by_name("AzzuNet/softmax_layers/combined/logit:0")

    feed_dict = {words: words_test,
                     deps: deps_test,
                     prob: 1.0,
                     l_c: labels_c_test,
                     l_f: labels_f_test,
                     l_b: labels_b_test}

    ops = [acc_f, acc_b, acc_c]
    forward_ans, back_ans, comb_ans = sess.run(ops, feed_dict)

    print("Test accuracy:", forward_ans, back_ans, comb_ans)
