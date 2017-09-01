import tensorflow as tf
import numpy as np

class AzzuNet:
    def convolution(self, x, w, b):
        x = tf.expand_dims(x, 2)
        xw = tf.nn.conv1d(x, w, stride=1, padding="VALID")
        xwb = tf.squeeze(tf.add(xw, b), 2)
        return tf.nn.tanh(xwb)

    def cnn_block(self, name, w, b, word, deps, length):
        with tf.variable_scope(name):
            word = tf.transpose(word, perm=[1,0,2])
            deps = tf.transpose(deps, perm=[1,0,2])
            word_start = tf.gather(word, np.arange(length), name = "words_start")
            word_end = tf.gather(word, np.arange(1, length+1), name = "words_end")
            d_units = tf.concat([word_start, deps, word_end], axis = 2, name = "d_units")
            local_features = tf.map_fn(lambda x: self.convolution(x,w,b), d_units, dtype=tf.float32)
            local_features = tf.transpose(local_features, perm=[1, 0, 2])
        return local_features

    def global_max_pool(self, name, local_f):
        with tf.variable_scope(name):
            global_features = tf.reduce_max(local_f, reduction_indices=[2], name="global_features")
        return global_features

    def lstm_block(self, name, length, data):
        with tf.variable_scope(name):
            cell = tf.contrib.rnn.LSTMCell(length, state_is_tuple=True)
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        return tf.identity(output, name="output")

    def softmax_layer(self, name, w, b, x, y_):
        with tf.variable_scope(name):
            y = tf.add(tf.matmul(x, w), b, name="logit")
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), name="loss")
        return y, loss

    def accuracy(self, name, y, y_):
        with tf.variable_scope(name):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_prediction")
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        return accuracy

    def __init__(self, input_length, embedding_length_word, embedding_length_dep):

        w_shape = [200, 1, 1]

        with tf.variable_scope('AzzuNet'):
            with tf.variable_scope('variables'):
                with tf.variable_scope('forward'):
                    w_con_f = tf.Variable(tf.random_normal(w_shape), name="w_con", dtype=tf.float32)
                    b_con_f = tf.Variable(tf.ones([1]), name="b_con", dtype=tf.float32)
                with tf.variable_scope('backward'):
                    w_con_b = tf.Variable(tf.random_normal(w_shape), name="w_con", dtype=tf.float32)
                    b_con_b = tf.Variable(tf.ones([1]), name="b_con", dtype=tf.float32)
                w_f = tf.Variable(tf.random_normal([input_length-1, 19]), dtype=tf.float32, name="w_f")
                b_f = tf.Variable(tf.ones([19]), dtype=tf.float32, name="b_f")
                w_c = tf.Variable(tf.random_normal([2*input_length-2, 10]), dtype=tf.float32, name="w_c")
                b_c = tf.Variable(tf.ones([10]), dtype=tf.float32, name="b_c")

            with tf.variable_scope('labels'):
                self.l_c = tf.placeholder(shape=[None, 10], dtype=tf.float32)
                self.l_f = tf.placeholder(shape=[None, 19], dtype=tf.float32)
                self.l_b = tf.placeholder(shape=[None, 19], dtype=tf.float32)

            with tf.variable_scope('inputs'):
                self.words = tf.placeholder(shape=[None, input_length, embedding_length_word], dtype=tf.float32, name='words')
                self.deps = tf.placeholder(shape=[None, input_length-1, embedding_length_dep], dtype=tf.float32, name='dependencies')
                words_rev = tf.reverse(self.words, axis = [1], name='words_reverse')
                deps_rev = tf.reverse(self.deps, axis = [1], name='dependencies_reverse')
                self.lr = tf.placeholder(shape=[], dtype=tf.float32, name="learning_rate")
                self.reg = tf.placeholder(shape=[], dtype=tf.float32, name="regularization")

            with tf.variable_scope('bidirectional_lstm'):
                lstm_word_f = self.lstm_block('word_forward', embedding_length_word, self.words)
                lstm_word_b = self.lstm_block('word_backward', embedding_length_word, words_rev)
                lstm_deps_f = self.lstm_block('deps_forward', embedding_length_dep, self.deps)
                lstm_deps_b = self.lstm_block('deps_backward', embedding_length_dep, deps_rev)

            with tf.variable_scope('convolution_network'):
                local_features_f = self.cnn_block('forward', w_con_f, b_con_f, lstm_word_f, lstm_deps_f, input_length-1)
                local_features_b = self.cnn_block('backward', w_con_b, b_con_b, lstm_word_b, lstm_deps_b, input_length-1)

            with tf.variable_scope('max_pooling'):
                g_f = self.global_max_pool('forward', local_features_f)
                g_b = self.global_max_pool('backward', local_features_b)

            g_c = tf.concat([g_f, g_b], axis=1, name="global_features")

            with tf.variable_scope('softmax_layers'):
                self.y_f, self.loss_f = self.softmax_layer('forward', w_f, b_f, g_f, self.l_f)
                self.y_b, self.loss_b = self.softmax_layer('backward', w_f, b_f, g_b, self.l_b)
                self.y_c, self.loss_c = self.softmax_layer('combined', w_c, b_c, g_c, self.l_c)

            with tf.variable_scope('losses'):
                self.loss = self.loss_f + self.loss_b + self.loss_c
                reg_loss = tf.nn.l2_loss(w_con_f)+tf.nn.l2_loss(w_con_b)+tf.nn.l2_loss(w_f)+tf.nn.l2_loss(w_c)
                total_loss = self.loss + self.reg * reg_loss
                self.total_loss = tf.identity(total_loss, name="total_loss")

            with tf.variable_scope('metrics'):
                self.acc_f = self.accuracy('forward', self.y_f, self.l_f)
                self.acc_b = self.accuracy('backward', self.y_b, self.l_b)
                self.acc_c = self.accuracy('combined', self.y_c, self.l_c)

            with tf.name_scope('summaries'):
                with tf.name_scope('forward'):
                    tf.summary.scalar('loss', self.loss_f)
                    tf.summary.scalar('acc', self.acc_f)
                with tf.name_scope('backward'):
                    tf.summary.scalar('loss', self.loss_b)
                    tf.summary.scalar('acc', self.acc_b)
                with tf.name_scope('combined'):
                    tf.summary.scalar('loss', self.loss_c)
                    tf.summary.scalar('acc', self.acc_c)
                self.summary = tf.summary.merge_all()

        self.train_step = tf.train.AdadeltaOptimizer(self.lr).minimize(total_loss)
