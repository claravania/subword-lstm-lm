#!/usr/bin/python
# Author: Clara Vania

import tensorflow as tf


class WordModel(object):
    """
    RNNLM with LSTM + Dropout
    Code based on tensorflow tutorial on building a PTB LSTM model.
    https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html
    """
    def __init__(self, args, is_training, is_testing=False):

        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.model = model = args.model
        self.optimizer = args.optimization
        self.unit = args.unit

        word_dim = args.word_dim
        rnn_size = args.rnn_size
        rnn_cell = tf.nn.rnn_cell
        word_vocab_size = args.word_vocab_size
        out_vocab_size = args.out_vocab_size
        tf_device = "/gpu:" + str(args.gpu)

        if is_testing:
            self.batch_size = batch_size = 1
            self.num_steps = num_steps = 1

        if model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # placeholders for data
        self._input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps])

        # ********************************************************************************
        # RNNLM
        # ********************************************************************************
        # with tf.device(tf_device):
        cell = cell_fn(rnn_size, forget_bias=0.0)
        if is_training and args.keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)
        lm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self._initial_lm_state = lm_cell.zero_state(batch_size, tf.float32)
        self.embedding = tf.get_variable("embedding", [word_vocab_size, word_dim])

        with tf.device("/cpu:0"):
            inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)

        self.input_vectors = inputs
        if is_training and args.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, args.keep_prob)

        # split input into a list
        lm_inputs = tf.split(1, num_steps, inputs)

        if word_dim == rnn_size:
            lm_inputs = [tf.squeeze(input_, [1]) for input_ in lm_inputs]
        else:
            softmax_win = tf.get_variable("softmax_win", [word_dim, rnn_size])
            softmax_bin = tf.get_variable("softmax_bin", [rnn_size])
            inputs = []
            for input_ in lm_inputs:
                input_ = tf.squeeze(input_, [1])
                input_ = tf.matmul(input_, softmax_win) + softmax_bin
                inputs.append(input_)
            lm_inputs = inputs

        lm_outputs, lm_state = tf.nn.rnn(lm_cell, lm_inputs, initial_state=self._initial_lm_state)
        lm_outputs = tf.concat(1, lm_outputs)
        lm_outputs = tf.reshape(lm_outputs, [-1, rnn_size])

        # Weights for softmax function
        softmax_w = tf.get_variable("softmax_w", [out_vocab_size, rnn_size])
        softmax_b = tf.get_variable("softmax_b", [out_vocab_size])

        # compute cross entropy loss
        logits = tf.matmul(lm_outputs, softmax_w, transpose_b=True) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        # print('Logits shape:', logits.get_shape())

        # compute cost
        self.per_word_pp = loss
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = lm_state
        # print('Loss shape:', loss.get_shape())

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          args.grad_clip)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_lm_state(self):
        return self._initial_lm_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
