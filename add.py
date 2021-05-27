#!/usr/bin/python
# Author: Clara Vania

import tensorflow as tf


class AdditiveModel(object):
    """
    RNNLM using subword to word (S2W) model
    Code based on tensorflow tutorial on building a PTB LSTM model.
    https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html
    """
    def __init__(self, args, is_training, is_testing=False):
        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.model = model = args.model
        self.subword_vocab_size = subword_vocab_size = args.subword_vocab_size
        self.optimizer = args.optimization
        self.unit = args.unit

        rnn_size = args.rnn_size
        out_vocab_size = args.out_vocab_size
        tf_device = "/gpu:" + str(args.gpu)

        if is_testing:
            self.batch_size = batch_size = 1
            self.num_steps = num_steps = 1

        if model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        with tf.device(tf_device):
            # placeholders for data
            self._input_data = tf.placeholder(tf.float32, shape=[batch_size, num_steps, subword_vocab_size])
            self._targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps])

            # ********************************************************************************
            # RNNLM
            # ********************************************************************************

            lm_cell = cell_fn(rnn_size, forget_bias=0.0)
            if is_training and args.keep_prob < 1:
                lm_cell = tf.nn.rnn_cell.DropoutWrapper(lm_cell, output_keep_prob=args.keep_prob)
            lm_cell = tf.nn.rnn_cell.MultiRNNCell([lm_cell] * args.num_layers)

            self._initial_lm_state = lm_cell.zero_state(batch_size, tf.float32)

            inputs = self._input_data
            if is_training and args.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, args.keep_prob)

            softmax_win = tf.get_variable("softmax_win", [subword_vocab_size, rnn_size])
            softmax_bin = tf.get_variable("softmax_bin", [rnn_size])

            # split input into a list
            inputs = tf.split(inputs, num_steps, 1)
            lm_inputs = []
            for input_ in inputs:
                input_ = tf.squeeze(input_, [1])
                input_ = tf.matmul(input_, softmax_win) + softmax_bin
                lm_inputs.append(input_)
            lm_inputs = tf.stack(lm_inputs, axis=1)
            # print(lm_inputs)
            # print("===============")

            # print(inputs)
            lm_outputs, lm_state = tf.nn.dynamic_rnn(lm_cell, lm_inputs, initial_state=self._initial_lm_state)
            # lm_outputs, lm_state = tf.nn.dynamic_rnn(lm_cell, tf.reshape(lm_inputs, [batch_size, num_steps, rnn_size]), initial_state=self._initial_lm_state) #expected (rnn_size, batch_size)
            lm_outputs = tf.concat(lm_outputs, 1)
            lm_outputs = tf.reshape(lm_outputs, [-1, rnn_size])

            softmax_w = tf.get_variable("softmax_w", [out_vocab_size, rnn_size])
            softmax_b = tf.get_variable("softmax_b", [out_vocab_size])

            # compute cross entropy loss
            logits = tf.matmul(lm_outputs, softmax_w, transpose_b=True) + softmax_b

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.reshape(self._targets, [-1])
            )
            # loss = tfl.sequence_loss_by_example(
            #     [logits],
            #     [tf.reshape(self._targets, [-1])],
            #     [tf.ones([batch_size * num_steps])]
            # )

            # compute cost
            self._cost = cost = tf.reduce_sum(loss) / batch_size
            self._final_state = lm_state

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
