#!/usr/bin/python
# Author: Clara Vania

import tensorflow as tf


class BiLSTMModel(object):
    """
    RNNLM using subword to word (C2W) model:
    http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf
    Code based on tensorflow tutorial on building a PTB LSTM model:
    https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html
    """
    def __init__(self, args, is_training, is_testing=False, keep_num_step=False):

        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.bilstm_num_steps = bilstm_num_steps = args.bilstm_num_steps
        self.optimizer = args.optimization
        self.unit = args.unit

        model = args.model
        rnn_size = args.rnn_size
        word_dim = args.word_dim
        subword_vocab_size = args.subword_vocab_size
        out_vocab_size = args.out_vocab_size
        rnn_state_size = rnn_size
        tf_device = "/gpu:" + str(args.gpu)

        if args.unit == 'char':
            subword_dim = args.char_dim
        elif args.unit == 'char-ngram' or args.unit == 'morpheme' or args.unit == 'oracle':
            subword_dim = args.morph_dim

        if is_testing:
            self.batch_size = batch_size = 1
            if not keep_num_step:
                self.num_steps = num_steps = 1

        if model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            rnn_state_size = 2 * rnn_size
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # ********************************************************************************
        # C2W Model
        # ********************************************************************************

        # placeholders for data
        self._input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps, bilstm_num_steps])
        self._targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps])

        with tf.device(tf_device):
            with tf.variable_scope("c2w"):
                # LSTM cell for C2W, forward and backward
                with tf.variable_scope("forward"):
                    c2w_fw_cell = cell_fn(rnn_size, input_size=subword_dim, forget_bias=0.0)
                    if is_training and args.keep_prob < 1:
                        c2w_fw_cell = tf.nn.rnn_cell.DropoutWrapper(c2w_fw_cell, output_keep_prob=args.keep_prob)
                    self._initial_fw_state = c2w_fw_cell.zero_state(num_steps, tf.float32)
                with tf.variable_scope("backward"):
                    c2w_bw_cell = cell_fn(rnn_size, input_size=subword_dim, forget_bias=0.0)
                    if is_training and args.keep_prob < 1:
                        c2w_bw_cell = tf.nn.rnn_cell.DropoutWrapper(c2w_bw_cell, output_keep_prob=args.keep_prob)
                    self._initial_bw_state = c2w_bw_cell.zero_state(num_steps, tf.float32)

                # character embedding
                char_embedding = tf.get_variable("char_embedding", [subword_vocab_size, subword_dim])
                with tf.device("/cpu:0"):
                    inputs = tf.nn.embedding_lookup(char_embedding, self._input_data)

                if is_training and args.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, args.keep_prob)

                # print(inputs.get_shape())

                inputs = tf.split(0, batch_size, inputs)
                inputs = [tf.squeeze(input_, [0]) for input_ in inputs]

                c2w_outputs = []
                # Weight matrix to transform C2W outputs to have dimension word_dim
                # This is the D parameter in the paper
                softmax_w_fw = tf.get_variable("softmax_fw", [rnn_state_size, word_dim])
                softmax_w_bw = tf.get_variable("softmax_bw", [rnn_state_size, word_dim])
                b_c2w = tf.get_variable("c2w_biases", [word_dim])

                fw_state = self._initial_fw_state
                bw_state = self._initial_bw_state
                # process each word in the sentence
                for i in range(len(inputs)):
                    # reuse variable for each sentence, except for the first one
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    # process current sentence
                    input_ = inputs[i]

                    # split sentence into a sequence of chars
                    # here the sequence length is the bilstm_num_steps
                    # and the batch size is the number of words in the sentence
                    chars = tf.split(1, bilstm_num_steps, input_)
                    chars = [tf.squeeze(char_, [1]) for char_ in chars]

                    # run bi-rnn
                    c2w_output, fw_state, bw_state = tf.nn.bidirectional_rnn(c2w_fw_cell, c2w_bw_cell, chars,
                                                                                     initial_state_fw=fw_state,
                                                                                     initial_state_bw=bw_state)
                    # compute the word representation
                    # print fw_state.get_shape()
                    # print c2w_output[0].get_shape()
                    fw_param = tf.matmul(fw_state, softmax_w_fw)
                    bw_param = tf.matmul(bw_state, softmax_w_bw)
                    final_output = fw_param + bw_param + b_c2w
                    c2w_outputs.append(tf.expand_dims(final_output, 0))
                self._final_fw_state = fw_state
                self._final_bw_state = bw_state
                c2w_outputs = tf.concat(0, c2w_outputs)
                # print(c2w_outputs.get_shape())

            # ********************************************************************************
            # RNNLM
            # ********************************************************************************
            with tf.variable_scope("rnnlm"):
                cell = cell_fn(rnn_size, forget_bias=0.0)
                if is_training and args.keep_prob < 1:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)
                lm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                self._initial_lm_state = lm_cell.zero_state(batch_size, tf.float32)

                # split input into a list
                self.input_vectors = c2w_outputs
                inputs = tf.split(1, num_steps, c2w_outputs)

                if word_dim == rnn_size:
                    lm_inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                    # self.emb = c2w_outputs
                else:
                    softmax_win = tf.get_variable("softmax_win", [word_dim, rnn_size])
                    softmax_bin = tf.get_variable("softmax_bin", [rnn_size])

                    lm_inputs = []
                    for input_ in inputs:
                        input_ = tf.squeeze(input_, [1])
                        input_ = tf.matmul(input_, softmax_win) + softmax_bin
                        lm_inputs.append(input_)
                    # self.emb = tf.concat(0, lm_inputs)

                lm_outputs, lm_state = tf.nn.rnn(lm_cell, lm_inputs, initial_state=self._initial_lm_state)
                lm_outputs = tf.concat(1, lm_outputs)
                lm_outputs = tf.reshape(lm_outputs, [-1, rnn_size])

                softmax_w = tf.get_variable("softmax_w", [out_vocab_size, rnn_size])
                softmax_b = tf.get_variable("softmax_b", [out_vocab_size])

                # compute cross entropy loss
                logits = tf.matmul(lm_outputs, softmax_w, transpose_b=True) + softmax_b
                loss = tf.nn.seq2seq.sequence_loss_by_example(
                    [logits],
                    [tf.reshape(self._targets, [-1])],
                    [tf.ones([batch_size * num_steps])])

                # compute cost
                self.per_word_pp = loss
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
    def initial_fw_state(self):
        return self._initial_fw_state

    @property
    def initial_bw_state(self):
        return self._initial_bw_state

    @property
    def initial_lm_state(self):
        return self._initial_lm_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_fw_state(self):
        return self._final_fw_state

    @property
    def final_bw_state(self):
        return self._final_bw_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
