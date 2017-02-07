import tensorflow as tf


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]

    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1

            # [batch_size x max_word_length x embed_size x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(1, layers)
        else:
            output = layers[0]

    return output


class CharCNN(object):
    def __init__(self, args, is_training, is_testing=False):
        char_vocab_size = args.subword_vocab_size
        word_vocab_size = args.word_vocab_size
        out_vocab_size = args.out_vocab_size
        char_embed_size = args.char_dim
        batch_size = args.batch_size
        num_highway_layers = args.num_highway
        num_rnn_layers = args.num_layers
        rnn_size = args.rnn_size
        max_word_length = args.max_word_length
        kernels = [1, 2, 3, 4, 5, 6]
        # kernel_features = [50, 100, 150, 200, 200, 200, 200]
        kernel_features = [25, 50, 75, 100, 125, 150]
        num_unroll_steps = args.num_steps
        keep_prob = args.keep_prob

        self.batch_size = batch_size
        self.num_steps = num_unroll_steps

        if is_testing:
            batch_size = 1
            num_unroll_steps = 1

        assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

        self.input_data = input_ = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps, max_word_length], name="input")
        self.targets = targets = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps], name="targets")

        # print(input_.get_shape())
        # print(targets.get_shape())

        ''' First, embed characters '''
        with tf.variable_scope('Embedding'):
            char_embedding = tf.get_variable('char_embedding', [char_vocab_size, char_embed_size])

            ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
            of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
            zero embedding vector and ignores gradient updates. For that do the following in TF:
            1. after parameter initialization, apply this op to zero out padding embedding vector
            2. after each gradient update, apply this op to keep padding at zero'''
            clear_char_embedding_padding = tf.scatter_update(char_embedding, [0],
                                                             tf.constant(0.0, shape=[1, char_embed_size]))

            # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
            input_embedded = tf.nn.embedding_lookup(char_embedding, input_)
            input_embedded = tf.reshape(input_embedded, [-1, max_word_length, char_embed_size])

        ''' Second, apply convolutions '''
        # [batch_size x num_unroll_steps, cnn_size]  # where cnn_size=sum(kernel_features)
        input_cnn = tdnn(input_embedded, kernels, kernel_features)

        # print(input_cnn.get_shape())
        # print(input_cnn.get_shape()[-1])

        ''' Maybe apply Highway '''
        if num_highway_layers > 0:
            input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)

        ''' Finally, do LSTM '''
        with tf.variable_scope('LSTM'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0)
            if keep_prob != 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            if num_rnn_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_rnn_layers, state_is_tuple=True)

            self._initial_lm_state = cell.zero_state(batch_size, tf.float32)

            input_cnn = tf.reshape(input_cnn, [batch_size, num_unroll_steps, -1])
            input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(1, num_unroll_steps, input_cnn)]

            # print(len(input_cnn2))
            # print(input_cnn2[0].get_shape())

            outputs, final_rnn_state = tf.nn.rnn(cell, input_cnn2,
                                                 initial_state=self._initial_lm_state, dtype=tf.float32)

            # not perform this to compare with other models
            # if keep_prob != 1.0:
            #     outputs = [tf.nn.dropout(x, keep_prob=keep_prob) for x in outputs]

            # linear projection onto output (word) vocab
            # logits = []
            # with tf.variable_scope('WordEmbedding') as scope:
            #     for idx, output in enumerate(outputs):
            #         if idx > 0:
            #             scope.reuse_variables()
            #         logits.append(linear(output, word_vocab_size))
            outputs = tf.concat(1, outputs)
            outputs = tf.reshape(outputs, [-1, rnn_size])

            softmax_w = tf.get_variable("softmax_w", [out_vocab_size, rnn_size])
            softmax_b = tf.get_variable("softmax_b", [out_vocab_size])
            logits = tf.matmul(outputs, softmax_w, transpose_b=True) + softmax_b

        with tf.variable_scope('Loss'):
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(targets, [-1])],
                [tf.ones([batch_size * num_unroll_steps])]
            )

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = final_rnn_state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False, name="learning_rate")
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

    @property
    def initial_lm_state(self):
        return self._initial_lm_state
