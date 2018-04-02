import sys
sys.path.append('..')

import tensorflow as tf
from utils import load_pretrained_word2vec, pad_common
from nn_base import Base_Classifier
import numpy as np

class Multi_Classifier(Base_Classifier):
    def __init__(self, id2word, id2label,word_emb_dim, word_hid_dim=100, max_sen_len=39,
                 use_pretrained=None, lr=0.01, dropout_keep_prob=0.5,
                 filter_sizes=[1, 2, 3, 5, 6], num_filters=10,
                 conv_type='pool',l2_reg_loss=0.001, **kargs):

        Base_Classifier.__init__(self,**kargs)

        self.id2word = id2word
        self.id2label = id2label

        self.word_emb_dim = word_emb_dim
        self.word_hid_dim = word_hid_dim

        self.use_pretrained = use_pretrained
        if self.use_pretrained is not None:
            self.NP_WORD_EMB = load_pretrained_word2vec(emb_size=self.word_emb_dim, id2word=self.id2word,
                                                        pre_emb_path=use_pretrained)
        else:
            drange = np.sqrt(6. / (len(self.id2word) + self.word_emb_dim))
            self.NP_WORD_EMB = np.random.uniform(low=-1, high=1, size=(len(self.id2word), self.word_emb_dim)) * drange

        self.NP_WORD_EMB = self.NP_WORD_EMB.astype('float32')

        self.lr = lr
        self.n_tags = len(self.id2label)
        self.dropout_keep_prob = dropout_keep_prob

        self.max_sen_len = max_sen_len
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.l2_reg_loss = l2_reg_loss

        self.conv_type = conv_type # [pool,lstm]

    def __build_shared_matrix(self):
        self.WORD_EMB = tf.get_variable(name="word_emb", shape=(self.vocab_len, self.word_emb_dim), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())

        # initialize variables for fully connected
        self.W_fn = tf.get_variable(name='W_fn', shape=(self.num_filters_total, self.num_filters_total),
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.b_fn = tf.get_variable(name="b_proj", shape=[self.num_filters_total], dtype=tf.float32,
                                    initializer=tf.constant_initializer(value=0.1))

        # initialize variables for project to num_tags
        self.W_proj = tf.get_variable(name='W_proj', shape=(self.num_filters_total, self.n_tags), dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
        self.b_proj = tf.get_variable(name="b_proj_1", shape=[self.n_tags], dtype=tf.float32,
                                      initializer=tf.constant_initializer(value=0.1))

        # l2_loss
        self.l2_loss = tf.constant(0.0)

    def __build_placeholder(self):
        self.build_additional_placeholder()

        self.vocab_len = len(self.id2word)
        self.num_filters_total = self.num_filters * len(self.filter_sizes) # len(self.filter_sizes_layer2)

        # initialize placeholder
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sen_len),
                                       name='word_ids',)  # (batch_size, max_sen_len)

        self.target = tf.placeholder(dtype=tf.float32, shape=(None, self.n_tags), name='labels')  # (batch_size, n_tags)
        self.sentence_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='sentence_len')  # (batch_size)
        self.dropout = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_prob')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
        self.lr_placeholder = tf.placeholder_with_default(self.lr,shape=(),name='lr')

        # because if pretrained file is too large for Variable, Tensorflow does not allow it to be loaded.
        self.WORD_EMB_placeholder = tf.placeholder(tf.float32,shape=(self.vocab_len,self.word_emb_dim),name='word_emb_placeholder')

    def __encode_word(self):
        word = tf.nn.embedding_lookup(self.WORD_EMB, self.word_ids)  # (batch_size, max_sen_len, emb_dim)
        add_features, add_dim = self.encode_additional_word()

        self.word_emb_dim += add_dim
        return tf.concat([word] + add_features,axis=2)

    def __build_conv(self, input):
        # expand last dimension to correspond with required parameter of tf.nn.conv2d
        input_expand = tf.expand_dims(input, axis=3)
        outputs = []

        # convolutional layer 1
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.word_emb_dim, 1, self.num_filters]
                W = tf.get_variable(name='W',shape=filter_shape,dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable(name="b",shape=[self.num_filters],dtype=tf.float32,
                                    initializer=tf.constant_initializer(value=0.1))

                conv = tf.nn.conv2d(
                    input_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Batch norm
                pre_activation = tf.nn.bias_add(conv, b) #self.batch_normalize(tf.nn.bias_add(conv, b), scope='_', phase=self.is_training)
                # Apply nonlinearity
                h = tf.nn.relu(pre_activation, name="relu")

                outputs.append(h)

        return outputs

    def __build_lstm(self, input):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.word_hid_dim)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.word_hid_dim)

        output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=input,dtype=tf.float32)

        return tf.concat(state[:][1],axis=1)

    def __norm_conv_output(self,input,min_dim):
        # convert to the same length.
        norm_conv_outputs = []
        for conv_out in input:
            conv_shape = conv_out.get_shape().as_list()
            dim = conv_shape[1]
            conv_out = tf.reshape(conv_out, shape=[-1, dim, conv_shape[-1]])

            if dim > min_dim:
                with tf.variable_scope('conv_mapping_%i_to_%i' % (dim, min_dim)):
                    distance = dim - min_dim + 1

                    W_map = tf.get_variable(name="W_map",shape=[self.num_filters * distance, self.num_filters],dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

                    b_map = tf.get_variable(name="b",shape=[self.num_filters],dtype=tf.float32,
                                            initializer=tf.constant_initializer(value=0.1))

                    # split
                    main, remainder = tf.split(conv_out, [dim - distance, distance], axis=1)

                    # dot matrix
                    reshape_remainder = tf.reshape(remainder, shape=[-1, self.num_filters * distance])
                    new_remainder = tf.nn.xw_plus_b(reshape_remainder, W_map, b_map, name='dot_new_remainder')
                    new_remainder = tf.reshape(new_remainder, shape=[-1, 1, self.num_filters])

                    res = tf.concat([main, new_remainder], axis=1)
                    norm_conv_outputs.append(res)
            else:
                norm_conv_outputs.append(conv_out)
        return norm_conv_outputs

    def __build_optimizer(self):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr_placeholder)
        self.train_op = self.optimizer.minimize(self.loss)

    def __build_loss_function(self,scores):
        self.logits = tf.nn.xw_plus_b(scores, self.W_proj, self.b_proj, name="logits")

        self.l2_loss += tf.nn.l2_loss(self.W_proj)
        self.l2_loss += tf.nn.l2_loss(self.b_proj)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target)) + self.l2_reg_loss * self.l2_loss

        predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_predictions = tf.equal(predictions, tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def __build_fully_connected(self, input):
        fn = tf.nn.xw_plus_b(input, self.W_fn, self.b_fn, name="fully_connected")
        fn = tf.nn.tanh(fn)

        return fn

    def batch_normalize(self, data, scope, phase):
        norm_data = tf.contrib.layers.batch_norm(
            data,
            decay=0.9,
            center=True,
            scale=True,
            is_training=phase,
            scope=scope + '_bn')

        return norm_data

    def build_model(self,reuse=False,build_session=True):
        #
        # build input placeholder
        #
        with tf.variable_scope('build_placeholder',reuse=reuse):
            self.__build_placeholder()
            self.__build_shared_matrix()

        #
        # embedding word
        #
        with tf.variable_scope('encoding_input',reuse=reuse):
            self.word_enc =  self.__encode_word()

            self.final_word_enc = tf.nn.dropout(self.word_enc, keep_prob=self.dropout)

        #
        # build convolutional neural network`
        #
        with tf.variable_scope('neural_net',reuse=reuse):
            with tf.variable_scope('conv'):
                conv_outs = self.__build_conv(self.final_word_enc) # list of features_map

            if self.conv_type == 'pool': # traditional cnn
                with tf.variable_scope('pool'):
                    pool_outputs = []
                    for filter_size, conv_out in zip(self.filter_sizes,conv_outs):
                        pooled = tf.nn.max_pool(
                            conv_out,
                            ksize=[1, self.max_sen_len - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool_%i" % filter_size)

                        pool_outputs.append(pooled)
                    h_pool = tf.concat(pool_outputs, 3, name='h_pool')
                self.fn_nn_emb = tf.reshape(h_pool, [-1, self.num_filters_total])  # (batch_size, num_filter_total)

            elif self.conv_type == 'lstm': # c_lstm
                with tf.variable_scope('bi_lstm'):
                    # get min dim
                    min_dim = np.min([conv_out.get_shape().as_list()[1] for conv_out in conv_outs])
                    # because we have multiple filter size, so we need to normalize to make one for all.
                    norm_conv_outputs = self.__norm_conv_output(conv_outs,min_dim)

                    norm_output = tf.concat(norm_conv_outputs,axis=2)
                    lstm_emb = self.__build_lstm(norm_output)

                    self.W_conv_proj = tf.get_variable(name='W_conv_proj',shape=[2*self.word_hid_dim,self.num_filters_total],
                                                       dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                    self.b_conv_proj = tf.get_variable(name='b_conv_proj',shape=[self.num_filters_total],
                                                       dtype=tf.float32,initializer=tf.zeros_initializer())

                    self.fn_nn_emb = tf.nn.xw_plus_b(lstm_emb,self.W_conv_proj,self.b_conv_proj,name='fully_conv_proj')
            else:
                raise NotImplementedError('Conv type must be pool or lstm')


        #self.fn_nn_emb = self.fn_nn_emb

        #
        # build fully_connected
        #
        with tf.variable_scope('fully_connected',reuse=reuse):
            self.scores = self.fn_nn_emb

        #
        # build loss
        #
        with tf.variable_scope('loss',reuse=reuse):
            self.__build_loss_function(self.scores)

        #
        # build optimizer
        #
        with tf.variable_scope('optimizer',reuse=reuse):
            self.__build_optimizer()
        ##########################################

        ##########################################
        if not build_session:
            return

        #
        # build session
        #
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        self.sess.run(init_op,feed_dict={self.WORD_EMB_placeholder:self.NP_WORD_EMB})

    def create_input(self, batch):
        max_sentence_length = self.max_sen_len  # max(map(lambda x: len(x), [e['word_ids'] for e in batch]))

        ip_word_ids, sentence_length = pad_common(sequences=[e['word_ids'] for e in batch], pad_tok=1,
                                                  max_length=max_sentence_length)

        res = {
            self.word_ids: ip_word_ids,
            self.sentence_len: sentence_length,
            self.target: [e['label_ids'] for e in batch],
        }

        return res

if __name__ == '__main__':
    pass