import sys
sys.path.append('..')

import tensorflow as tf
from utils import load_pretrained_word2vec, pad_common
import numpy as np
from nn_base import Base_Classifier

class CustomLSTM(tf.contrib.rnn.LSTMCell):
    def __init__(self, *args, **kwargs):
        super(CustomLSTM, self).__init__(*args, **kwargs) # create an lstm cell
        self._output_size = self._state_size # change the output size to the state size

    def __call__(self, inputs, state):
        output, next_state = super(CustomLSTM, self).__call__(inputs, state)
        return next_state, next_state

class Multi_Classifier(Base_Classifier):
    def __init__(self, id2word, id2label, word_emb_dim, word_hid_dim=100, max_sen_len=39,
                 use_pretrained=None, lr=0.01, dropout_keep_prob=0.5, l2_reg_loss=0.001, **kargs):

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
            self.NP_WORD_EMB = np.random.uniform(low=-0.25,high=0.25,size=(len(self.id2word),self.word_emb_dim))
        self.NP_WORD_EMB = self.NP_WORD_EMB.astype('float32')

        self.lr = lr
        self.n_tags = len(self.id2label)
        self.dropout_keep_prob = dropout_keep_prob

        self.max_sen_len = max_sen_len

        self.l2_reg_loss = l2_reg_loss

        self.freq_summary = 50

    def __build_shared_matrix(self):
        self.WORD_EMB = tf.get_variable(name='word_emb', shape=[self.vocab_len, self.word_emb_dim], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.25))

        self.WORD_EMB.assign(self.WORD_EMB_placeholder)

        # for attention
        self.W_att = tf.get_variable(name='W_att', shape=(2 * self.word_hid_dim, 2 * self.word_hid_dim),
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
        self.b_att = tf.get_variable(name='b_att', shape=(2 * self.word_hid_dim), dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
        self.U_p = tf.get_variable(name='U_p', shape=(2 * self.word_hid_dim, 1), dtype=tf.float32,
                                   initializer=tf.zeros_initializer())

        # initialize variables for fully connected
        self.W_fn = tf.get_variable(name='W_fn', shape=(2 * self.word_hid_dim, self.word_hid_dim),
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.b_fn = tf.get_variable(name="b_proj", shape=[self.word_hid_dim], dtype=tf.float32,
                                    initializer=tf.constant_initializer(value=0.1))

        # initialize variables for project to num_tags
        self.W_proj = tf.get_variable(name='W_proj', shape=(self.word_hid_dim, self.n_tags), dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
        self.b_proj = tf.get_variable(name="b_proj_1", shape=[self.n_tags], dtype=tf.float32,
                                      initializer=tf.constant_initializer(value=0.1))

        # l2_loss
        self.l2_loss = tf.constant(0.0)

    def __build_placeholder(self):
        # for adding additional features
        self.build_additional_placeholder()

        self.vocab_len = len(self.id2word)

        # initialize placeholder
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sen_len),
                                       name='word_ids')  # (batch_size, max_sen_len)
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, self.n_tags), name='labels')  # (batch_size, n_tags)
        self.sentence_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='sentence_len')  # (batch_size)
        self.dropout = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_prob')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
        self.lr_placeholder = tf.placeholder_with_default(self.lr, shape=(), name='lr')

        # because if pretrained file is too large for Variable, Tensorflow does not allow it to be loaded.
        self.WORD_EMB_placeholder = tf.placeholder(tf.float32,shape=(self.vocab_len,self.word_emb_dim),name='word_emb_placeholder')

    def __encode_word(self):
        word = tf.nn.embedding_lookup(self.WORD_EMB, self.word_ids)  # (batch_size, max_sen_len, emb_dim)
        add_features, add_dim = self.encode_additional_word()

        self.word_emb_dim += add_dim
        return tf.concat([word] + add_features,axis=2)


    def __build_lstm(self, input, all_states=False):

        if all_states == False:
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.word_hid_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.word_hid_dim)
        else:
            cell_fw = CustomLSTM(num_units=self.word_hid_dim)
            cell_bw = CustomLSTM(num_units=self.word_hid_dim)

        if all_states == False:
            output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=input,
                                                            dtype=tf.float32,sequence_length=self.sentence_len)
            return tf.concat(state[:][1],axis=1)
        else:
            with tf.variable_scope('lstm_forward'):
                fw_output,_ = tf.nn.dynamic_rnn(cell=cell_fw,inputs=input,dtype=tf.float32,
                                                sequence_length=self.sentence_len)
            # reverse input
            reverse_input = tf.reverse_sequence(input=input,seq_lengths=self.sentence_len,seq_dim=1,batch_dim=0)
            with tf.variable_scope('lstm_backward'):
                bw_output,_ = tf.nn.dynamic_rnn(cell=cell_bw,inputs=reverse_input,dtype=tf.float32,
                                                sequence_length=self.sentence_len)

            return tf.concat([fw_output[1],bw_output[1]],axis=2)

    def __build_attention(self, input):
        # reshape input
        input_shape = input.get_shape().as_list()
        rs_input = tf.reshape(input, shape=(-1, input_shape[-1]))
        # calculate scores
        scores = tf.tanh(tf.nn.xw_plus_b(rs_input, self.W_att, self.b_att))
        scores = tf.matmul(scores,self.U_p)
        scores = tf.reshape(scores,shape=(-1,input_shape[1]))
        # calculate alpha
        alpha = tf.nn.softmax(scores)
        # calculate context vector
        c = tf.reduce_sum(tf.expand_dims(alpha,axis=2) * input, axis=1)

        return c

    def __build_optimizer(self):
        self.optimizer = tf.train.RMSPropOptimizer(self.lr_placeholder)
        self.train_op = self.optimizer.minimize(self.loss)

    def __build_loss_function(self,scores):
        self.logits = tf.nn.xw_plus_b(scores, self.W_proj, self.b_proj, name="logits")  # (batch_size, n_tags)

        self.l2_loss += tf.nn.l2_loss(self.W_proj)
        self.l2_loss += tf.nn.l2_loss(self.b_proj)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target)) \
                    + self.l2_reg_loss * self.l2_loss

        self.loss += self.create_additional_loss()

        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def __build_fully_connected(self, input):
        fn = tf.nn.xw_plus_b(input, self.W_fn, self.b_fn, name="fully_connected")

        return fn

    def build_model(self,reuse=False,build_session=True):
        #
        # build input placeholder
        #
        with tf.variable_scope('build_placeholder',reuse=reuse):
            self.__build_placeholder()
            self.__build_shared_matrix()

        #
        # embedding word then encode using bi-lstm
        #
        with tf.variable_scope('encoding_input',reuse=reuse):
            self.word_enc = self.__encode_word()

            self.final_word_enc = self.word_enc
            self.final_word_enc = tf.nn.dropout(self.final_word_enc, keep_prob=self.dropout)

            self.lstm_out = self.__build_lstm(input=self.final_word_enc,all_states=True)

        #
        # attention
        #
        with tf.variable_scope('attention',reuse=reuse):
            self.att_c = self.__build_attention(self.lstm_out)

        #
        # build fully connected
        #
        with tf.variable_scope('fully_connected',reuse=reuse):
            self.scores = self.__build_fully_connected(self.att_c )

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
        #
        # build session
        #
        if not build_session: return

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # True

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