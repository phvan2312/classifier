import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
from utils import pad_common
from sklearn.metrics import f1_score, precision_score, recall_score

class Base_Classifier:
    def __init__(self, **kargs):
        if kargs.get('reset_graph', True):
            tf.reset_default_graph()

        # must have
        self.sess = None
        self.train_op = None
        self.loss = None
        self.accuracy = None
        self.logits = None
        self.target = None
        self.saver = None
        self.lr_placeholder = None

        self.id2lexicon, self.lexicon_emb_dim = kargs.get('id2lexicon', None), 20
        self.id2pos, self.pos_emb_dim = kargs.get('id2pos', None), 30
        self.id2vaderlexicon, self.id2vaderscore, self.vader_lexicon_dim = kargs.get('id2vaderlexicon', None), \
                                                                           kargs.get('id2vaderscore', None), 30
    #
    # build placeholder for additional features
    #
    def build_additional_placeholder(self):
        if self.id2lexicon is not None and hasattr(self, 'max_sen_len'):
            self.lexicon_ids = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sen_len),
                                              name='lexicon_ids')  # (batch_size, max_sen_len)
            self.LEXICON_EMB = tf.get_variable(name='lexicon_emb', shape=(len(self.id2lexicon), self.lexicon_emb_dim),
                                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        if self.id2pos is not None and hasattr(self, 'max_sen_len'):
            self.pos_ids = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sen_len),
                                          name='pos_ids')  # (batch_size, max_sen_len)
            self.POS_EMB = tf.get_variable(name='pos_emb',
                                           shape=(len(self.id2pos), self.pos_emb_dim),
                                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        if self.id2vaderlexicon is not None and hasattr(self, 'max_sen_len'):
            self.vader_lexicon_ids = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sen_len),
                                                    name='vader_lexicon_ids')  # (batch_size, max_sen_len)
            self.VADER_LEXICON_EMB = tf.get_variable(name='vader_lexicon_emb',
                                                     shape=(len(self.id2vaderlexicon), self.vader_lexicon_dim),
                                                     dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer())
            self.vader_lexicon_target = tf.placeholder(dtype=tf.float32, shape=(None, self.max_sen_len),
                                                       name='vader_lexicon_score')
            self.VADER_LEXICON_W = tf.get_variable(name='vader_lexion_W', shape=(self.vader_lexicon_dim,1),
                                                   dtype=tf.float32,
                                                   initializer=tf.contrib.layers.xavier_initializer())

    #
    # get embedding vector for additional features
    #
    def encode_additional_word(self):
        add_features = []
        add_total_dim = 0

        if self.id2lexicon is not None:
            self.lexicon_vect = tf.nn.embedding_lookup(self.LEXICON_EMB, self.lexicon_ids)
            add_features.append(self.lexicon_vect)
            add_total_dim += self.lexicon_emb_dim

        if self.id2pos is not None:
            self.pos_vect = tf.nn.embedding_lookup(self.POS_EMB, self.pos_ids)
            add_features.append(self.pos_vect)
            add_total_dim += self.pos_emb_dim

        if self.id2vaderlexicon is not None:
            self.vader_lexicon_vect = tf.nn.embedding_lookup(self.VADER_LEXICON_EMB, self.vader_lexicon_ids)
            add_features.append(self.vader_lexicon_vect)
            add_total_dim += self.vader_lexicon_dim

        return add_features, add_total_dim

    #
    # create tensor input for additional features
    #
    def create_additional_input(self, batch):
        if hasattr(self, 'max_sen_len'):
            max_sentence_length = self.max_sen_len

            result = {}
            if self.id2lexicon is not None:
                ip_lexicon_ids, _ = pad_common(sequences=[e['lexicon_ids'] for e in batch], pad_tok=1,
                                               max_length=max_sentence_length)
                result[self.lexicon_ids] = ip_lexicon_ids
            if self.id2pos is not None:
                ip_pos_ids, _ = pad_common(sequences=[e['pos_ids'] for e in batch], pad_tok=1,
                                           max_length=max_sentence_length)
                result[self.pos_ids] = ip_pos_ids
            if self.id2vaderlexicon is not None:
                ip_vader_lexicon_ids, _ = pad_common(sequences=[e['vader_lexicon_ids'] for e in batch], pad_tok=0,
                                               max_length=max_sentence_length)
                ip_vader_lexicon_target = [[self.id2vaderscore[_id] for _id in id] for id in ip_vader_lexicon_ids]

                result[self.vader_lexicon_ids] = ip_vader_lexicon_ids
                result[self.vader_lexicon_target] = ip_vader_lexicon_target

            return result
        else:
            raise Exception('Unknown max_sen_len')

    # left for children
    def create_input(self, batch):
        pass

    def create_additional_loss(self):
        add_loss = 0.0

        if self.id2vaderscore is not None and hasattr(self, 'max_sen_len'):
            tmp = tf.reshape(self.vader_lexicon_vect,shape=(-1,self.vader_lexicon_dim))
            tmp_dot = tf.matmul(tmp,self.VADER_LEXICON_W)

            predictions = tf.reshape(tmp_dot,shape=(-1,self.max_sen_len))

            add_loss += tf.losses.mean_squared_error(labels=self.vader_lexicon_target,
                                                     predictions=predictions)

        return add_loss

    def batch_run(self, batch, i, lr=0.005, mode='train', metric='precision'):

        ip_feed_dict = self.create_input(batch)

        ip_feed_dict[self.lr_placeholder] = lr

        for k, v in self.create_additional_input(batch).items():
            ip_feed_dict[k] = v

        if mode == 'train':
            if hasattr(self, 'dropout') and hasattr(self, 'dropout_keep_prob'):
                ip_feed_dict[self.dropout] = self.dropout_keep_prob

            if hasattr(self, 'is_training'):
                ip_feed_dict[self.is_training] = True


            _, loss, logit = self.sess.run([self.train_op, self.loss, self.logits], feed_dict=ip_feed_dict)

        elif mode == 'eval':
            if hasattr(self, 'dropout'):
                ip_feed_dict[self.dropout] = 1.0

            if hasattr(self, 'is_training'):
                ip_feed_dict[self.is_training] = False

            loss, logit = self.sess.run([self.loss, self.logits], feed_dict=ip_feed_dict)

        y_pred = np.argmax(logit, axis=1)
        y_true = np.argmax(ip_feed_dict[self.target], axis=1)

        if metric == 'precision':
            acc = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        elif metric == 'recall':
            acc = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        elif metric == 'f1':
            acc = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

        return loss, acc

    def save(self, save_path):
        save_path = self.saver.save(self.sess, save_path)
        #print "Model saved in file: %s" % save_path

    def restore(self, path):
        self.saver.restore(self.sess, path)
