import sys
sys.path.append('..')

import tensorflow as tf
import cPickle
from vars import *
from nn_base import Base_Classifier

class Multi_Classifier(Base_Classifier):
    def __init__(self, model_paths, id2word, id2label, max_sen_len=70,lr=0.001, **kargs):

        Base_Classifier.__init__(self, **kargs)

        # non use additional features
        if hasattr(self,'id2pos'):
            self.id2pos = None
        if hasattr(self,'id2lexicon'):
            self.id2lexicon = None
        ###

        self.model_paths = model_paths
        self.id2word = id2word
        self.id2label = id2label
        self.max_sen_len = max_sen_len
        self.lr = lr
        self.n_tags = len(self.id2label)

    def __build_single_model(self, model_paths):
        models = {}

        for model_path in model_paths:
            params_path = model_path + '/vocab/params.pkl'
            save_model_path = model_path + '/model.ckpt'

            with open(params_path,'r') as f:
                params = cPickle.load(f)

            model_type = params['model_type']

            with tf.Graph().as_default():
                params['use_pretrained'] = None
                params['reset_graph']    = False

                cur_model = get_corressponding_model(model_type)(**params)
                cur_model.build_model()

                cur_model.saver.restore(cur_model.sess, save_model_path)

                tmp_name = model_type
                count = 0
                while tmp_name in models:
                    count += 1
                    tmp_name = model_type + '_version_' +  str(count)

                model_type = tmp_name

                models[model_type] = {
                    'tf_model' : cur_model,
                    'params'   : params
                }

        self.target = tf.placeholder(dtype=tf.float32, shape=(None, self.n_tags), name='labels')  # (batch_size, n_tags)
        for k,v in models.items():
            v['tf_logits'] = tf.placeholder(tf.float32,shape=[None, self.n_tags], name='logits_%s' % k),


        self.build_additional_placeholder()
        return models

    def build_model(self):

        # build single models
        with tf.variable_scope('build_single_model'):
            self.models = self.__build_single_model(self.model_paths)

        # ensemble logits from each models
        self.logits_esembs = []
        with tf.variable_scope('ensemble'):
            for k,v in self.models.items():
                logits = tf.reshape(v['tf_logits'],shape=(-1,len(self.id2label)))

                W_esemb = tf.get_variable(name='W_esemb_%s' % k, shape=(self.n_tags,self.n_tags),
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

                b_esemb = tf.Variable(tf.constant(0.1, shape=[self.n_tags]), name='b_esemb_%s' % k)


                W_esemb_1 = tf.get_variable(name='W_esemb_%s_1' % k, shape=(self.n_tags, self.n_tags),
                                          dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

                b_esemb_1 = tf.Variable(tf.constant(0.1, shape=[self.n_tags]), name='b_esemb_%s_1' % k)

                logits_esemb = tf.nn.xw_plus_b(logits,W_esemb,b_esemb,name='mul_esemb_%s' % k)
                logits_esemb = tf.nn.xw_plus_b(logits_esemb,W_esemb_1,b_esemb_1,name='mul_esemb_%s_1' % k)

                self.logits_esembs.append(tf.expand_dims(logits_esemb,axis=1))

            fn_logits_esembs = tf.concat(self.logits_esembs,axis=1)
            self.logits = tf.reduce_sum(fn_logits_esembs,axis=1) # (batch x num_tags)

        # build loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))

            predictions = tf.argmax(self.logits, 1, name="predictions")
            correct_predictions = tf.equal(predictions, tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # build optimizer
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss)

        #
        # build session
        #
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

    def create_input(self,batch):
        ip = {}

        for k, v in self.models.items():
            cur_model = v['tf_model']

            cur_input = cur_model.create_input(batch)

            for _k,_v in cur_model.create_additional_input(batch).items():
                cur_input[_k] = _v

            if hasattr(cur_model,'dropout'):
                cur_input[cur_model.dropout] = 1.0

            if hasattr(cur_model,'is_training'):
                cur_input[cur_model.is_training] = False

            logits = cur_model.sess.run([cur_model.logits], feed_dict=cur_input)[-1]

            ip[self.models[k]['tf_logits'][-1]] = logits

        ip[self.target] = [e['label_ids'] for e in batch]

        return ip

if __name__ == '__main__':

    pass