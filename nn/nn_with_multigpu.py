import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from vars import *

from math import ceil

class Multi_Classifier:

    def __init__(self,base_model_type, n_gpus,**kargs):
        self.base_model_type = get_corressponding_model(base_model_type)
        self.n_gpus = n_gpus
        self.kargs = kargs
        self.kargs['reset_graph'] = False

    def __average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, v in grad_and_vars:
                norm_g = tf.zeros_like(v) if g == None else g
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(norm_g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, axis=0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def build_model(self):
        self.model_gpu = {
            'model' : [],
            'loss'  : [],
            'logit' : [],
            'grad'  : []
        }

        reuse = False
        for i in range(self.n_gpus):
            with tf.device('/device:GPU:%d' % i):
                #
                # get single model
                #
                model = self.base_model_type(**self.kargs)
                model.build_model(build_session=False,reuse=reuse)
                self.model_gpu['model'].append(model)

                reuse = True

                #
                # get loss, logit & optimizer
                #
                loss, logit = model.loss, model.logits
                opt = model.optimizer
                self.model_gpu['logit'].append(logit)
                self.model_gpu['loss'].append(loss)

                #
                # compute gradients
                #
                grads = opt.compute_gradients(loss)
                self.model_gpu['grad'].append(grads)


        #
        # average gradients
        #
        grads = self.__average_gradients(self.model_gpu['grad'])

        #
        # apply the gradients to adjust the shared variables.
        #
        self.train_op = opt.apply_gradients(grads)
        self.loss = tf.reduce_mean(self.model_gpu['loss'])
        self.logits = tf.concat(self.model_gpu['logit'],axis=0)


        #
        # build session
        #
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        config.allow_soft_placement = True

        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        init_feed_dict = {}

        for model in self.model_gpu['model']:
            init_feed_dict[model.WORD_EMB_placeholder] = model.NP_WORD_EMB

        self.sess.run(init_op, feed_dict=init_feed_dict)

    def __split_batch(self, batch, n_gpus):
        batch_len = len(batch)
        batches = []
        gap = int(ceil(float(batch_len) / n_gpus ))

        if gap < 1:
            batches = [batch] * n_gpus
            raise Exception ("Not enough samples, just %d samples for %d GPU" % (batch_len,n_gpus))
        else:
            for i in range(n_gpus):
                start_id = i * gap
                next_id  = (i+1) * gap if (i+1)*gap < batch_len else batch_len

                batches.append(batch[start_id:next_id])

        return batches

    def create_input(self, batch):
        split_batches = self.__split_batch(batch,self.n_gpus)

        ip_feed_dict = {}
        for split_batch, model in zip(split_batches,self.model_gpu['model']):
            _ip_feed_dict = model.create_input(split_batch)

            for k,v in _ip_feed_dict.items():
                ip_feed_dict[k] = v

            for k, v in model.create_additional_input(split_batch).items():
                ip_feed_dict[k] = v

        return ip_feed_dict

    def batch_run(self, batch, i, mode='train', metric='precision'):
        ip_feed_dict = self.create_input(batch)

        for model in self.model_gpu['model']:
            if mode == 'train':
                if hasattr(model, 'dropout') and hasattr(model, 'dropout_keep_prob'):
                    ip_feed_dict[model.dropout] = model.dropout_keep_prob

                if hasattr(model, 'is_training'):
                    ip_feed_dict[model.is_training] = True
            elif mode == 'eval':
                if hasattr(model, 'dropout'):
                    ip_feed_dict[model.dropout] = 1.0

                if hasattr(model, 'is_training'):
                    ip_feed_dict[model.is_training] = False

        if mode == 'train':
            _, loss, logit = self.sess.run([self.train_op, self.loss, self.logits], feed_dict=ip_feed_dict)
        elif mode == 'eval':
            loss, logit = self.sess.run([self.loss, self.logits], feed_dict=ip_feed_dict)

        y_pred = np.argmax(logit, axis=1)
        y_true = np.argmax([e['label_ids'] for e in batch], axis=1)

        if metric == 'precision':
            acc = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        elif metric == 'recall':
            acc = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        elif metric == 'f1':
            acc = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

        return loss, acc

    def save(self, save_path):
        save_path = self.saver.save(self.sess, save_path)
        print "Model saved in file: %s" % save_path