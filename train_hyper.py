#!/usr/bin/python
import cPickle
import sys

import numpy as np
import tensorflow as tf
from train import build_config,load_all_data,train
from vars import *

import os

reload(sys)
sys.setdefaultencoding('utf-8')

import argparse as _argparse
tf.app.flags._global_parser = _argparse.ArgumentParser() # for reset flags
flags = tf.app.flags

"""
Define some global arguments
"""
flags.DEFINE_string('saved_path','./saved_model_hyper/sentiment_standard_standford_full_fine_test','path for storing model and all of its corresponding components.')
flags.DEFINE_string('train_path','./data/sentiment_standard_standford/fine/merge/all.txt','path of training dataset.')
flags.DEFINE_string('test_path', './data/sentiment_standard_standford/fine/merge/test.txt' ,'path of testing dataset')
flags.DEFINE_string('word2vec', None, 'path of word2vec') # './data/word2vec/fasttext_pretrained_200d.vec'
flags.DEFINE_string('model_type', ATTENTION_MODEL_TYPE,'model type: [attention, cnn or anything else]')

flags.DEFINE_boolean('convert_slang',True,'convert slang words or not')
flags.DEFINE_boolean('use_sentiment_lexicon',False,'use sentiment lexicon as additional features or not ?')
flags.DEFINE_boolean('use_pos',False,'use pos (part-of-speech) as additional features')
flags.DEFINE_boolean('vader_lexicon',False,'use vader leixicon as additional features')

flags.DEFINE_string('metric','f1', 'type of metric, [precision, recall, f1]')
flags.DEFINE_string('reload_model','./data/sentiment_standard_standford/fine/merge/train_test_dataset_using_nltk.pkl',
                    'path to save dataset, id2x (vocab)')
###################################

from shutil import rmtree # for remove folder
from distutils.dir_util import copy_tree # for copy

from sklearn.base import BaseEstimator, ClassifierMixin

class HyperClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""
    best_test_for_all_time = - np.inf

    def __init__(self,params, train_dataset, test_dataset, config):
        """
        Called when initializing the classifier
        """
        self.params = params # default parameters, get from ./params
        self.base_model = config['model']
        self.saved_path = config['save_model_path']

        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset

        self.hyper_note_path = config['save_hyper_note']
        self.save_params_path = config['save_params_path']

        self.config = config

    def save(self, src, dest):
        if os.path.isfile(dest):
            rmtree(dest)

        copy_tree(src,dest)

    def fit(self, X, y = None):
        try:
            self.cur_score, saved_path = train(self.config,self.params,self.train_dataset,self.test_dataset,False)

            #
            # Choose to save best model (based on its score)
            #
            if self.cur_score > HyperClassifier.best_test_for_all_time:
                HyperClassifier.best_test_for_all_time = self.cur_score
                self.save(saved_path,self.config['best_candidate'])

            #
            # save log to hyper_note
            #
            tmp_params = self.params.copy()
            del tmp_params['id2word']
            del tmp_params['id2label']
            del tmp_params['id2lexicon']
            del tmp_params['id2pos']

            print ('cur_parameter: ', tmp_params)
            print ('best_test_for_cur_time: ', self.cur_score)
            print ('best_test_for_all_time: ', self.best_test_for_all_time)

            with open(self.hyper_note_path,'a') as f:
                f.write('\n##########################\n')
                f.write('\n# params: %s\n' % str(tmp_params))
                f.write('\n# best_test_for_cur_time: %s\n' % str(self.cur_score))
                f.write('\n# best_test_for_all_time: %s\n' % str(self.best_test_for_all_time))
                f.write('\n##########################\n')
        except:
            self.cur_score = 0

        return self

    # not need if we specified scoring accuracy
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return self.cur_score

def main(_):

    ####################### ######################### ########################## #########################
    #
    # build config
    #
    print ('-- build config')

    flags = tf.app.flags.FLAGS

    saved_path = flags.saved_path
    flags.saved_path = flags.saved_path + '/candidate'

    config = build_config(FLAGS = flags)
    config['best_candidate'] = saved_path + '/best_candidate'
    config['save_hyper_note'] = saved_path + '/hyper_note.txt'
    config['hypers_path'] = get_corressponding_hypers(config['model_type'])

    out = load_all_data(config)

    id2word, id2label, id2lexicon, id2pos, id2vaderlexicon, id2vaderscore = out['word']['id2word'], out['label']['id2label'],out['lexicon']['id2lexicon'], out['pos']['id2pos'], out['vader_lexicon']['id2vaderlexicon'],out['vader_lexicon']['id2vaderscore']

    train_dataset, test_dataset = out['train_dataset'], out['test_dataset']
    train_dataset, test_dataset = train_dataset[:15000], test_dataset[:15000] # for test only

    ####################### ######################### ########################## #########################

    #
    # loading parameter for each model.
    #
    with open(config['params_path'],'r') as f:
        import json
        params = json.load(f)
        print ('parameters: ', params)

    params['id2word'] = id2word
    params['id2label'] = id2label

    params['id2lexicon'] = id2lexicon
    params['id2pos'] = id2pos
    params['id2vaderlexicon'] = id2vaderlexicon
    params['id2vaderscore'] = id2vaderscore

    params['use_pretrained'] = config['word2vec']
    params['model_type'] = config['model_type']
    params['sentiment_lexicon'] = config['sentiment_lexicon']
    params['pos'] = config['pos']
    params['convert_slang'] = config['convert_slang']

    params['model_name'] = config['model_name']

    #
    # build hyper_parameter classifier
    #
    with open(config['save_hyper_note'], 'w') as f:
        f.write('')

    hyper_clf = HyperClassifier(params=params, train_dataset=train_dataset, test_dataset=test_dataset, config=config)

    #
    # prepare turned parameters
    #
    all_params = []
    with open(config['hypers_path']) as f:
        hypers = json.load(f)

        for k, vs in hypers.items():
            assert type(vs) is list
            assert k in params

            tmp_all_params = [params] if len(all_params) == 0 else list(all_params)
            all_params = []

            for tmp_all_param in tmp_all_params:
                for v in vs:
                    tmp_params = tmp_all_param.copy()
                    tmp_params[k] = v

                    all_params.append(tmp_params)

        print ('hyper parameters: ', hypers)

    tuned_params = {"params": all_params}

    #
    # run
    #
    from sklearn.model_selection import RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=hyper_clf, param_distributions=tuned_params, n_jobs=1, verbose=1, n_iter=30,
                            cv=[(range(0, 200), range(200, 300))], return_train_score=False, refit=False)

    # range(100,200) and range(200,300) is faked data.
    rs.fit(range(0, 300), range(0, 300))

if __name__ == '__main__':
    tf.app.run()
    pass