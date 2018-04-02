#!/usr/bin/python
import cPickle
import sys

import numpy as np
import tensorflow as tf
from utils import load_train_file, word_mapping, common_mapping, create_batch, create_dataset, add_to_dataset, \
    word_tokenize, pos_tagging
import os
from vars import *

import pandas as pd

reload(sys)
sys.setdefaultencoding('utf-8')

"""
Count time execution
"""
import time
time_log = {}
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            time_log[method.__name__] = '%2.2f ms' % ((te - ts) * 1000)
            print '%r : %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result

    return timed

"""
Define some global arguments
"""
flags = tf.app.flags
flags.DEFINE_string('saved_path','./saved_model/sentiment_standard_standford_full_fine_attention_test','path for storing model and all of its corresponding components.')
flags.DEFINE_string('train_path','./data/sentiment_standard_standford/fine/merge/all.txt','path of training dataset.')
flags.DEFINE_string('test_path', './data/sentiment_standard_standford/fine/merge/test.txt' ,'path of testing dataset')
flags.DEFINE_string('word2vec', None, 'path of word2vec')
flags.DEFINE_string('model_type', ATTENTION_MODEL_TYPE,'model type: [attention, cnn or anything else]')

flags.DEFINE_boolean('convert_slang',True,'convert slang words or not')
flags.DEFINE_boolean('use_sentiment_lexicon',False,'use sentiment lexicon as additional features or not ?')
flags.DEFINE_boolean('use_pos',False,'use pos (part-of-speech) as additional features')
flags.DEFINE_boolean('vader_lexicon',False,'use vader leixicon as additional features')

flags.DEFINE_string('metric','f1', 'type of metric, [precision, recall, f1]')
flags.DEFINE_string('reload_model','./data/sentiment_standard_standford/fine/merge/train_test_dataset_using_nltk_with_emoji.pkl',
                    'path to save dataset, id2x (vocab)')

"""
Work flow: 
Step 1: we create a config parameter for storing all of user-defined arguments.
Step 2: check if <reload_model> path file is already existed ? If not, we create one from scratch by calling 
<load_all_data_from_scratch> method, otherwise, we just load from its saved path <reload_model>.
Step 3: call <train> method to create tensorflow model, training, ...  
"""

@timeit
def build_config(FLAGS):
    config = {}

    config['train_path'] = FLAGS.train_path
    config['test_path']  = FLAGS.test_path
    config['saved_path'] = FLAGS.saved_path
    config['word2vec']   = FLAGS.word2vec
    config['model_type'] = FLAGS.model_type

    config['convert_slang'] = FLAGS.convert_slang
    is_ensemble = config['model_type'] in [ENSEMBLE_MODEL_TYPE]

    config['sentiment_lexicon'] = True if is_ensemble else FLAGS.use_sentiment_lexicon
    config['pos'] = True if is_ensemble else FLAGS.use_pos
    config['vader_lexicon'] = True if is_ensemble else FLAGS.vader_lexicon

    config['metric'] = FLAGS.metric
    config['model_name'] = FLAGS.saved_path.split('/')[-1]

    config['reload_model'] = FLAGS.reload_model

    # validate
    for path in [config['train_path'],config['test_path']]:
        if not os.path.isfile(path): raise Exception('%s is not a file .' % path)

    if not os.path.exists(config['saved_path']):
        os.makedirs(config['saved_path'])
        os.makedirs(config['saved_path'] + '/vocab')

    if config['word2vec'] is not None:
        if not os.path.isfile(config['word2vec']): raise Exception('%s is not a file .' % config['word2vec'])

    # save path
    save_path = config['saved_path'] + '/'

    config['save_model_path'] = save_path + "model.ckpt"
    config['save_word_path'] = save_path + "vocab/word.pkl"
    config['save_label_path'] = save_path + "vocab/label.pkl"
    config['save_lexicon_path'] = save_path + "vocab/lexicon.pkl"
    config['save_pos_path'] = save_path + "vocab/pos.pkl"
    config['save_params_path'] = save_path + "vocab/params.pkl"
    config['save_vader_lexicon_path']  = save_path + "vocab/vader_lexicon.pkl"

    config['params_path'] = get_corressponding_params(config['model_type'])

    config['model'] = get_corressponding_model(config['model_type'])
    config['save_summary_path'] = save_path + "summary"

    ### Print ###
    print ('Training path: ' , config['train_path'])
    print ('Testing path: ', config['test_path'])
    print ('Saved path: ', config['saved_path'])
    print ('Word2vec: ', config['word2vec'])
    print ('Model type: ', config['model_type'])
    print ('Use Lexicon: ', config['sentiment_lexicon'])
    print ('Use POS, (Part Of Speech): ', config['pos'])
    print ('User vader_lexicon: ', config['vader_lexicon'])

    return config

def load_dict(path):
    with open(path, 'r') as f:
        return cPickle.load(f)

@timeit
def convert_slang(slang_dict, tokenized_texts):
    norm_words = [[slang_dict[word] if word in slang_dict else word for word in tokenized_text]
                  for tokenized_text in tokenized_texts]

    return norm_words

@timeit
def convert_sentiment_lexicon(lexicon_dict, tokenized_texts, replace_word):
    norm_lexicons = [[lexicon_dict[word] if word in lexicon_dict else replace_word for word in tokenized_text]
                     for tokenized_text in tokenized_texts]

    return norm_lexicons

@timeit
def convert_pos(tokenized_texts):
    norm_pos = [pos_tagging(tokenized_text,is_tokenizered=True) for tokenized_text in tokenized_texts]

    return norm_pos

@timeit
def convert_vader_lexicon(lexicon2id,tokenized_texts):
    return [[word if word in lexicon2id else '<unk>' for word in tokenized_text] for tokenized_text in tokenized_texts]

def load_all_data_from_scratch(config):
    #
    # load & normalize data
    #
    @timeit
    def load_raw_data():
        print ('-- load data')
        raw_data, raw_labels = load_train_file(config['train_path'])
        test_data, test_labels = load_train_file(config['test_path'])

        #raw_data, raw_labels = raw_data[:100], raw_labels[:100] # for test only

        return raw_data, raw_labels, test_data, test_labels

    raw_data, raw_labels, test_data, test_labels = load_raw_data()
    len_train = len(raw_data)

    #
    # tokenized sentence into words
    #
    @timeit
    def tokenize_word():
        print ('-- tokenize words')
        words = [word_tokenize(data) for data in raw_data + test_data]
        labels = [label for label in raw_labels + test_labels]

        return words, labels

    words, labels = tokenize_word()

    #
    # convert slang into its corresponding word (maybe not?)
    #
    if config['convert_slang']:
        slang_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/slang/slang.pkl'
        slang_dict = load_dict(slang_path)

        words = convert_slang(slang_dict,words)

    #
    # split train and test.
    #
    print ('-- split train and test data')
    train_words, train_labels, test_words, test_labels = words[:len_train], labels[:len_train], \
                                                         words[len_train:], labels[len_train:]

    #
    # build vocabulary
    #
    print ('-- building vocabulary')
    _, id2word, word2id = word_mapping(train_words, pre_emb=config['word2vec'])
    _, id2label, label2id = common_mapping([[e] for e in train_labels], name='label')

    saved_word = {'id2word': id2word, 'word2id': word2id,}
    saved_label = {'id2label': id2label, 'label2id': label2id}

    cPickle.dump(saved_word, open(config['save_word_path'], 'w'))
    cPickle.dump(saved_label, open(config['save_label_path'], 'w'))

    #
    # create dataset
    #
    train_dataset = create_dataset(train_words, train_labels, word2id, label2id)
    test_dataset = create_dataset(test_words, test_labels, word2id, label2id)

    #
    # add sentiment_lexicon as additional features (maybe not?)
    #
    id2lexicon, lexicon2id = None, None
    if config['sentiment_lexicon']:
        lexicon_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/lexicon/lexicon.pkl'
        lexicon_dct = load_dict(lexicon_path)

        lexicons = convert_sentiment_lexicon(lexicon_dct,words,'neu')
        train_lexicons, test_lexicons = lexicons[:len_train], lexicons[len_train:]
        _, id2lexicon, lexicon2id = word_mapping(train_lexicons)

        train_dataset = add_to_dataset(train_dataset, train_lexicons, lexicon2id, 'lexicon_ids')
        test_dataset  = add_to_dataset(test_dataset, test_lexicons, lexicon2id, 'lexicon_ids')

    saved_lexicon = {'id2lexicon':id2lexicon, 'lexicon2id':lexicon2id}
    cPickle.dump(saved_lexicon, open(config['save_lexicon_path'], 'w'))

    #
    # add pos (part-of-speech) as additional features (maybe not?)
    #
    id2pos, pos2id = None, None
    if config['pos']:
        pos = convert_pos(words)
        train_pos, test_pos = pos[:len_train], pos[len_train:]
        _, id2pos, pos2id = word_mapping(train_pos)

        train_dataset = add_to_dataset(train_dataset, train_pos, pos2id, 'pos_ids')
        test_dataset = add_to_dataset(test_dataset, test_pos, pos2id, 'pos_ids')

    saved_pos = {'id2pos': id2pos, 'pos2id': pos2id}
    cPickle.dump(saved_pos, open(config['save_pos_path'], 'w'))

    #
    # add vader lexicon as additional features (maybe not?)
    #
    id2vaderscore, vaderlexicon2id,id2vaderscore = None, None, None
    if config['vader_lexicon']:
        lexicon_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/vaderSentiment/vader_lexicon.txt'
        lexicon_df = pd.read_csv(lexicon_path, sep='\t', encoding='utf-8', names=['token', 'pos_score', 'other1',
                                                                                  'other2'])[['token', 'pos_score']]

        id2vaderlexicon = {k: v for k, v in enumerate(['<unk>'] + lexicon_df['token'].tolist())}
        vaderlexicon2id = {v: k for k, v in id2vaderlexicon.items()}

        vaderlexicon2score = {k: v for k, v in zip(lexicon_df['token'].tolist(), lexicon_df['pos_score'].tolist())}
        id2vaderscore = {k: vaderlexicon2score[v] if v in vaderlexicon2score else 0.0 for k, v in id2vaderlexicon.items()}

        vaderlexicons = convert_vader_lexicon(vaderlexicon2id,words)
        train_vaderlexicons, test_vaderlexicons = vaderlexicons[:len_train], vaderlexicons[len_train:]

        train_dataset = add_to_dataset(train_dataset, train_vaderlexicons, vaderlexicon2id, 'vader_lexicon_ids')
        test_dataset = add_to_dataset(test_dataset, test_vaderlexicons, vaderlexicon2id, 'vader_lexicon_ids')

    saved_vader_lexicon = {
        'id2vaderlexicon': id2vaderscore,
        'vaderlexicon2id': vaderlexicon2id,
        'id2vaderscore': id2vaderscore
    }
    cPickle.dump(saved_vader_lexicon, open(config['save_vader_lexicon_path'], 'w'))

    saved_to_dataset = {
        'word': saved_word,
        'label': saved_label,
        'lexicon': saved_lexicon,
        'vader_lexicon': saved_vader_lexicon,
        'pos': saved_pos,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }

    return saved_to_dataset

def load_all_data(config):
    load_dataset_path = config.get('reload_model')

    if os.path.isfile(load_dataset_path):
        with open(load_dataset_path, 'r') as f:
            out = cPickle.load(f)

        # just for sure
        if config['sentiment_lexicon'] == False: out['lexicon'] = {'id2lexicon':None,'lexicon2id':None}
        if config['pos'] == False: out['pos'] = {'id2pos':None,'pos2id':None}
        if config['vader_lexicon'] == False: out['vader_lexicon'] =  {'id2vaderlexicon': None, 'vaderlexicon2id': None,'id2vaderscore':None}

        # save to file
        cPickle.dump(out['word'], open(config['save_word_path'], 'w'))
        cPickle.dump(out['label'], open(config['save_label_path'], 'w'))
        cPickle.dump(out['lexicon'], open(config['save_lexicon_path'], 'w'))
        cPickle.dump(out['pos'], open(config['save_pos_path'], 'w'))

    else:
        out = load_all_data_from_scratch(config)

        cPickle.dump(out,open(load_dataset_path,'w'))

    return out

from progress.bar import ChargingBar as Bar
from prettytable import PrettyTable

@timeit
def train(config, params, train_dataset, test_dataset, add_training_log):

    #
    # build batches
    #
    train_batch, test_batch = create_batch(train_dataset, params['batch_size']), create_batch(test_dataset, params['batch_size'])

    #
    # save params
    #
    cPickle.dump(params, open(config['save_params_path'], 'w'))

    #
    # build tensorflow model
    #
    @timeit
    def build_tf_model():
        print ('-- build model')
        model = config['model'](**params)
        model.build_model()

        # build custom tensorboard
        print ('-- build tensorboard')
        # tensorboard = CustomTensorboard(dir_summary=config['save_summary_path'], model_graph=model.sess.graph,
        #                                 metric=config['metric'])
        tensorboard = None

        return model, tensorboard

    model, tensorboard = build_tf_model()

    #
    # training parameters
    #
    print ('-- start training')
    train_count = 0
    eval_count = 0
    best_test = 0.0
    decay_lr_every = 500
    lr_decay = 0.9
    init_lr = model.lr

    #
    # training
    #
    for epoch in range(params['nepochs']):
        print ("\n[E]poch %i, lr %.4f" % (epoch + 1, init_lr))
        bar = Bar('training', max=len(train_batch), suffix='')
        tabl = PrettyTable( ['status', 'train_epoch', 'train_batch', 'test_avg_score', 'test_avg_loss'])

        loss_total, acc_total = [], []

        for i, batch_id in enumerate(np.random.permutation(len(train_batch))):
            train_count += 1
            batch = train_batch[i]

            loss, acc = model.batch_run(batch=batch, i=train_count, mode='train', metric=config['metric'])

            loss_total.append(loss)
            acc_total.append(acc)

            bar.bar_prefix = " | batch %i |" % (batch_id + 1)
            bar.bar_suffix = " | cur_loss: %.3f | cur_acc: %.3f | best_test_acc: %.3f" % (loss,acc,best_test)

            bar.next()

            if train_count % decay_lr_every == 0: init_lr = init_lr * lr_decay

            if train_count % params['freq_eval'] == 0:
                test_loss_lst, test_acc_lst = [], []

                for t_i, t_batch_id in enumerate(range(len(test_batch))):
                    t_batch = test_batch[t_batch_id]
                    eval_count += 1

                    loss, acc = model.batch_run(batch=t_batch, i=eval_count, mode='eval', metric=config['metric'],lr=init_lr)

                    test_loss_lst.append(loss)
                    test_acc_lst.append(acc)

                test_loss = np.array(test_loss_lst).mean()
                test_acc = np.array(test_acc_lst).mean()

                status = '++++'
                if test_acc > best_test:
                    best_test = test_acc
                    status = 'best'

                    model.save(config['save_model_path'])

                tabl.add_row([status,epoch + 1,batch_id+1,'%.3f'%test_acc,'%.3f'%test_loss])

        print "\nmean_train_loss: %.3f, mean_train_acc: %.3f" % (np.mean(loss_total),np.mean(acc_total))
        print (tabl.get_string(title="Local All Test Accuracies"))

    return best_test, config['saved_path']

from shutil import rmtree # for remove folder
from distutils.dir_util import copy_tree # for copy

def copy(src_folder_path, dest_folder_path):
    if os.path.isfile(dest_folder_path):
        rmtree(dest_folder_path)

    copy_tree(src_folder_path, dest_folder_path)

def main(_):

    ####################### ######################### ########################## #########################
    #
    # build config
    #
    print ('-- build config')
    config = build_config(FLAGS = tf.app.flags.FLAGS)

    out = load_all_data(config)

    #
    # loading dataset, vocab
    #
    id2word, id2label, id2lexicon, id2pos, id2vaderlexicon, id2vaderscore = out['word']['id2word'], out['label']['id2label'], \
                                            out['lexicon']['id2lexicon'], out['pos']['id2pos'], out['vader_lexicon']['id2vaderlexicon'], out['vader_lexicon']['id2vaderscore']
    train_dataset, test_dataset = out['train_dataset'], out['test_dataset']

    #
    # loading parameter for each model.
    #
    with open(config['params_path'], 'r') as f:
        import json
        params = json.load(f)
        print ('parameters: ', params)

    #
    # processing for multi gpu, it doesn't have explicit model type (LSTM, Attention,...). All of its defined model we called
    # its base model, and we wanna use gpu to speed up training. So we need to process for its base model parametes instead.
    #

    if config['model_type'] == MULTIGPU_MODEL_TYPE:
        base_params_path = get_corressponding_params(params['base_model_type'])
        with open(base_params_path,'r') as f:
            base_params = json.load(f)
            print ('base parameters: ', base_params)

        for k,v in base_params.items():
            params[k] = v

    params['id2word'] = id2word
    params['id2label'] = id2label

    params['id2lexicon'] = id2lexicon
    params['id2pos'] = id2pos
    params['id2vaderlexicon'] = id2vaderlexicon
    params['id2vaderscore'] = id2vaderscore

    params['use_pretrained'] = config['word2vec']
    params['model_type'] = config['model_type'] if config['model_type'] != MULTIGPU_MODEL_TYPE else params['base_model_type']
    params['sentiment_lexicon'] = config['sentiment_lexicon']
    params['pos'] = config['pos']
    params['convert_slang'] = config['convert_slang']

    params['model_name'] = config['model_name']

    #
    # processing for ensemble model. We need to copy all of its dependent child model into same folder. So it will be
    # easy for further process.
    #
    if params['model_type'] in [ENSEMBLE_MODEL_TYPE]:
        assert 'model_paths' in params
        new_model_paths = []

        for model_path in params['model_paths']:
            new_model_path = model_path.replace('./', config['saved_path'] + '/')
            new_model_paths.append(new_model_path)

            copy(model_path,new_model_path)

    #
    # training model
    #
    train(config,params,train_dataset,test_dataset,True)

if __name__ == '__main__':
    tf.app.run()
    pass