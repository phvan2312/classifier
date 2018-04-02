#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
from gensim.models.keyedvectors import KeyedVectors

from sklearn.model_selection import train_test_split
from gensim.scripts.glove2word2vec import glove2word2vec

import pandas as pd
import emoji

use_binary = True
pretrain_word2vec = None


def clean_str(text):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if type(text) is not unicode:
        text = text.decode('utf-8')

    # deemojize first
    text = emoji.demojize(text)

    FLAGS = re.MULTILINE | re.DOTALL

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " urlsignal")
    text = re_sub(r"@\w+", " usersignal ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " numbersignal ")

    text = re_sub(r'\s\s+', ' ')
    return text.lower()

def load_pretrained_word2vec(emb_size, id2word, pre_emb_path):
    word2vec_model = None
    vocab_size = len(id2word)

    # xavier initializer
    drange = np.sqrt(6. / (vocab_size + emb_size))
    W = np.random.uniform(low=-1, high=1, size=(vocab_size, emb_size)) * drange

    # load pre-trained word2vec
    if word2vec_model is None:
        word2vec_model = KeyedVectors.load_word2vec_format(pre_emb_path,unicode_errors='ignore',binary=use_binary)

    # assign
    total_loaded = 0
    for key, word in id2word.items():
        if word in word2vec_model.wv:
            W[key] = word2vec_model.wv[word]
            total_loaded += 1

    print ('-- Loaded from pretrained/ all word: %i/%i' % (total_loaded,vocab_size))

    return W

def load_train_file(file_path):
    df = pd.read_csv(file_path,sep='\t',encoding='utf-8',names=['label','text'])
    df['text'] = map(lambda x: clean_str(x), df['text'].tolist())

    print 'Loaded %i samples from data' % df.shape[0]

    return df['text'].tolist(), df['label'].tolist()

    # raw_data, raw_label = [], []
    #
    # for line in open(file_path,'r'):
    #     data = line.strip().split('\t')
    #
    #     if len(data) < 2:
    #         continue
    #
    #     label = data[0]
    #     text  = data[1]
    #
    #     if type(text) is not unicode:
    #         norm_data = text.decode('utf-8')
    #     else:
    #         norm_data = text
    #
    #     text = clean_str(norm_data)
    #
    #     raw_data.append(text)
    #     raw_label.append(label)
    #
    # print 'Loaded %i samples from data' % len(raw_data)
    #
    # return raw_data, raw_label

def mapping(dico):
    assert type(dico) is dict

    # sort item by descending its value (frequency of word)
    sorted_items = sorted(dico.items(),key=lambda elem: -elem[1])
    id2item = {i:v[0] for i,v in enumerate(sorted_items)}
    item2id = {v[0]:i for i,v in enumerate(sorted_items)}

    return id2item,item2id

def dict_from_list(lst):
    """
    create dict from list (occurrence)
    """
    assert type(lst) is list
    dict = {}

    for elems in lst:
        for elem in elems:
            if elem in dict:
                dict[elem] += 1
            else:
                dict[elem] = 1

    return dict


def word_mapping(lst_sentence,pre_emb=None):
    """
    mapping from list of sentences to dictionary
    with key = <word> and value = <frequency>
    example of input: [['hello','world'],...]
    """

    assert type(lst_sentence) is list
    global pretrain_word2vec

    # get dict of words
    dict_word = dict_from_list(lst_sentence)
    dict_word['<unk>'] = 1000001
    dict_word['<pad>'] = 1000000

    if pre_emb != None:
        word2vec_model = KeyedVectors.load_word2vec_format(pre_emb,unicode_errors='ignore',binary=use_binary)
        vocabs = word2vec_model.wv.vocab.keys()
        for word in vocabs:
            if word not in dict_word:
                dict_word[word] = 0
        pretrain_word2vec = word2vec_model

    id2word,word2id = mapping(dict_word)

    print "Found %i unique words (%i in total)" % (
        len(dict_word), sum(len(x) for x in lst_sentence)
    )

    return dict_word,id2word,word2id


def common_mapping(lst_x,name='x'):
    """
    similar to word_mapping,char_mapping
    but now for other type instead
    """
    assert type(lst_x) is list

    dict_x = dict_from_list(lst_x)
    id2x,x2id = mapping(dict_x)

    print "Found %i unique %s" % (len(dict_x),name)

    return dict_x,id2x,x2id

def create_batch(dataset,batch_size):
    batch_datas = []

    pre_batchs  = range(0,len(dataset),batch_size)
    next_batchs = [(i + batch_size) if (i + batch_size) < len(dataset) else len(dataset) for i in pre_batchs ]

    for s_i, e_i in zip(pre_batchs,next_batchs):
        if e_i > s_i:
            batch_datas.append(dataset[s_i:e_i])

    return batch_datas

def pad_common(sequences, pad_tok, max_length):
    """
    code from : https://github.com/guillaumegenthial/sequence_tagging original method name : _pad_sequence
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def create_dataset(words, labels, word2id, label2id):
    dataset = []

    for word, label in zip(words, labels):
        word_ids = [word2id[_token if _token in word2id else '<unk>'] for _token in word]

        label_ids = [0] * len(label2id)
        label_ids[label2id[label]] = 1

        data = {
            'word_ids': word_ids,
            'label_ids': label_ids,
        }

        dataset.append(data)

    return dataset

def add_to_dataset(datasets, xs, x2id, label):
    """
    add new feature to dataset
    """
    for dataset, x in zip(datasets,xs):
        x = [x2id[_x] for _x in x]
        dataset[label] = x

    return datasets

def split_batch(words, labels, test_percent):
    train_words, test_words, train_labels, test_labels = train_test_split(words,labels,test_size=test_percent,random_state=43,stratify=labels)

    return train_words, train_labels, test_words, test_labels

def convertglove(glove_path,word2vec_path):
    glove2word2vec(glove_path,word2vec_path)

import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

def word_tokenize(data,stemming=True):
    if type(data) is not unicode:
        data = unicode(data)

    tokens =  nltk.word_tokenize(data)

    return [porter_stemmer.stem(token.strip()) for token in tokens] if stemming else tokens

def pos_tagging(data, is_tokenizered = False):
    if is_tokenizered == False:
        tokens = word_tokenize(data=data)
        data = word_tokenize(tokens)

    pos = nltk.pos_tag(data)

    return [_pos[1].strip() for _pos in pos]

##########################################################
### Below code just for testing, plot confusion_matrix ###
##########################################################

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
from datetime import datetime
import random

def plot_conf_matrix(y_test, y_pred, class_names, prefix_image_path = '/home/hoaivan/Desktop/sentiment_testing_image'):
    cnf_matrix = confusion_matrix(y_test, y_pred, class_names)
    np.set_printoptions(precision=2)

    if not os.path.isdir(prefix_image_path):
        os.makedirs(prefix_image_path)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, save_image_path='tmp_image'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(save_image_path)

    # Plot non-normalized confusion matrix
    plt.figure()
    saved_image_without_normalize = '%s/%s_image_without_normalize_%s.png' % (prefix_image_path,str(datetime.now()),
                                                                              str(random.randint(0,100000)))
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization',
                          save_image_path = saved_image_without_normalize)


    # Plot normalized confusion matrix
    plt.figure()
    saved_image_with_normalize = '%s/%s_image_with_normalize_%s.png' % (prefix_image_path,str(datetime.now()),
                                                                              str(random.randint(0,100000)))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix',
                          save_image_path = saved_image_with_normalize)

    plt.show()

    return saved_image_without_normalize, saved_image_with_normalize

if __name__ == '__main__':
    str = clean_str(u'Python is 💙')
    import nltk

    #convertglove('./Data/glove.twitter.27B.200d.txt','./Data/w2v.twitter.27B.200d.txt')