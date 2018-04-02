#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import cPickle

from vars import get_corressponding_model
from utils import create_dataset, add_to_dataset, load_train_file, word_tokenize, pos_tagging, create_batch, clean_str

import train
import pandas as pd

from progress.bar import ChargingBar as Bar
from prettytable import PrettyTable

load_train_file('./add.txt')

def load_dict(path):
    with open(path, 'r') as f:
        dct = cPickle.load(f)
    return dct

class OldModel:
    def __init__(self, save_path):
        assert os.path.exists(save_path)
        self.save_path = save_path

        # hyper parameter
        self.save_model_path = save_path + "model.ckpt"
        self.save_word_path = save_path + "vocab/word.pkl"
        self.save_label_path = save_path + "vocab/label.pkl"
        self.save_lexicon_path = save_path + "vocab/lexicon.pkl"
        self.save_param_path = save_path + "vocab/params.pkl"
        self.save_pos_path = save_path + "vocab/pos.pkl"
        self.save_vader_lexicon_path = save_path + "vocab/vader_lexicon.pkl"

        self.__build_vocab()
        self.__build_model()

        slang_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/slang/slang.pkl'
        self.slang_dict = load_dict(slang_path)

        lexicon_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/lexicon/lexicon.pkl'
        self.lexicon_dct = load_dict(lexicon_path)

        #self.out_path = out_path

    def __build_vocab(self):
        def load(save_path):
            if os.path.isfile(save_path):
                with open(save_path, 'r') as f:
                    return cPickle.load(f)
            else:
                return None

        self.word_vocab  = load(self.save_word_path)
        self.label_vocab = load(self.save_label_path)
        self.params      = load(self.save_param_path)
        self.lexicon_vocab = load(self.save_lexicon_path)
        self.pos_vocab = load(self.save_pos_path)
        self.vader_lexicon_vocab = load(self.save_vader_lexicon_path)

    def predict(self,raw_sentences):
        if type(raw_sentences) is not list: raw_sentences = [raw_sentences]

        if self.params.get('convert_slang', True):
            split_sentences = [sentence.split(' ') for sentence in raw_sentences]
            convert_sentences = [[self.slang_dict[_word] if _word in self.slang_dict else _word for _word in word]
                                 for word in split_sentences]

            raw_sentences = [clean_str(' '.join(sentence)) for sentence in convert_sentences]

        with tf.device('/cpu:0'):
            sess, word2id, label2id = self.model.sess, self.word_vocab['word2id'], self.label_vocab['label2id']

        # tokenize word and create faked labels
        sentences = [word_tokenize(sentence) for sentence in raw_sentences]
        labels = [self.label_vocab['id2label'][0]] * len(sentences)  # just faked label

        dataset = create_dataset(sentences, labels, word2id, label2id)

        #
        if self.params.get('sentiment_lexicon', False):
            lexicons = train.convert_sentiment_lexicon(self.lexicon_vocab['lexicon2id'],sentences)
            dataset = add_to_dataset(dataset, lexicons, self.lexicon_vocab['lexicon2id'], 'lexicon_ids')

        #
        if self.params.get('pos', False):
            pos = train.convert_pos(self.pos_vocab['pos2id'],sentences)
            dataset = add_to_dataset(dataset, pos, self.pos_vocab['pos2id'], 'pos_ids')

        #
        if self.params.get('vader_lexicon',False):
            vaderlexicons = train.convert_vader_lexicon(self.vader_lexicon_vocab['vaderlexicon2id'], sentences)
            dataset = add_to_dataset(dataset, vaderlexicons, self.vader_lexicon_vocab['vaderlexicon2id'], 'vader_lexicon_ids')

        # prepare input feed dict
        ip_feed_dict = self.model.create_input(dataset)
        for k, v in self.model.create_additional_input(dataset).items():
            ip_feed_dict[k] = v

        if hasattr(self.model, 'dropout'):
            ip_feed_dict[self.model.dropout] = 1.0

        if hasattr(self.model, 'is_training'):
            ip_feed_dict[self.model.is_training] = False

        predict, predict_proba = sess.run([self.model.logits, tf.nn.softmax(self.model.logits)],
                                          feed_dict=ip_feed_dict)

        predict_ids = np.argmax(predict, axis=1)
        return \
            [
                ((
                    self.label_vocab['id2label'][predict_id],
                    predict_proba[id][predict_id],
                    {self.label_vocab['id2label'][i]: p for i, p in enumerate(predict_proba[id])}
                ))
                for id, predict_id in enumerate(predict_ids)
            ]

    def __build_model(self):

        params = self.params
        params['use_pretrained'] = None

        with tf.device('/cpu:0'):
            self.model = get_corressponding_model(params['model_type'])(**params)
            self.model.build_model()

            self.saver = tf.train.Saver()
            self.saver.restore(self.model.sess, self.save_model_path)

        ### PRINT ###
        print 'Model restored ...'

    def make_dataset(self,train_path,test_path,is_convert_slang,is_sentiment_lexicon,is_pos,is_vader_lexicon):
        # load file
        raw_data, raw_labels = load_train_file(train_path)
        test_data, test_labels = load_train_file(test_path)
        len_train = len(raw_data)

        print ('-- tokenize words')
        words, labels = [word_tokenize(data) for data in raw_data + test_data], [label for label in raw_labels + test_labels]

        #
        # convert slang into its corresponding word (maybe not?)
        #
        if is_convert_slang:
            slang_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/slang/slang.pkl'
            slang_dict = load_dict(slang_path)

            words = train.convert_slang(slang_dict, words)

        #
        # split train and test.
        #
        print ('-- split train and test data')
        train_words, train_labels, test_words, test_labels = words[:len_train], labels[:len_train], words[len_train:], labels[len_train:]

        #
        # create dataset
        #
        train_dataset = create_dataset(train_words, train_labels, self.word_vocab['word2id'], self.label_vocab['label2id'])
        test_dataset = create_dataset(test_words, test_labels, self.word_vocab['word2id'], self.label_vocab['label2id'])

        #
        # add sentiment_lexicon as additional features (maybe not?)
        #
        if is_sentiment_lexicon:
            lexicon_path = os.path.dirname(os.path.realpath(__file__)) + '/data/preprocess/lexicon/lexicon.pkl'
            lexicon_dct = load_dict(lexicon_path)

            lexicons = train.convert_sentiment_lexicon(lexicon_dct, words, 'neu')
            train_lexicons, test_lexicons = lexicons[:len_train], lexicons[len_train:]

            train_dataset = add_to_dataset(train_dataset, train_lexicons, self.lexicon_vocab['lexicon2id'], 'lexicon_ids')
            test_dataset = add_to_dataset(test_dataset, test_lexicons, self.lexicon_vocab['lexicon2id'], 'lexicon_ids')

        #
        # add pos (part-of-speech) as additional features (maybe not?)
        #
        if is_pos:
            pos = train.convert_pos(words)
            train_pos, test_pos = pos[:len_train], pos[len_train:]

            train_dataset = add_to_dataset(train_dataset, train_pos, self.pos_vocab['pos2id'], 'pos_ids')
            test_dataset = add_to_dataset(test_dataset, test_pos, self.pos_vocab['pos2id'], 'pos_ids')

        #
        # add vader_lexicon as additional features (maybe not?)
        #
        if is_vader_lexicon:
            vaderlexicons = train.convert_vader_lexicon(self.vader_lexicon_vocab['vaderlexicon2id'],words)
            train_vaderlexicons, test_vaderlexicons = vaderlexicons[:len_train], vaderlexicons[len_train:]

            train_dataset = add_to_dataset(train_dataset, train_vaderlexicons, self.vader_lexicon_vocab['vaderlexicon2id'], 'vader_lexicon_ids')
            test_dataset = add_to_dataset(test_dataset, test_vaderlexicons, self.vader_lexicon_vocab['vaderlexicon2id'], 'vader_lexicon_ids')

        return train_dataset, test_dataset

    def train_next(self, train_dataset, test_dataset, batch_size, out_path, params = {},**kargs):
        # copy existing materials to new path
        if out_path != self.save_path:
            train.copy(self.save_path,out_path)

        save_model_path = os.path.join(out_path,'model.ckpt')

        # setting some parameters
        n_epoch = params.get('n_epoch',10)
        freq_eval = params.get('freq_eval',20)
        metric = params.get('metric','f1')

        #
        # build batches
        #
        train_batch, test_batch = create_batch(train_dataset, batch_size), create_batch(test_dataset,batch_size)

        print ('-- start training')
        train_count = 0
        eval_count = 0
        best_test = - np.inf
        decay_lr_every = 500
        lr_decay = 0.9
        init_lr = self.model.lr


        #
        # training
        #
        for epoch in range(n_epoch):
            print ("\n[E]poch %i, lr %.4f" % (epoch + 1, init_lr))
            bar = Bar('training', max=len(train_batch), suffix='')
            tabl = PrettyTable(['status', 'train_epoch', 'train_batch', 'test_avg_score', 'test_avg_loss'])

            loss_total, acc_total = [], []

            for i, batch_id in enumerate(np.random.permutation(len(train_batch))):
                train_count += 1
                batch = train_batch[i]

                loss, acc = self.model.batch_run(batch=batch, i=train_count, mode='train', metric=metric)

                loss_total.append(loss)
                acc_total.append(acc)

                bar.bar_prefix = " | batch %i |" % (batch_id + 1)
                bar.bar_suffix = " | cur_loss: %.3f | cur_acc: %.3f | best_test_acc: %.3f" % (loss, acc, best_test)

                bar.next()

                if train_count % decay_lr_every == 0: init_lr = init_lr * lr_decay

                if train_count % freq_eval == 0:
                    test_loss_lst, test_acc_lst = [], []

                    for t_i, t_batch_id in enumerate(range(len(test_batch))):
                        t_batch = test_batch[t_batch_id]
                        eval_count += 1

                        loss, acc = self.model.batch_run(batch=t_batch, i=eval_count, mode='eval', metric=metric,
                                                    lr=init_lr)

                        test_loss_lst.append(loss)
                        test_acc_lst.append(acc)

                    test_loss = np.array(test_loss_lst).mean()
                    test_acc = np.array(test_acc_lst).mean()

                    status = '++++'
                    if test_acc > best_test:
                        best_test = test_acc
                        status = 'best'

                        self.model.save(save_model_path)

                    tabl.add_row([status, epoch + 1, batch_id + 1, '%.3f' % test_acc, '%.3f' % test_loss])

            print "\nmean_train_loss: %.3f, mean_train_acc: %.3f" % (np.mean(loss_total), np.mean(acc_total))
            print (tabl.get_string(title="Local All Test Accuracies"))

if __name__ == '__main__':
    save_path, out_path = './saved_model/sentiment_standard_standford_full_fine_attention_test_v2/', './saved_model/sentiment_standard_standford_full_fine_attention_test_v2/'

    train_path = './add.txt' #'./data/sentiment_standard_standford/fine/merge/test.txt'
    test_path  = './add.txt' #'./data/sentiment_standard_standford/fine/merge/test.txt'

    old = OldModel(save_path)

    """
    Example for continue training
    """
    # train_dataset, test_dataset = old.make_dataset(train_path=train_path, test_path=test_path,
    #                                                is_convert_slang=old.params.get('convert_slang', True),
    #                                                is_pos=old.params.get('pos', False),
    #                                                is_sentiment_lexicon=old.params.get('sentiment_lexicon', False),
    #                                                is_vader_lexicon=old.params.get('vader_lexicon', False))
    #
    # params = {
    #     'n_epoch': 20,
    #     'freq_eval':2,
    #     'metric':'f1'
    # }
    # # """
    # # n_epoch = params.get('n_epoch', 10)
    # # freq_eval = params.get('freq_eval', 20)
    # # metric = params.get('metric', 'f1')
    # # """
    # old.train_next(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=512, out_path=out_path,
    #                params=params)
    #
    # exit()

    """
    Example for prediction
    """

    sentences = [
        "the club isn't the best place to find the lover",
        "#messi #neymar #rolnaldo",
        "Soul is what's lacking in every character in this movie and , subsequently , the movie itself .",
        "The 3D images only enhance the film's otherworldly quality , giving it a strange combo of you-are-there closeness with the disorienting unreality of the seemingly broken-down fourth wall of the movie screen .",
        "i don't like its decoration",
        "much",
        "123456",
        "i don't like it",
        "this post is great",
        "fuck you bitch",
        "i hate her",
        "i love you and i like you",
        "i love her but i hate her song",
        "i hate her but i love her song",
        "i like it ",
        "i luv you",
        "i love it so much !",
        "you are poor",
        "An utterly compelling ` who wrote it ' in which the reputation of the most famous author who ever lived comes into question .",
        "this post is fucking good",
        "this post is fucking awesome",
        "Happy new year",
        "bad movie",
        "i'm happy",
        "this post is great .",
        "this post is not great",
        "The front desk were very cordis",
        "The front desk were very cordial",
        "All the staffs were very friendly and helpful",
        "We had a great experience at the hotel",
        "Modern and stylish",
        "Highly recommended if you want a bit of luxury",
        "Highly recommend this place to stay",
        "Looking a relaxing beach. Thanks",
        "Quiet, comfortable, clean and in tip top condition",
        "This was hands down one of the nicest hotels we've ever stayed",
        "Good furniture, comfortable beds",
        "I stayed there many days ago. It was perfect",
        "The hotel was so nice",
        "Good drink",
        "What a good place to stay",
        "I'm not satisfied about services",
        "Pick up service is so poor",
        "The shuttle service was so bad",
        "Poor facilities",
        "this is good one",
        "💙",
        ":blue_heart:",
        "blue heart",
        ":the_gioi_dong_vat:",
        "💕"
    ]

    # df = pd.read_csv('./processed_data_emoji.csv',sep='\t',encoding='utf-8',names=['label','text'])
    # sentences = sentences + df['text'].tolist()

    #sentences = pd.read_csv('./add.txt',sep='\t',encoding='utf-8',names=['label','text'])['text'].tolist()

    results = old.predict(sentences)
    for (sen, res) in zip(sentences, results):
        print ('sen: %s <--> label: %s,%f' % (sen, res[0], res[1]))

    """
    Nonsense. Don't care
    """

    # import pandas as pd
    # df = pd.read_csv('./processed_data.csv',sep='\t',names=['label','text'],encoding='utf-8')
    #
    # sentences = df['text'].tolist()
    # real_labels = df['label'].tolist()
    #
    # results = old.predict(sentences)
    # new_real_labels = []
    # for (real_label, res) in zip(real_labels, results):
    #     if real_label == '__label__positive' and res[0] in ['__label__positive', '__label__very_positive']:
    #         new_real_labels.append(res[0])
    #     elif real_label == '__label__negative' and res[0] in ['__label__negative', '__label__very_negative']:
    #         new_real_labels.append(res[0])
    #     else:
    #         new_real_labels.append(real_label)
    #     #print ('sen: %s <--> label: %s,%f' % (sen, res[0], res[1]))
    #
    # df['label'] = new_real_labels
    #
    # df.to_csv('./processed_data.csv',sep='\t',header=False,index=False,columns=['label','text'])