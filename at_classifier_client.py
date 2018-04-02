#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, json
from celery import Celery
import os

app = Celery('at_classifier_celery',
             broker='amqp://atml:atml123@localhost/atml_vhost',
             backend='rpc://',
             include=['at_classifier'])

"""
Example for predict
"""
long_sent = "i'll cheer you up boo"

sents = [long_sent] * 1
list_text = json.dumps(sents)

result = app.send_task("at_classifier.predict",[list_text,True],queue='at_sentiment')

print(result.ready())

time.sleep(2)

print(result.ready())
print(json.loads(result.get()))

# """
# Example for train_next
# """
# prefix_path = '/home/hoaivan/Desktop/atvn_workspace/sw_follou/Classifier/General_Classifier'
# train_data_path = os.path.join(prefix_path,'data','sentiment_standard_standford','fine','merge','test.txt')
# test_data_path  = os.path.join(prefix_path,'data','sentiment_standard_standford','fine','merge','test.txt')
# out_model_path = os.path.join(prefix_path,'saved_model','sentiment_standard_standford_full_fine_attention_test_v2')
# params = {
#     'n_epoch': 10,
#     'freq_eval': 20,
#     'metric': 'f1' # [precision, recall, f1]
# }
#
# app.send_task("at_classifier.train_next",[out_model_path,train_data_path,test_data_path,json.dumps(params)],queue='at_sentiment')
