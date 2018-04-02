from celery import Celery

app = Celery('at_classifier_celery',
             broker='amqp://atml:atml123@localhost/atml_vhost',
             backend='rpc://',
             include=['at_classifier'])
