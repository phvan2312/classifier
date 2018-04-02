import json

from celery.signals import worker_process_init

from at_classifier_server import app
from train_continue import OldModel as Model

best_model = None
best_model_path = './saved_model/sentiment_standard_standford_full_fine_attention_2603/'

@worker_process_init.connect
def init(sender, **k):
    global best_model, best_model_path
    best_model = Model(save_path=best_model_path)

@app.task
def train_next(out_model_path, train_data_path, test_data_path, params_str):
    print ('-- reading parametes', params_str)
    params = json.loads(params_str)

    print ('-- making dataset')
    train_dataset, test_dataset = best_model.make_dataset(train_path=train_data_path, test_path=test_data_path,
                                                   is_convert_slang=best_model.params.get('convert_slang', True),
                                                   is_pos=best_model.params.get('pos', False),
                                                   is_sentiment_lexicon=best_model.params.get('sentiment_lexicon', False),
                                                   is_vader_lexicon=best_model.params.get('vader_lexicon', False))

    print ('-- start training')
    best_model.train_next(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=512,
                          out_path=out_model_path,params=params)


@app.task
def predict(list_text, get_probs = False):
    global best_model
    _list_text = json.loads(list_text)
    results = best_model.predict(_list_text)

    #
    # logging,
    #
    for (sen, res) in zip(_list_text, results):
        print ('sen: %s <--> predicted label: %s,%f' % (sen, res[0].replace('__label__',''), res[1]))

    #
    # normalize responds from server and return
    #

    labels = [label[0].replace('__label__', '') for label in results]

    if get_probs :
        # get corressponding probabilities
        probs = [{k:str(v) for k,v in elem[2].items()} for elem in results]

        responds = [(label,prob) for label,prob in zip(labels,probs)]

    else:
        responds = labels

    return json.dumps(responds,indent=2)


if __name__ == '__main__':
    pass
    #best_model = Inquiry()