from nn.nn_with_attention import Multi_Classifier as Multi_Classifier_ATT
from nn.nn_with_cnn import Multi_Classifier as Multi_Classifier_CNN
from nn.nn_with_ensemble import Multi_Classifier as Multi_Classifier_ENSEMBLE
from nn.nn_with_multigpu import Multi_Classifier as Multi_Classifier_MULTIGPU

CNN_MODEL_TYPE = 'cnn'
ATTENTION_MODEL_TYPE = 'attention'
ENSEMBLE_MODEL_TYPE = 'ensemble'
MULTIGPU_MODEL_TYPE = 'multigpu'

def get_corressponding_hypers(model_type):
    if model_type == ATTENTION_MODEL_TYPE:
        return './hyper_parameters/hyper_with_attention.json'
    elif model_type == CNN_MODEL_TYPE:
        return './hyper_parameters/hyper_with_cnn.json'
    else:
        raise Exception('Unknown Model Type')
    pass

def get_corressponding_params(model_type):
    if model_type == ATTENTION_MODEL_TYPE:
        return './params/params_with_attention.json'
    elif model_type == CNN_MODEL_TYPE:
        return './params/params_with_cnn.json'
    elif model_type == ENSEMBLE_MODEL_TYPE:
        return './params/params_with_ensemble.json'
    elif model_type == MULTIGPU_MODEL_TYPE:
        return './params/params_with_multigpu.json'
    else:
        raise Exception('Unknown Model Type.')
    pass

def get_corressponding_model(model_type):
    if model_type == CNN_MODEL_TYPE:
        model = Multi_Classifier_CNN
    elif model_type == ATTENTION_MODEL_TYPE:
        model = Multi_Classifier_ATT
    elif model_type == ENSEMBLE_MODEL_TYPE:
        model = Multi_Classifier_ENSEMBLE
    elif model_type == MULTIGPU_MODEL_TYPE:
        model = Multi_Classifier_MULTIGPU
    else:
        raise Exception('Unknown Model Type')

    return model