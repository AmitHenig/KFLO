from base_model.vgg import *
from base_model.cfqk import create_CFQKBNC

CIFAR10_MODEL_MAP = {
    'vc':create_vc,
    'cfqkbnc':create_CFQKBNC
}

DATASET_TO_MODEL_MAP = {
    'cifar10': CIFAR10_MODEL_MAP,
}


#   return the model creation function
def get_model_fn(dataset_name, model_name):
    # print(DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')].keys())
    return DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')][model_name]

def get_dataset_name_by_model_name(model_name):
    for dataset_name, model_map in DATASET_TO_MODEL_MAP.items():
        if model_name in model_map:
            return dataset_name
    return None