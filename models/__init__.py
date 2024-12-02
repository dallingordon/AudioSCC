from .models import SeqEncodingSeqPred  # Import your models here
from.layers_2 import Model1
from .SwissArmyModel import SeqModel
from .swissarmy_prev import SeqModel as SeqPred
from .SwissArmyModel_dev import SeqModel as ResPred
def create_model(config):
    model_type = config['type']

    if model_type == 'seq_encoding_seq_pred':
        return SeqEncodingSeqPred(config)
    if model_type == 'model_1':
        return Model1()
    if model_type == 'seq_model':
        return SeqModel(config)
    if model_type == 'seq_pred':
        return SeqPred(config)
    if model_type == 'res_pred':
        return ResPred(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
