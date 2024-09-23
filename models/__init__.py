from .models import SeqEncodingSeqPred  # Import your models here
from.layers_2 import Model1
def create_model(config):
    model_type = config['type']

    if model_type == 'seq_encoding_seq_pred':
        return SeqEncodingSeqPred(config)
    if model_type == 'model_1':
        return Model1()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
