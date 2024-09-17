from .model import SeqEncodingSeqPred  # Import your models here

def create_model(config):
    model_type = config['model']['type']

    if model_type == 'seq_encoding_seq_pred':
        return SeqEncodingSeqPred(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
