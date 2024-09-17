from models.layers import SeqEmbedding, SeqInput,FileEmbeddingModel
import torch
import torch.nn as nn
from models.layers import SeqEmbedding, SeqInput, FileEmbeddingModel  # Import your layer classes

class SeqEncodingSeqPred(nn.Module):
    def __init__(self, config):
        super(SeqEncodingSeqPred, self).__init__()
        # Initialize hyperparameters from the config
        seq_embedding_params = config['seq_embedding']
        seq_input_params = config['seq_input']
        file_embedding_params = config['file_embedding']

        # Instantiate SeqEmbedding
        self.seq_embedding = SeqEmbedding(
            seq_embedding_params['seq_bits'],
            seq_embedding_params['vocab_len'],
            seq_embedding_params['embedding_dim'],
            seq_embedding_params['hidden_dim'],
            seq_embedding_params['output_dim'],
            seq_embedding_params['num_layers'],
            seq_embedding_params['padding_idx']
        )

        # Instantiate SeqInput
        self.seq_input = SeqInput(
            seq_input_params['seq_bits'],
            seq_input_params['input_embedding_dim'],
            seq_input_params['hidden_dim'],
            seq_input_params['output_dim'],
            seq_input_params['num_layers'],
            seq_embedding_params['vocab_len'],
            seq_embedding_params['padding_idx']
        )

        # Instantiate FileEmbeddingModel
        self.file_embedding = FileEmbeddingModel(
            file_embedding_params['embedding_input_dim'],
            file_embedding_params['bits'],
            file_embedding_params['hidden_dim'],
            file_embedding_params['num_layers']
        )

    def forward(self, t_seq, seq_indices, t):
        # Pass t_seq and seq_indices through SeqEmbedding
        seq_embedding_output = self.seq_embedding(t_seq, seq_indices)

        # Sum over the last dimension of SeqEmbedding's output
        summed_seq_embedding_output = torch.sum(seq_embedding_output, dim=-2)
        summed_seq_embedding_output = summed_seq_embedding_output.unsqueeze(1)
        summed_seq_embedding_output = summed_seq_embedding_output.repeat(1, 10, 1)

        # Pass summed output and t_seq through SeqInput
        seq_input_output = self.seq_input(t_seq, summed_seq_embedding_output, seq_indices)

        # Sum over the last dimension of SeqInput's output
        summed_seq_input_output = torch.sum(seq_input_output, dim=-2)

        # Pass the summed output and t through FileEmbeddingModel
        bce_out, mse_out = self.file_embedding(summed_seq_input_output, t)

        return bce_out, mse_out
