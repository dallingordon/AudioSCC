import torch
import torch.nn as nn


class SeqEmbeddingLayer(nn.Module):

    def __init__(self, t_bits, t_bits_hidden, t_bits_layers, vocab_len, embedding_dim, hidden_dim, hidden_add,
                 output_dim, num_layers, input_dim=0):
        super(SeqEmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=-1)

        self.t_layers = nn.ModuleList()
        if t_bits_layers > 0:
            self.t_layers.append(nn.Linear(t_bits, t_bits_hidden))  # First layer: t_bits x t_bits_hidden
            for _ in range(t_bits_layers - 1):
                self.t_layers.append(nn.Linear(t_bits_hidden, t_bits_hidden))

        self.fc_input = nn.Linear(
            t_bits_hidden + input_dim + embedding_dim if t_bits_layers > 0 else t_bits + input_dim + embedding_dim,
            hidden_dim)

        # Define hidden layers
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim + i * hidden_add, hidden_dim + (i + 1) * hidden_add) for i in range(num_layers)])

        # Output layer
        self.fc_output = nn.Linear(hidden_dim + hidden_add * num_layers, output_dim)

    def forward(self, t_seq, one_hot, input_embedding=None):

        # t_seq: [batch, t_seq_len, t_bits]
        # one_hot: [batch, t_seq_len, vocab_dim]
        # input_embedding: [batch, t_seq_len, input_dim] (optional)

        e = self.embedding(one_hot)  # Shape: [batch, t_seq_len, embedding_dim]

        # Process t_seq through t_layers if they exist
        if len(self.t_layers) > 0:
            for layer in self.t_layers:
                t_seq = torch.relu(layer(t_seq))  # Shape: [batch, t_seq_len, t_bits_hidden]

        # Else keep t_seq as is

        t = t_seq

        # Concatenate t_seq, embedding, and input_embedding (if provided)
        if input_embedding is not None:

            x = torch.cat((t, e, input_embedding),
                          dim=-1)  # Shape: [batch, t_seq_len, t_bits_hidden + embedding_dim + input_dim]

        else:

            x = torch.cat((t, e), dim=-1)  # Shape: [batch, t_seq_len, t_bits_hidden + embedding_dim]

        # Pass through the input layer

        x = self.fc_input(x)

        # Pass through the hidden layers
        for layer in self.layers:
            x = torch.relu(layer(x))

        # Output layer

        x = torch.relu(self.fc_output(x))

        return x




class SeqEmbeddingDecoderLayer(nn.Module):
    def __init__(self, t_bits, t_bits_hidden, t_bits_layers, hidden_dim, hidden_add, num_layers, fc_output_dim,
                 input_dim=0):
        super(SeqEmbeddingDecoderLayer, self).__init__()

        self.t_layers = nn.ModuleList()
        if t_bits_layers > 0:
            self.t_layers.append(nn.Linear(t_bits, t_bits_hidden))  # First layer: t_bits x t_bits_hidden
            for _ in range(t_bits_layers - 1):
                self.t_layers.append(nn.Linear(t_bits_hidden, t_bits_hidden))

        self.fc_input = nn.Linear(t_bits_hidden + input_dim if t_bits_layers > 0 else t_bits + input_dim, hidden_dim)

        # Define hidden layers
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim + i * hidden_add, hidden_dim + (i + 1) * hidden_add) for i in range(num_layers)])

        # Output layer before splitting into separate stacks
        self.fc_output = nn.Linear(hidden_dim + hidden_add * num_layers, fc_output_dim)

    def forward(self, t_seq, input_embedding):
        # Process t_seq through t_layers if they exist

        if len(self.t_layers) > 0:
            for layer in self.t_layers:
                t_seq = torch.relu(layer(t_seq))  # Shape: [batch, t_seq_len, t_bits_hidden]

        # Concatenate t_seq with input_embedding
        x = torch.cat((t_seq, input_embedding),
                      dim=-1)  # Shape: [batch, t_seq_len, t_bits_hidden + embedding_dim + input_dim]

        # Pass through the input layer
        x = torch.relu(self.fc_input(x))

        # Pass through the hidden layers
        for layer in self.layers:
            x = torch.relu(layer(x))

        # Pass through the output layer
        x = torch.relu(self.fc_output(x))

        return x  # a latent space



class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        # Constants for SeqEmbeddingLayer
        t_bits = 5
        vocab_len = 11

        self.bce_layers = nn.ModuleList([nn.Linear(512, 100)
                                            , nn.Linear(100, 100)
                                            , nn.Linear(100, 100)])
        self.mse_layers = nn.ModuleList([nn.Linear(512, 100)
                                            , nn.Linear(100, 100)
                                            , nn.Linear(100, 100)])

        # Final output heads: BCE and MSE
        self.fc_bce = nn.Linear(100, 1)  # Output for BCE: [batch, 1]
        self.fc_mse = nn.Linear(100, 1)
        # Initialize 3 Encoding layers (SeqEmbeddingLayer)
        self.encoder_layers = nn.ModuleList([
            SeqEmbeddingLayer(
                t_bits=t_bits,  # for t_seq
                t_bits_hidden=20,
                t_bits_layers=2,
                vocab_len=vocab_len,
                embedding_dim=64,
                hidden_dim=64,
                hidden_add=20,
                output_dim=128,  # Output hidden_dim for stacking
                num_layers=3,
                input_dim=0  # No input embedding for the first layer
            ),
            SeqEmbeddingLayer(
                t_bits=t_bits,
                t_bits_hidden=20,
                t_bits_layers=2,
                vocab_len=vocab_len,
                embedding_dim=64,
                hidden_dim=128,
                hidden_add=40,
                output_dim=256,  # Output hidden_dim for stacking
                num_layers=3,
                input_dim=128  # Previous layer's output as input embedding
            ),
            SeqEmbeddingLayer(
                t_bits=t_bits,
                t_bits_hidden=20,
                t_bits_layers=2,
                vocab_len=vocab_len,
                embedding_dim=64,
                hidden_dim=256,
                hidden_add=60,
                output_dim=512,  # Output hidden_dim for stacking
                num_layers=3,
                input_dim=256  # Previous layer's output as input embedding
            )
        ])

        # Initialize 3 Decoding layers (SeqEmbeddingDecoderLayer)
        self.decoder_layers = nn.ModuleList([
            SeqEmbeddingDecoderLayer(
                t_bits=20,  # for t not t_seq
                t_bits_hidden=40,
                t_bits_layers=3,
                hidden_dim=256,
                hidden_add=50,
                num_layers=3,
                fc_output_dim=512,
                input_dim=512  # Input from the encoder output
            ),
            SeqEmbeddingDecoderLayer(
                t_bits=20,
                t_bits_hidden=40,
                t_bits_layers=3,
                hidden_dim=512,
                hidden_add=0,
                num_layers=3,
                fc_output_dim=512,
                input_dim=512  # Input from previous decoder layer
            ),
            SeqEmbeddingDecoderLayer(
                t_bits=20,
                t_bits_hidden=40,
                t_bits_layers=3,
                hidden_dim=512,
                hidden_add=0,
                num_layers=3,
                fc_output_dim=512,
                input_dim=512
            )
        ])

    def forward(self, t_seq, one_hot, t):
        # Pass through the encoding layers
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                latent = layer(t_seq, one_hot)  # Pass the output from one layer to the next

            else:

                latent = latent.sum(dim=1, keepdim=True).expand(-1, 10, -1)

                latent = layer(t_seq, one_hot, latent)

        # Use the encoder's final output as input for the decoder
        input_embedding = latent.sum(dim=1, keepdim=False)

        # Pass through the decoding layers
        for i, layer in enumerate(self.decoder_layers):
            input_embedding = layer(t, input_embedding)

        bce_out = input_embedding
        mse_out = input_embedding
        for i, layer in enumerate(self.bce_layers):
            bce_out = torch.relu(layer(bce_out))

        for i, layer in enumerate(self.mse_layers):
            mse_out = torch.relu(layer(mse_out))

        bce_out = torch.sigmoid(self.fc_bce(bce_out))
        mse_out = torch.relu(self.fc_mse(mse_out))

        return bce_out, mse_out
