import torch
import torch.nn as nn

class SeqEmbedding(nn.Module):
    def __init__(self, seq_bits, vocab_len, embedding_dim, hidden_dim, output_dim, num_layers, padding_idx=-1):
        super(SeqEmbedding, self).__init__()
        # Embedding layer with padding index
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=padding_idx)
        self.fc_layers = nn.ModuleList()

        # First layer takes in concatenated input (seq_bits + embedding_dim) and outputs to hidden_dim
        self.fc_layers.append(nn.Linear(seq_bits + embedding_dim, hidden_dim))

        # Create intermediate hidden layers if num_layers > 1
        for _ in range(num_layers - 2):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Final layer goes from hidden_dim to output_dim
        self.fc_layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, t_seq, seq_indices):
        # Pass the seq_indices through the embedding layer

        embedded_seq = self.embedding(seq_indices)
        # Concatenate t_seq and embedded_seq along the last dimension

        x = torch.cat((t_seq, embedded_seq), dim=-1)
        # Pass through all the fully connected layers
        for layer in self.fc_layers:
            x = torch.relu(layer(x))  # Apply ReLU after each layer

        return x


class SeqInput(nn.Module):
    def __init__(self, seq_bits, input_embedding_dim, hidden_dim, output_dim, num_layers ,vocab_len, padding_idx = -1):
        super(SeqInput, self).__init__()
        self.embedding = nn.Embedding(vocab_len, input_embedding_dim, padding_idx=padding_idx)
        self.fc_layers = nn.ModuleList()  # To store the layers

        # First layer takes in concatenated input (seq_bits + input_embedding_dim) and outputs to hidden_dim
        self.fc_layers.append(nn.Linear(seq_bits + 2* input_embedding_dim, hidden_dim))

        # Create intermediate hidden layers if num_layers > 1
        for _ in range(num_layers - 2):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Final layer goes from hidden_dim to output_dim
        self.fc_layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, t_seq, input_embedding, seq_indices):
        # Concatenate t_seq and input_embedding along the last dimension
        embedding = self.embedding(seq_indices)
        x = torch.cat((t_seq, input_embedding, embedding), dim=-1)  ## i need the individual file here too!!

        # Pass through all the fully connected layers
        for layer in self.fc_layers:
            x = torch.relu(layer(x))  # Apply ReLU after each layer

        return x


class FileEmbeddingModel(nn.Module):
    def __init__(self, embedding_input_dim, bits, hidden_dim, num_layers):
        super(FileEmbeddingModel, self).__init__()

        input_dim = bits + embedding_input_dim  # Adjust input dimension
        # Create a list of layers
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())  # Add ReLU activation after each linear layer

        # Use nn.Sequential to stack the layers
        self.hidden_layers = nn.Sequential(*layers)

        # Output layers for binary cross-entropy and MSE
        self.bce_output = nn.Linear(hidden_dim, 1)  # For binary cross-entropy (single scalar)
        self.mse_output = nn.Linear(hidden_dim, 1)  # For mean squared error (single scalar)

    def forward(self, embedding_tensor, bits_tensor):
        # Concatenate the input embedding with the bits_tensor
        concatenated = torch.cat((embedding_tensor, bits_tensor),
                                 dim=1)  # Output: [batch_size, bits + embedding_input_dim]

        # Pass through the dynamically created hidden layers
        hidden_out = self.hidden_layers(concatenated)

        # Compute both outputs
        bce_out = torch.sigmoid(self.bce_output(hidden_out))  # Binary classification output
        mse_out = self.mse_output(hidden_out)  # Regression output

        return bce_out, mse_out
