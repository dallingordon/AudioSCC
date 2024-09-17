import argparse
import yaml
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from data.dataloader import WaveformDatasetPreload  # Import your custom dataloader
from data.sampler import RandomConsecutiveSampler
from your_model import YourModel  # Import your model class
from losses.losses import ConsecutiveDifferenceHigherOrderLossBatch, ConsecutiveDifferenceHigherOrderLoss  # Updated import for loss functions
from scripts.utils import get_max_required_length, binary_sequence_tensor
from tqdm import tqdm

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Setup device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Create dataloader
    max_len = get_max_required_length(config['data_directory'])
    target_pad = config['target_pad']  # Update this in your config file
    bits = config['bits']  # Number of bits for t_input
    seq_bits = config['seq_bits']  # Number of bits for seq_t
    seq_max_len = config['seq_max_len']
    seq_vocab_len = config['seq_vocab_len']

    t_input = binary_sequence_tensor(bits, max_len + target_pad)
    seq_t_input = binary_sequence_tensor(seq_bits, seq_max_len + 1)

    # Create the dataset
    dataset = WaveformDatasetPreload(
        directory=config['data_directory'],
        t_input=t_input,
        max_len=max_len,
        terminal_pad=target_pad,
        seq_vocab_len=seq_vocab_len,
        seq_max_len=seq_max_len,
        seq_t=seq_t_input
    )

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    consecutive_size = config['consecutive_size']

    sampler = RandomConsecutiveSampler(dataset, batch_size, consecutive_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers)

    # Initialize model, optimizer, and loss functions
    model = YourModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()
    cdifb_loss = ConsecutiveDifferenceHigherOrderLossBatch(config['consecutive_size'], order=config['order'])
    cdif_loss = ConsecutiveDifferenceHigherOrderLoss(config['consecutive_size'], order=config['order'])
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            wav_data, t_step, target, file_idx, seq_inputs = batch
            wav_data, target, seq_inputs = wav_data.to(device), target.to(device), seq_inputs.to(device)

            # Forward pass
            bce_output, mse_output = model(seq_inputs, file_idx, t_step)

            # Compute losses
            mse_loss = mse_loss_fn(mse_output * target, wav_data)
            bce_loss = bce_loss_fn(bce_output, target)
            cdif = cdif_loss(mse_output * target, wav_data)
            cdif_b = cdifb_loss(mse_output * target, wav_data)

            # Combine losses
            total_loss = mse_loss + 0.1 * bce_loss + 1.6 * cdif + 0.4 * cdif_b

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Print progress for each epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} MSE: {mse_loss.item():.6f} BCE: {bce_loss.item():.6f} CDIF: {cdif.item():.6f} CDIF_B: {cdif_b.item():.6f} Total Loss: {total_loss.item():.8f}")

        # Save model checkpoint
        torch.save(model.state_dict(), "digits5k_second_try.pth")

    print("Training complete!")


if __name__ == "__main__":
    main()
