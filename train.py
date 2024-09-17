import os
import argparse
import yaml
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from data.dataloader import WaveformDatasetPreload
from data.samplers import RandomConsecutiveSampler
from losses.losses import ConsecutiveDifferenceHigherOrderLossBatch, ConsecutiveDifferenceHigherOrderLoss
from models import create_model
from scripts.utils import get_max_required_length, binary_sequence_tensor
from tqdm import tqdm

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create experiment directory
    experiment_name = config['experiment_name']
    output_dir = os.path.join("outputs", experiment_name)
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Create dataloader
    data_directory = config['data']['data_directory']
    max_len = get_max_required_length(data_directory)
    target_pad = config['data']['target_pad']
    bits = config['data']['bits']
    seq_bits = config['data']['seq_bits']
    seq_max_len = config['data']['seq_max_len']
    seq_vocab_len = config['data']['seq_vocab_len']

    t_input = binary_sequence_tensor(bits, max_len + target_pad)
    seq_t_input = binary_sequence_tensor(seq_bits, seq_max_len + 1)

    # Create the dataset
    dataset = WaveformDatasetPreload(
        directory=data_directory,
        t_input=t_input,
        max_len=max_len,
        terminal_pad=target_pad,
        seq_vocab_len=seq_vocab_len,
        seq_max_len=seq_max_len,
        seq_t=seq_t_input
    )

    num_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    consecutive_size = config['train']['consecutive_size']
    consec_loss_order = config['train']['order']
    bce_weight = config['train']['bce_weight']
    cdif_weight = config['train']['cdif_weight']
    cdif_batch_weight = config['train']['cdif_batch_weight']

    sampler = RandomConsecutiveSampler(dataset, batch_size, consecutive_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=config['train']['num_workers'])

    # Initialize model, optimizer, and loss functions
    model = create_model(config['model'])
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()
    cdifb_loss = ConsecutiveDifferenceHigherOrderLossBatch(consecutive_size, order=consec_loss_order)
    cdif_loss = ConsecutiveDifferenceHigherOrderLoss(consecutive_size, order=consec_loss_order)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        first_20_losses = []
        last_20_losses = []

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
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
            total_loss = mse_loss + bce_weight * bce_loss + cdif_weight * cdif + cdif_batch_weight * cdif_b

            if i < 20:
                first_20_losses.append(total_loss.item())
            if i >= len(dataloader) - 20:
                last_20_losses.append(total_loss.item())


            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_first_20_loss = sum(first_20_losses) / len(first_20_losses) if first_20_losses else 0
        avg_last_20_loss = sum(last_20_losses) / len(last_20_losses) if last_20_losses else 0

        # Print progress for each epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} MSE: {mse_loss.item():.6f} BCE: {bce_loss.item():.6f} CDIF: {cdif.item():.6f} CDIF_B: {cdif_b.item():.6f} Total Loss: {total_loss.item():.8f}")

        # Log the averages to the log file
        with open(os.path.join(output_dir, "training_log.txt"), "a") as log_file:
            log_file.write(
                f"Epoch {epoch + 1}: Avg First 20 Loss: {avg_first_20_loss:.6f}, Avg Last 20 Loss: {avg_last_20_loss:.6f}\n")

        # Save model checkpoint in the experiment directory
        torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("Training complete!")


if __name__ == "__main__":
    main()
