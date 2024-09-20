import os
import argparse
import yaml
import torch
from datetime import datetime
from models import create_model
from scripts.utils import get_max_required_length, binary_sequence_tensor
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

def tensor_to_wav(tensor, experiment_name, output_dir, digits, current_date, sample_rate=24000, cut_off=-1):
    # Convert tensor to numpy array and detach if needed
    data = tensor.detach().cpu().numpy()[:cut_off]
    # Normalize to the range [-1, 1]
    #data = data / np.max(np.abs(data))  # Uncomment if you need normalization

    # Convert to 16-bit PCM format (values between -32768 and 32767)
    data_int16 = np.int16(data * 32768)

    # Write the .wav file

    wav_file_name = f"{experiment_name}_{digits}_pred_wav_{current_date}.wav"
    wav_file_path = os.path.join(output_dir, wav_file_name)

    write(wav_file_path, sample_rate, data_int16)
    print(f"Saved as {wav_file_name}")

def tensor_to_graph(bce_outputs,mse_outputs,experiment_name,digits,output_dir,current_date):
    # Get current date for unique file name
    plt.figure(figsize=(20, 5))

    # Plot BCE outputs
    plt.subplot(1, 2, 1)
    plt.plot(bce_outputs.cpu().numpy(), label='BCE Output', color='blue')
    plt.title('BCE Outputs')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # Plot MSE outputs
    plt.subplot(1, 2, 2)
    plt.plot(mse_outputs.cpu().numpy(), label='MSE Output', color='orange')
    plt.title('MSE Outputs')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # Save the plot as an image in the output directory
    image_file_name = f"{experiment_name}_{digits}_pred_plot_{current_date}.png"
    image_file_path = os.path.join(output_dir, image_file_name)

    plt.tight_layout()  # Adjust layout to make room for titles and labels
    plt.savefig(image_file_path)
    plt.close()

    print(f"Prediction plots saved to {image_file_path}")

def main():
    # Parse command-line arguments
    """
    to retrieve preditctions from the output folder for the experiment:
    predictions = torch.load(file_path)
    bce_outputs = predictions['bce_outputs']
    mse_outputs = predictions['mse_outputs']


    """
    parser = argparse.ArgumentParser(description='Validate the model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    parser.add_argument('--digits', type=str, required=True, help='digits to generate')

    args = parser.parse_args()
    digits = args.digits
    config_path = args.config

    #clean up config path, for convenience and so i don't have to remember
    if config_path.startswith("/config/"):
        config_path = config_path.lstrip('/')  # Remove the leading '/'
    # Otherwise, if it doesn't start with 'configs/', prepend 'configs/'
    elif not config_path.startswith("config/"):
        config_path = os.path.join("config", config_path)

    print(f"Using configuration file: {config_path}")
    print(f"Generating predictions for digits: {digits}")

    # Load configuration (assuming your config is a YAML file)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experiment directory
    experiment_name = config['experiment_name']
    output_dir = os.path.join("outputs", experiment_name)

    # Create dataloader
    data_directory = config['data']['data_directory']
    max_len = get_max_required_length(data_directory)
    target_pad = config['data']['target_pad']
    bits = config['data']['bits']
    seq_bits = config['data']['seq_bits']
    seq_max_len = config['data']['seq_max_len']
    seq_vocab_len = config['data']['seq_vocab_len']

    t_input = binary_sequence_tensor(bits, max_len + target_pad)
    seq_t_input = binary_sequence_tensor(seq_bits, seq_max_len )


    eval_batch_size = config['train']['batch_size'] #just reuse it from train.  this will be smalles since no consec so, no overflow.


    # Initialize model
    model = create_model(config['model']).to(device)

    # Load model checkpoint

    checkpoint_path = os.path.join(output_dir, f"{experiment_name}_latest.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set model to evaluation mode

    #make inputs
    file_name_integers = [int(char) for char in digits]
    # Pad the list with zeros until it matches the required length
    padded_file_name = file_name_integers + [seq_vocab_len] * (seq_vocab_len - len(file_name_integers))
    seq_t_input[len(file_name_integers):] = 0

    rep = t_input.shape[0]
    seq_t_input = seq_t_input.unsqueeze(0).repeat(rep, 1, 1).to(device)
    file_indexes = torch.tensor(padded_file_name).to(device)
    file_indexes = file_indexes.unsqueeze(0).repeat(rep, 1)

    print(file_indexes.shape, seq_t_input.shape, t_input.shape)

    bce_outputs = []
    mse_outputs = []

    # Get the total number of batches
    total_batches = (rep + eval_batch_size - 1) // eval_batch_size

    # Loop over batches
    for i in range(total_batches):
        # Define the start and end of the batch
        start_idx = i * eval_batch_size
        end_idx = min((i + 1) * eval_batch_size, rep)

        # Slice the batch from each input
        batch_file = file_indexes[start_idx:end_idx].to(device)
        batch_input_seq_eval = seq_t_input[start_idx:end_idx].to(device)
        batch_t_input = t_input[start_idx:end_idx].to(device)

        # Run the model in evaluation mode (assuming the model is in eval mode already)
        with torch.no_grad():  # Disable gradient calculation for evaluation
            bce_output, mse_output = model(batch_input_seq_eval, batch_file, batch_t_input)

        # Append the outputs
        bce_outputs.append(bce_output)
        mse_outputs.append(mse_output)

    # Optionally, concatenate the outputs into single tensors
    bce_outputs = torch.cat(bce_outputs, dim=0)
    mse_outputs = torch.cat(mse_outputs, dim=0)

    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{experiment_name}_{digits}_pred_{current_date}.pt"

    # Full path for saving
    file_path = os.path.join(output_dir, file_name)

    # Save the outputs as a dictionary
    torch.save({
        'bce_outputs': bce_outputs,
        'mse_outputs': mse_outputs
    }, file_path)

    print(f"Predictions saved to {file_path}")

    tensor_to_graph(bce_outputs, mse_outputs, experiment_name, digits, output_dir, current_date)
    tensor_to_wav(mse_outputs, experiment_name, output_dir, digits, current_date)

if __name__ == "__main__":
    main()
