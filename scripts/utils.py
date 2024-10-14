import os
import torch
from scipy.io import wavfile


def get_max_required_length(dir):
    max_length = 0

    # Iterate over all files in the directory
    for filename in os.listdir(dir):
        if filename.endswith('.wav'):  # Only process .wav files
            file_path = os.path.join(dir, filename)

            # Read the .wav file
            sample_rate, data = wavfile.read(file_path)

            # Get the length of the audio file (number of samples)
            file_length = data.shape[0]  # shape[0] gives the number of samples (time dimension)

            # Update the max length if this file is longer
            if file_length > max_length:
                max_length = file_length

    return max_length


# import torch
# import torch.nn as nn
# import torch.optim as optim


def binary_sequence_tensor(num_bits, length):
    # Create a tensor of shape (length,) with values from 0 to length - 1
    t_values = torch.arange(1, length + 1)  # start with 1

    # Create a tensor to store the binary representations
    binary_tensor = ((t_values.unsqueeze(1) >> torch.arange(num_bits)) & 1).float()
    binary_tensor[binary_tensor == 0] = -1
    return binary_tensor


def generate_sine_tensor(num_bits, length):
    # Create an array of integers from 0 to length - 1
    t = np.arange(length)
    # Generate the sine waves for each bit
    sine_tensor = np.zeros((length, num_bits))  # Initialize the tensor

    for i in range(num_bits):
        frequency = (np.pi / (2 ** i))  # Calculate frequency based on the number of bits
        sine_tensor[:, i] = np.sin(frequency * (t + 0.5))  # Fill the tensor with sine values

    return sine_tensor