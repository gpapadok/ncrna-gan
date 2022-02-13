import numpy as np


def encode_sequence(sequence):
    # encode sequence from string to one-hot
    base_to_one_hot = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 'G': [0,0,1,0,0],
                       'T': [0,0,0,1,0], 'N': [0,0,0,0,1]}
    encoded = []
    for base in sequence:
        encoded.append(base_to_one_hot[base])
    return encoded


def pad_sequence(seq, max_length):
    # pad with zeros
    padded = np.zeros((max_length,5))
    padded[:len(seq)] = seq
    padded[len(seq):,4] = 1
    return padded
