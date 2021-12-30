import smtplib, ssl, os

import numpy as np
import matplotlib.pyplot as plt

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

### functions to plot
def smooth(values, p=.9):
    # smooth curves
    t = [0]
    for v in values:
        t += [p*t[-1] + (1-p)*v]
    return t

def plot_sequence(sequence):
    plt.figure(figsize=(21, 3))
    plt.plot(sequence[0,:])
    plt.plot(sequence[1,:])
    plt.plot(sequence[2,:])
    plt.plot(sequence[3,:])
    plt.legend(['A', 'C', 'G', 'T'])
    plt.xticks(range(SAMPLE_SIZE))
    plt.grid(axis='x')
    plt.show()

