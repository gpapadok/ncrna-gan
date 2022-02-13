#!/usr/bin/env python
# coding: utf-8

import sys

import torch


def _one_hot_to_base(oh):
    oh_to_base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    base = oh_to_base_dict[oh.argmax().item()]
    return base


def _trim_sequence(seq):
    if seq[-1] == 'N':
        j = -1
    else:
        return seq

    try:
        while seq[j-1] == 'N':
            j -= 1
    except IndexError:
        pass

    return seq[:j]


def _save_to_fasta(sequences, labels, name):
    with open(name, 'w') as fasta:
        for j, seq in enumerate(sequences):
            fasta.write(
                f">{j} artificial hsa ncrna sequence | {str(labels[j])}\n")
            fasta.write(seq + "\n")

    return


def _decode_sequence(one_hot_seq):
    seq = ''.join([_one_hot_to_base(oh_base)
                   for oh_base in one_hot_seq.transpose(0, 1)])
    return seq


def _decode_sequences(sequences):
    seqs = [_trim_sequence(_decode_sequence(seq))
            for seq in sequences]
    return seqs


def _generate_oh_sequences(generator, labels):
    noise = torch.randn(len(labels), generator.latent_size)
    if torch.cuda.is_available():
        noise = noise.cuda()
    with torch.no_grad():
        oh_seqs = generator(noise, labels).detach()
    return oh_seqs


def create_fake_fasta(generator, fasta, labels):
    # generate sequences
    oh_seqs = _generate_oh_sequences(generator, labels)
    # decode to string
    decoded_sequences = _decode_sequences(oh_seqs)
    # save to fasta
    _save_to_fasta(decoded_sequences, labels, fasta)
    return


if __name__ == '__main__':
    from models import Generator
    gen_wfile = sys.argv[1]
    out_file = sys.argv[2]
    gen = Generator(n_chars=5, seq_len=200, residual=.1)
    if torch.cuda.is_available():
        gen.cuda()
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    gen.load_state_dict(torch.load(gen_wfile, map_location=dev))
    create_fake_fasta(gen, out_file, n=2000)
