import json
import logging
import os

import torch
from torchtext.data import get_tokenizer
from tqdm import tqdm


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        # splits text into tokens
        self.tokenizer = get_tokenizer('basic_english')
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

    def fit_on_texts_and_embeddings(self, sentences: list[str], embeddings):
        # Add special tokens to the start of the vocab
        for special_token in self.special_tokens:
            self.vocab[special_token] = len(self.vocab)

        # Add each unique word in the sentences to the vocab if there is also an embedding for it
        for sentence in tqdm(sentences):
            for word in self.tokenizer(sentence):
                if word not in self.vocab and word in embeddings.stoi:
                    self.vocab[word] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # create an embbeding matrix from a set of pretrained embeddings based on the vocab
    def get_embeddings_matrix(self, embeddings):
        # Create a matrix of zeroes of the shape of the vocab size
        embeddings_matrix = torch.zeros((len(self.vocab), embeddings.dim))

        # For each word in the vocab get its index and add its embedding to the matrix
        for word, idx in self.vocab.items():
            # ignore special tokens
            if word in self.special_tokens:
                continue
            # if the word is in the embeddings, add its embedding to the matrix
            if word in embeddings.stoi:
                embeddings_matrix[idx] = embeddings[word]
            else:
                raise KeyError(f"Word {word} not in embeddings. Please create tokenizer based on embeddings")

        # Initialize the <pad> token with the mean of the embeddings of the vocab
        embeddings_matrix[1] = torch.mean(embeddings_matrix[len(self.special_tokens):], dim=0)

        # Initialize the <sos> and <eos> tokens with the mean of the embeddings of the vocab
        # plus or minus a small amount of noise to avoid them matching the <unk> token
        # and avoiding having identical embeddings which the model can not distinguish
        noise = torch.normal(mean=0, std=0.1, size=(embeddings.dim,))
        embeddings_matrix[2] = torch.mean(embeddings_matrix[len(self.special_tokens):] + noise, dim=0)
        embeddings_matrix[3] = torch.mean(embeddings_matrix[len(self.special_tokens):] - noise, dim=0)

        return embeddings_matrix

    # add start of sentence and end of sentence tokens to the tokenizer sentence
    def add_special_tokens(self, tokens):
        return ["<sos>"] + tokens + ["<eos>"]

    # convert a sequence of words to a sequence of indices based on the vocab
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def pad_sequences(self, sequences, max_length=None):
        # Pads the vectorized sequences

        # If max_length is not specified, pad to the length of the longest sequence
        if not max_length:
            max_length = max(len(seq) for seq in sequences)

        # Create a tensor for the lengths of the sequences
        sequence_lengths = torch.LongTensor([min(len(seq), max_length) for seq in sequences])

        # Create a tensor for the sequences with zeros
        seq_tensor = torch.zeros((len(sequences), max_length)).long()

        # Create a tensor for the masks with zeros
        seq_mask = torch.zeros((len(sequences), max_length)).long()

        # For each sequence add the values to the seq_tensor
        #  and add 1s to the seq_mask according to its length
        for idx, (seq, seq_len) in enumerate(zip(sequences, sequence_lengths)):
            # truncate the sequence if it exceeds the max length
            seq = seq[:seq_len]

            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
            seq_mask[idx, :seq_len] = torch.LongTensor([1])

        return seq_tensor, seq_mask, sequence_lengths

    # split the text into tokens
    def tokenize(self, text):
        return self.tokenizer(text)

    def save(self, output_dir):
        import os
        import pickle
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, input_dir):
        import pickle
        with open(os.path.join(input_dir, 'tokenizer.pkl'), 'rb') as f:
            return pickle.load(f)

    def encode(self, texts, max_length=None):
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for text in texts:
            tokens = self.tokenize(text)
            tokens = self.add_special_tokens(tokens)
            ids = self.convert_tokens_to_ids(tokens)
            sequences.append(ids)

        seq_tensor, seq_mask, sequence_lengths = self.pad_sequences(sequences, max_length)

        return seq_tensor, seq_mask, sequence_lengths
