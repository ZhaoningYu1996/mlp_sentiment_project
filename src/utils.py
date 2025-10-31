import torch
import numpy as np
from datasets import load_dataset
from collections import Counter, OrderedDict
from tqdm import tqdm

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def get_sst2_data():
    """Loads the SST-2 dataset from Hugging Face datasets."""
    dataset = load_dataset("stanfordnlp/sst2")

    train_data = [
        {"text": example["sentence"], "label": example["label"]}
        for example in dataset["train"]
    ]
    val_data = [
        {"text": example["sentence"], "label": example["label"]}
        for example in dataset["validation"]
    ]
    return train_data, val_data

def build_vocab(text_iter, min_freq=2):
    """
    Builds a vocabulary from an iterator of texts.
    Returns a 'vocab' object with word2idx and idx2word mappings.
    """
    counter = Counter()
    for text in text_iter:
        counter.update(text.lower().split())

    # Create an ordered dictionary of tokens, ordered by frequency
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # Filter by min_freq
    word2idx = {}
    for word, freq in ordered_dict.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    # Add special tokens
    word2idx[PAD_TOKEN] = len(word2idx)
    word2idx[UNK_TOKEN] = len(word2idx)
    
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Create a simple vocab object
    class Vocab:
        def __init__(self, w2i, i2w):
            self.word2idx = w2i
            self.idx2word = i2w
            self.pad_idx = w2i[PAD_TOKEN]
            self.unk_idx = w2i[UNK_TOKEN]

        def __len__(self):
            return len(self.word2idx)
            
        def text_to_indices(self, text):
            tokens = text.lower().split()
            return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    return Vocab(word2idx, idx2word)

def load_glove_matrix(glove_path, vocab, embed_dim=300):
    """
    Loads GloVe embeddings and creates a weight matrix
    matching the given vocabulary.
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    # Initialize matrix with zeros
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    words_found = 0

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing GloVe"):
            parts = line.split()
            word = parts[0]
            
            if word in vocab.word2idx:
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    embedding_matrix[vocab.word2idx[word]] = vector
                    words_found += 1
                except ValueError:
                    pass
    
    print(f"Loaded {words_found} pre-trained vectors for {vocab_size} vocab words.")
    return torch.from_numpy(embedding_matrix)