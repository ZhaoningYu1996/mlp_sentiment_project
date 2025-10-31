import torch
from torch.utils.data import Dataset

class GloveSentimentDataset(Dataset):
    """PyTorch Dataset for GloVe model."""
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]
        
        # Convert text to indices
        indices = self.vocab.text_to_indices(text)
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn_glove(batch, pad_idx):
    """
    Custom collate_fn for the GloVe DataLoader.
    This is required for nn.EmbeddingBag.
    """
    labels = []
    texts = []
    
    for (text_indices, label) in batch:
        labels.append(label)
        texts.append(text_indices) # Keep as list of tensors for now

    # Create offsets for EmbeddingBag
    # Offsets are the starting indices of each sequence in the concatenated tensor
    offsets = [0] + [len(t) for t in texts[:-1]]
    offsets = torch.tensor(offsets, dtype=torch.long).cumsum(dim=0)
    
    # Concatenate all text indices into a single tensor
    texts_tensor = torch.cat(texts)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return labels_tensor, texts_tensor, offsets