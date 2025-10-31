import torch
import torch.nn as nn

class MLP_TFIDF(nn.Module):
    """MLP for TF-IDF features."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_TFIDF, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        return x

class MLP_Glove(nn.Module):
    """
    MLP for GloVe features, using nn.EmbeddingBag.
    EmbeddingBag is a simple and efficient way to average embeddings.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(MLP_Glove, self).__init__()
        
        # EmbeddingBag layer
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim, 
            mode='mean',  # 'mean', 'sum', or 'max'
            sparse=False
        )
        
        # Load pre-trained weights
        self.embedding.weight.data.copy_(pretrained_embeddings)
        # We freeze the embedding layer during training
        self.embedding.weight.requires_grad = False
        
        # Classifier layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_indices, offsets):
        # text_indices: [total_length_of_batch]
        # offsets: [batch_size]
        embedded = self.embedding(text_indices, offsets)
        
        # embedded shape is [batch_size, embed_dim]
        x = self.fc1(embedded)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x