import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import accuracy_score
from src.utils import get_sst2_data, build_vocab, load_glove_matrix
from src.dataset import GloveSentimentDataset, collate_fn_glove
from src.model_mlp import MLP_Glove
from tqdm import tqdm

# --- Configuration ---
GLOVE_PATH = 'data/glove.6B.300d.txt'
EMBED_DIM = 300
HIDDEN_DIM = 64
OUTPUT_DIM = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10
MIN_FREQ = 2 # Min word frequency to be included in vocab

def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for labels, text, offsets in tqdm(loader, desc="Training"):
        labels, text, offsets = labels.to(device), text.to(device), offsets.to(device)
        
        optimizer.zero_grad()
        outputs = model(text, offsets)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for labels, text, offsets in tqdm(loader, desc="Evaluating"):
            labels, text, offsets = labels.to(device), text.to(device), offsets.to(device)
            
            outputs = model(text, offsets)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(loader), accuracy

def main():
    print("--- Baseline 2: GloVe + MLP (EmbeddingBag) ---")
    
    # 1. Load Data
    print("Loading SST-2 data...")
    train_data, val_data = get_sst2_data()
    
    # 2. Build Vocabulary
    print(f"Building vocab (min_freq={MIN_FREQ})...")
    train_texts = [d["text"] for d in train_data]
    vocab = build_vocab(train_texts, min_freq=MIN_FREQ)
    print(f"Vocab size: {len(vocab)}")
    
    # 3. Create Datasets
    train_dataset = GloveSentimentDataset(train_data, vocab)
    val_dataset = GloveSentimentDataset(val_data, vocab)
    
    # 4. Create DataLoaders
    # We use a partial to pass the pad_idx to the collate_fn
    collate_batch = partial(collate_fn_glove, pad_idx=vocab.pad_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # 5. Load Embeddings
    embedding_matrix = load_glove_matrix(GLOVE_PATH, vocab, embed_dim=EMBED_DIM)

    # 6. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MLP_Glove(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    # We only train the classifier layers, not the frozen embeddings
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 7. Training Loop
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02}/{NUM_EPOCHS}:")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal. Loss:  {val_loss:.4f} | Val. Acc: {val_acc*100:.2f}%")

    print("Training complete.")

if __name__ == "__main__":
    main()