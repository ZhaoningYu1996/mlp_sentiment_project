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
import wandb
import os

# --- Configuration ---
GLOVE_PATH = 'data/glove.6B.300d.txt'
EMBED_DIM = 300
HIDDEN_DIM = 64
OUTPUT_DIM = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10
MIN_FREQ = 2

# --- Paths ---
MODEL_SAVE_DIR = "models"
VOCAB_PATH = os.path.join(MODEL_SAVE_DIR, "glove_vocab.pt")
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "glove_best_model.pt")

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
    # --- 1. Init Wandb ---
    config = {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "min_freq": MIN_FREQ,
        "model_type": "glove_mlp"
    }
    wandb.init(
        project="mlp-sentiment-baselines", 
        name="baseline-glove", 
        config=config
    )
    
    # --- 2. Setup ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    print("Loading SST-2 data...")
    train_data, val_data = get_sst2_data()
    
    # 4. Build Vocabulary
    print(f"Building vocab (min_freq={MIN_FREQ})...")
    train_texts = [d["text"] for d in train_data]
    vocab = build_vocab(train_texts, min_freq=MIN_FREQ)
    print(f"Vocab size: {len(vocab)}")
    print(f"Saving vocab to {VOCAB_PATH}")
    torch.save(vocab, VOCAB_PATH)
    
    # 5. Create Datasets
    train_dataset = GloveSentimentDataset(train_data, vocab)
    val_dataset = GloveSentimentDataset(val_data, vocab)
    
    collate_batch = partial(collate_fn_glove, pad_idx=vocab.pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # 6. Load Embeddings
    embedding_matrix = load_glove_matrix(GLOVE_PATH, vocab, embed_dim=EMBED_DIM)
    wandb.config.update({"vocab_size": len(vocab)})

    # 7. Initialize Model
    model = MLP_Glove(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    wandb.watch(model, log="all")

    # 8. Training Loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02}/{NUM_EPOCHS}:")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal. Loss:  {val_loss:.4f} | Val. Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"\tNew best model! Saving to {MODEL_PATH}")
            torch.save(model.state_dict(), MODEL_PATH)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc
        })

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()