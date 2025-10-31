import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from src.utils import get_sst2_data
from src.model_mlp import MLP_TFIDF
from tqdm import tqdm
import wandb
import joblib
import os

# --- Configuration ---
VOCAB_SIZE = 10000
HIDDEN_DIM = 128
OUTPUT_DIM = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100

# --- Paths ---
MODEL_SAVE_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_SAVE_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "tfidf_best_model.pt")

def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for features, labels in tqdm(loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
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
        for features, labels in tqdm(loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
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
        "vocab_size": VOCAB_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "model_type": "tfidf_mlp"
    }
    wandb.init(
        project="mlp-sentiment-baselines", 
        name="baseline-tfidf", 
        config=config
    )
    
    # --- 2. Setup ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Load Data
    print("Loading SST-2 data...")
    train_data, val_data = get_sst2_data()
    
    train_texts = [d["text"] for d in train_data]
    train_labels = [d["label"] for d in train_data]
    val_texts = [d["text"] for d in val_data]
    val_labels = [d["label"] for d in val_data]
    
    # 4. Vectorize Text
    print(f"Vectorizing text with TF-IDF (max_features={VOCAB_SIZE})...")
    vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE)
    
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()
    
    print(f"Saving vectorizer to {VECTORIZER_PATH}")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    # 5. Create PyTorch DataLoaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(val_labels, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Initialize Model
    model = MLP_TFIDF(input_dim=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    wandb.watch(model, log="all")

    # 7. Training Loop
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