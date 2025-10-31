import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from src.utils import get_sst2_data
from src.model_mlp import MLP_TFIDF
from tqdm import tqdm

# --- Configuration ---
VOCAB_SIZE = 10000  # Max features for TF-IDF
HIDDEN_DIM = 128
OUTPUT_DIM = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10

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
    print("--- Baseline 1: TF-IDF + MLP ---")
    
    # 1. Load Data
    print("Loading SST-2 data...")
    train_data, val_data = get_sst2_data()
    
    train_texts = [d["text"] for d in train_data]
    train_labels = [d["label"] for d in train_data]
    val_texts = [d["text"] for d in val_data]
    val_labels = [d["label"] for d in val_data]
    
    # 2. Vectorize Text
    print(f"Vectorizing text with TF-IDF (max_features={VOCAB_SIZE})...")
    vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE)
    
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()
    
    # 3. Create PyTorch DataLoaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(val_labels, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MLP_TFIDF(input_dim=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
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