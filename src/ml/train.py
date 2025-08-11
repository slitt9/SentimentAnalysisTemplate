import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

from models.lstm import LSTMClassifier
from utils.text import SimpleVocab, simple_tokenize, pad_sequences


class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.tweets = list(data.text)
        self.labels = list(data.target)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return {'text': self.tweets[index], 'label': self.labels[index]}


def create_vocab(texts, min_freq=5):
    print("Building vocabulary...")
    word_freq = {}
    for text in tqdm(texts, desc="Processing texts"):
        tokens = simple_tokenize(text)
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    
    vocab_words = [word for word, freq in word_freq.items() if freq >= min_freq]
    vocab_words = ['<pad>', '<unk>'] + sorted(vocab_words)
    
    stoi = {word: idx for idx, word in enumerate(vocab_words)}
    itos = vocab_words
    
    return SimpleVocab(stoi, itos)


def collate_batch(batch, vocab):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Tokenize and convert to indices
    tokenized = [simple_tokenize(text) for text in texts]
    indices = [[vocab.stoi.get(token, vocab.unk_index) for token in tokens] for tokens in tokenized]
    
    # Pad sequences
    max_len = max(len(seq) for seq in indices)
    padded = []
    lengths = []
    
    for seq in indices:
        if len(seq) < max_len:
            seq = seq + [vocab.pad_index] * (max_len - len(seq))
        padded.append(seq)
        lengths.append(len(seq))
    
    return {
        'text': torch.tensor(padded, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.float32),
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


def train_model():
    print("Loading Sentiment140 dataset...")
    
    try:
        df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'],
                         encoding='latin-1')
        print(f"Loaded Sentiment140: {len(df)} samples")
    except FileNotFoundError:
        print("Sentiment140 dataset not found!")
        return
    
    df = df.head(100000)
    print(f"Using subset: {len(df)} samples")
    
    # Split into train/val
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create vocabulary
    vocab = create_vocab(train_df.text, min_freq=5)
    print(f"Vocabulary created: {len(vocab.stoi)} words")
    
    # Save vocabulary
    vocab_data = {'stoi': vocab.stoi, 'itos': vocab.itos}
    with open('artifacts/vocab.json', 'w') as f:
        json.dump(vocab_data, f)
    print("Saved vocab.json")
    
    # Create embeddings
    vocab_size = len(vocab.stoi)
    embedding_dim = 100
    embeddings = torch.randn(vocab_size, embedding_dim) * 0.01
    torch.save(embeddings, 'artifacts/embeddings.pt')
    print("Saved embeddings.pt")
    
    # Create model
    model = LSTMClassifier(embeddings=embeddings)
    print("Created LSTM model")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create datasets
    train_dataset = SentimentDataset(train_df)
    val_dataset = SentimentDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: collate_batch(x, vocab))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: collate_batch(x, vocab))
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            text = batch['text']
            labels = batch['labels']
            lengths = batch['lengths']
            
            # Forward pass
            outputs = model((text, torch.ones_like(text, dtype=torch.float32), lengths))
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                text = batch['text']
                labels = batch['labels']
                lengths = batch['lengths']
                
                outputs = model((text, torch.ones_like(text, dtype=torch.float32), lengths))
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'artifacts/best_model.ckpt')
            print(f"New best model saved with val acc: {val_acc:.4f}")
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    print("All artifacts created successfully!")


if __name__ == "__main__":
    train_model() 