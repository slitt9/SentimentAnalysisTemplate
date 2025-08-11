import json
import torch
import pandas as pd
from tqdm import tqdm

from models.lstm import LSTMClassifier
from src.ml.utils.text import SimpleVocab, simple_tokenize, pad_sequences


def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    print("Loading model and vocabulary...")
    
    # Load vocabulary
    with open('artifacts/vocab.json', 'r') as f:
        vocab_data = json.load(f)
    vocab = SimpleVocab(vocab_data['stoi'], vocab_data['itos'])
    print(f"Vocabulary loaded: {len(vocab.stoi)} words")
    
    # Load embeddings
    embeddings = torch.load('artifacts/embeddings.pt', map_location='cpu')
    print(f"Embeddings loaded: {embeddings.shape}")
    
    # Load model
    model = LSTMClassifier(embeddings=embeddings)
    
    # Load checkpoint and extract just the model state
    checkpoint = torch.load('artifacts/best_model.ckpt', map_location='cpu')
    if 'state_dict' in checkpoint:
        # Extract model weights from PyTorch Lightning checkpoint
        model_state = checkpoint['state_dict']
        # Remove 'model.' prefix if it exists
        cleaned_state = {}
        for key, value in model_state.items():
            if key.startswith('model.'):
                cleaned_state[key[6:]] = value
            else:
                cleaned_state[key] = value
        model.load_state_dict(cleaned_state)
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    return model, vocab


def test_on_sentiment140(model, vocab, test_samples=1000):
    """Test the model on Sentiment140 test data"""
    print(f"\nTesting on {test_samples} Sentiment140 samples...")
    
    # Load test data (use different portion than training)
    try:
        df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'],
                         encoding='latin-1')
        
        # Use last 1000 samples as test set (different from training)
        test_df = df.tail(test_samples)
        print(f"Loaded {len(test_df)} test samples")
        
    except FileNotFoundError:
        print("Sentiment140 dataset not found!")
        return
    
    # Test the model
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
            text = row['text']
            true_label = row['target']
            
            # Tokenize and predict
            tokens = simple_tokenize(text)
            indices = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
            
            if not indices:
                indices = [vocab.pad_index]
            
            # Pad sequence
            max_len = max(len(indices), 1)
            padded = indices + [vocab.pad_index] * (max_len - len(indices))
            
            # Convert to tensor
            seq = torch.tensor([padded], dtype=torch.long)
            lengths = torch.tensor([len(indices)])
            mask = torch.ones_like(seq, dtype=torch.float32)
            
            # Predict
            output = model((seq, mask, lengths))
            prob = torch.sigmoid(output.squeeze()).item()
            predicted_label = 1 if prob > 0.5 else 0
            
            # Compare with true label
            if predicted_label == true_label:
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"\nTest Results:")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def test_custom_sentences(model, vocab):
    """Test on custom sentences"""
    print("\nTesting on custom sentences...")
    
    test_sentences = [
        ("I love this movie! It was amazing!", 1),
        ("This is terrible, I hate it!", 0),
        ("Great experience, highly recommend!", 1),
        ("Awful service, never coming back!", 0),
        ("The food was delicious and the staff was friendly", 1),
        ("Worst decision ever, complete waste of money", 0),
        ("Neutral experience, nothing special", 0),  # This might be tricky
        ("Absolutely fantastic, exceeded all expectations!", 1)
    ]
    
    correct = 0
    total = len(test_sentences)
    
    with torch.no_grad():
        for text, true_label in test_sentences:
            # Tokenize and predict
            tokens = simple_tokenize(text)
            indices = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
            
            if not indices:
                indices = [vocab.pad_index]
            
            # Pad sequence
            max_len = max(len(indices), 1)
            padded = indices + [vocab.pad_index] * (max_len - len(indices))
            
            # Convert to tensor
            seq = torch.tensor([padded], dtype=torch.long)
            lengths = torch.tensor([len(indices)])
            mask = torch.ones_like(seq, dtype=torch.float32)
            
            # Predict
            output = model((seq, mask, lengths))
            prob = torch.sigmoid(output.squeeze()).item()
            predicted_label = 1 if prob > 0.5 else 0
            
            # Show results
            status = "CORRECT" if predicted_label == true_label else "WRONG"
            print(f"{status} '{text}' -> Predicted: {predicted_label} ({prob:.3f}), True: {true_label}")
            
            if predicted_label == true_label:
                correct += 1
    
    accuracy = correct / total
    print(f"\nCustom Test Results:")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


if __name__ == "__main__":
    # Load model and vocab
    model, vocab = load_model_and_vocab()
    
    # Test on Sentiment140 test data
    sentiment140_acc = test_on_sentiment140(model, vocab, test_samples=1000)
    
    # Test on custom sentences
    custom_acc = test_custom_sentences(model, vocab)
    
    print(f"\nOverall Test Summary:")
    print(f"Sentiment140 Test Accuracy: {sentiment140_acc:.4f} ({sentiment140_acc*100:.2f}%)")
    print(f"Custom Sentences Accuracy: {custom_acc:.4f} ({custom_acc*100:.2f}%)")