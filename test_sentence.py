import torch
import json
from models.lstm import LSTMClassifier
from src.ml.utils.text import SimpleVocab, simple_tokenize

# Load model and vocabulary
vocab = SimpleVocab(**json.load(open('artifacts/vocab.json')))
model = LSTMClassifier(embeddings=torch.load('artifacts/embeddings.pt', map_location='cpu'))
checkpoint = torch.load('artifacts/best_model.ckpt', map_location='cpu')
model_state = checkpoint['state_dict']
cleaned_state = {key[6:] if key.startswith('model.') else key: value for key, value in model_state.items()}
model.load_state_dict(cleaned_state)
model.eval()


text = "Im going to tell elon i like him"

# Process and predict
tokens = simple_tokenize(text)
indices = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
seq = torch.tensor([indices + [vocab.pad_index] * (max(len(indices), 1) - len(indices))], dtype=torch.long)
lengths = torch.tensor([len(indices)])
mask = torch.ones_like(seq, dtype=torch.float32)
prob = torch.sigmoid(model((seq, mask, lengths)).squeeze()).item()

print(f"'{text}' -> Sentiment: {'Positive' if prob > 0.5 else 'Negative'} ({prob:.3f})") 