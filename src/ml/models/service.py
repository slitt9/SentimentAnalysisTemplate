import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from models.lstm import LSTMClassifier
from src.config.config import AppConfig
from src.ml.utils.text import SimpleVocab, simple_tokenize, pad_sequences


class ModelService:
    def __init__(self, model: LSTMClassifier, vocab: SimpleVocab, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()

    @classmethod
    def initialize_from_artifacts(cls, config: AppConfig) -> "ModelService":
        """Initialize the service from saved artifacts."""
        print(f"Loading vocabulary from {config.vocab_file_path}")
        with open(config.vocab_file_path, 'r') as f:
            vocab_data = json.load(f)
        
        vocab = SimpleVocab(vocab_data['stoi'], vocab_data['itos'])
        print(f"Vocabulary loaded with {len(vocab)} words")
        
        print(f"Loading embeddings from {config.embeddings_file_path}")
        # Fix: Add weights_only=False for PyTorch 2.6+ compatibility
        embeddings = torch.load(config.embeddings_file_path, map_location='cpu', weights_only=False)
        print(f"Embeddings loaded with shape {embeddings.shape}")
        
        print(f"Loading model from {config.model_ckpt_path}")
        model = LSTMClassifier(embeddings=embeddings)
        # Fix: Add weights_only=False for PyTorch 2.6+ compatibility
        model.load_state_dict(torch.load(config.model_ckpt_path, map_location='cpu', weights_only=False))
        model.eval()
        print("Model loaded successfully")
        
        return cls(model, vocab)

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        tokens = simple_tokenize(text)
        indices = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens]
        if not indices:
            indices = [self.vocab.pad_index]
        lengths = torch.tensor([len(indices)], dtype=torch.long, device=self.device)
        padded, mask = pad_sequences([indices], pad_value=self.vocab.pad_index)
        seq = torch.tensor(padded, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)

        logits = self.model((seq, mask_tensor, lengths))
        prob = torch.sigmoid(logits.squeeze(1)).item()
        return float(prob) 