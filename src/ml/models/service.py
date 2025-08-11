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
        """Initialize the model service from saved artifacts."""
        try:
            # Load vocabulary FIRST
            vocab_path = os.path.join(config.artifacts_dir, config.vocab_file)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            vocab = SimpleVocab.from_dict(vocab_data)
            
            # Load embeddings
            embeddings_path = os.path.join(config.artifacts_dir, config.embeddings_file)
            embeddings = torch.load(embeddings_path, map_location='cpu')
            
            # Create model with EXACT same vocabulary size
            model = LSTMClassifier(
                embeddings=embeddings,
                lstm_hidden_size=512,
                lstm_num_layers=3,
                mlp_hidden_sizes=[512, 256, 128],
                dropout=0.3,
            )
            
            # Load the trained weights
            model_path = os.path.join(config.artifacts_dir, config.model_ckpt)
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state)
            
            return cls(model, vocab)
            
        except Exception as e:
            raise

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