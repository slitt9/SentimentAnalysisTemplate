import os
import json
import torch
import numpy as np
import sys

# Ensure project root is on sys.path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
VOCAB_FILE = os.getenv("VOCAB_FILE", "vocab.json")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "embeddings.pt")
MODEL_CKPT = os.getenv("MODEL_CKPT", "model.ckpt")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "50"))


def ensure_minimal_vocab():
    path = os.path.join(ARTIFACTS_DIR, VOCAB_FILE)
    if os.path.exists(path):
        return
    vocab = {
        "stoi": {"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3},
        "itos": {"0": "<pad>", "1": "<unk>", "2": "good", "3": "bad"},
        "pad_index": 0,
        "unk_index": 1,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)


def ensure_minimal_embeddings():
    path = os.path.join(ARTIFACTS_DIR, EMBEDDINGS_FILE)
    if os.path.exists(path):
        return
    # 4 tokens x embedding_dim
    weights = torch.randn(4, EMBEDDING_DIM) * 0.01
    torch.save(weights, path)


def ensure_minimal_model():
    path = os.path.join(ARTIFACTS_DIR, MODEL_CKPT)
    if os.path.exists(path):
        return
    # Create a tiny LSTM model state dict compatible with 4xEMBEDDING_DIM embeddings
    from models.lstm import LSTMClassifier
    embeddings = torch.load(os.path.join(ARTIFACTS_DIR, EMBEDDINGS_FILE), map_location="cpu")
    model = LSTMClassifier(embeddings=embeddings)
    torch.save(model.state_dict(), path)


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    ensure_minimal_vocab()
    ensure_minimal_embeddings()
    ensure_minimal_model()
    print(f"Bootstrapped minimal artifacts in {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main() 