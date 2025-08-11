import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

def clean_tweet_text(text: str) -> str:
    """Clean tweet text for better sentiment analysis."""
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Keep only alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
    
    return text.strip()

def simple_tokenize(text: str) -> List[str]:
    """Better tokenization with cleaned text."""
    text = clean_tweet_text(text)
    tokens = text.split()
    # Filter out very short tokens
    tokens = [token for token in tokens if len(token) > 1]
    return tokens

@dataclass
class SimpleVocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad_index: int = 0
    unk_index: int = 1

    @staticmethod
    def build_from_texts(texts: List[str], min_freq: int = 1) -> "SimpleVocab":
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(simple_tokenize(t))
        itos = {0: "<pad>", 1: "<unk>"}
        stoi = {"<pad>": 0, "<unk>": 1}
        for token, freq in counter.items():
            if freq >= min_freq and token not in stoi:
                idx = len(itos)
                itos[idx] = token
                stoi[token] = idx
        return SimpleVocab(stoi=stoi, itos=itos)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}, "pad_index": self.pad_index, "unk_index": self.unk_index}

    @staticmethod
    def from_dict(data: Dict) -> "SimpleVocab":
        itos = {int(k): v for k, v in data["itos"].items()}
        return SimpleVocab(stoi=data["stoi"], itos=itos, pad_index=data.get("pad_index", 0), unk_index=data.get("unk_index", 1))

def pad_sequences(sequences: List[List[int]], pad_value: int = 0) -> Tuple[List[List[int]], List[List[int]]]:
    max_len = max(len(seq) for seq in sequences)
    padded = []
    masks = []
    for seq in sequences:
        padding = [pad_value] * (max_len - len(seq))
        padded_seq = seq + padding
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        padded.append(padded_seq)
        masks.append(mask)
    return padded, masks 