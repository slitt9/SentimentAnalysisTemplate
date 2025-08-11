import os
from dataclasses import dataclass
from pathlib import Path


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    artifacts_dir: str = "artifacts"
    model_ckpt: str = "best_model.ckpt"
    vocab_file: str = "vocab.json"
    embeddings_file: str = "embeddings.pt"
    use_cuda: bool = False

    @property
    def model_ckpt_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.model_ckpt)

    @property
    def vocab_file_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.vocab_file)

    @property
    def embeddings_file_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.embeddings_file)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls() 