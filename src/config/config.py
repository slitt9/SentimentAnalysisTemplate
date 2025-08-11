import os
from dataclasses import dataclass
from typing import Optional


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    artifacts_dir: str
    model_ckpt: str
    vocab_file: str
    embeddings_file: str
    use_cuda: bool = False
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "./artifacts"),
            model_ckpt=os.getenv("MODEL_CKPT", "best_model.ckpt"),
            vocab_file=os.getenv("VOCAB_FILE", "vocab.json"),
            embeddings_file=os.getenv("EMBEDDINGS_FILE", "embeddings.pt"),
            use_cuda=os.getenv("USE_CUDA", "false").lower() == "true"
        )

    @property
    def model_ckpt_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.model_ckpt)

    @property
    def vocab_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.vocab_file)

    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.embeddings_file) 