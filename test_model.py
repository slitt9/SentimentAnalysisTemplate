from src.ml.models.service import ModelService
from src.config.config import AppConfig

config = AppConfig.from_env()
service = ModelService.initialize_from_artifacts(config)
print("Model loaded successfully!")
