from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).parent

config = {
    "general": {
        "experiment_name": "baseline",
        "seed": 42,
    },
    "paths": {
        "train": str(ROOT / "data" / "train.csv"),
        "test": str(ROOT / "data" / "test.csv"),
    },
    "columns": {
        "target": "Survived",
        "id": "PassengerId",
    },
    "features": {
        "drop": ["PassengerId", "Ticket", "Cabin", "Name"],
        "categorical": ["Sex", "Embarked", "Title"],
    },
    "split": {"n_splits": 5},
    "mlp": {
        "lr": 0.001,
        "epochs": 200,
        "batch_size": 32,
    },
}

config = OmegaConf.create(config)
