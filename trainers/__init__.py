from .trainer import train

trainer_dict = {
    "trainer": train,
}

def get_trainer(name: str):
    name = name.lower()
    if name not in trainer_dict:
        raise ValueError(f"Trainer '{name}' not found. Available: {list(trainer_dict.keys())}")
    return trainer_dict[name]
