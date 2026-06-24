import yaml

from pathlib import Path

def load_config():
    root_dir = Path(__file__).parent
    
    with open(Path(__file__).parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    
    for key, value in config["paths"].items():
        path = Path(value)
        if not path.is_absolute():
            config["paths"][key] = root_dir / path
        else:
            config["paths"][key] = path
    
    return config