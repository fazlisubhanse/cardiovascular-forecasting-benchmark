from pathlib import Path
from src.phase2b_config import load_phase2b_config
from src.tuning.forward_search import candidate_grid


def test_phase2b_config_and_grid_sizes():
    config=load_phase2b_config(Path("configs/phase2b_ml.yaml"))
    assert [len(candidate_grid(config["models"][m])) for m in config["models"]]==[48,36,32]
    assert config["representations"]["diagnostic"]==["raw_level_direct"]
