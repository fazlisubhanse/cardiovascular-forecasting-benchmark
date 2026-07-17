from pathlib import Path
from src.config import load_config
from src.deep_learning.development_selection import development_origins


def test_development_targets_never_exceed_1999():
    c=load_config(Path("configs/phase2c_pilot.yaml"))
    for h in c["evaluation"]["horizons"]:
        origins=development_origins(1970,h,c["development"])
        assert origins and max(o+h for o in origins)<=1999
