import os
from pathlib import Path

class Paths:
    root = Path(__file__).resolve().parent.parent.parent.parent

    data = root / 'data'
    base_data = data / 'base'
    smpl_models = data / 'smpl' / 'models'
    amass_data = data / 'amass'

    train_data = data / 'train'
    test_data = data / 'test'
    raw_data = data / 'raw'

    config = root / 'conf'
