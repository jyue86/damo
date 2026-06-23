from types import SimpleNamespace

import pytest
import torch

from damo.trainer import Trainer


class DummyLogger:
    def __init__(self):
        self.calls = []

    def log(self, data, step):
        self.calls.append((data, step))


def test_log_loss_adds_readable_rep_offset_metrics():
    trainer = Trainer.__new__(Trainer)
    trainer.logger = DummyLogger()
    trainer.model = SimpleNamespace(n_rep_joints=3)
    trainer.global_step = 17

    trainer._log_loss(
        {
            "weights": torch.tensor(0.5),
            "rep_offsets": torch.tensor(0.9),
            "total": torch.tensor(90.5),
        },
        epoch=2,
        is_training=False,
    )

    data, step = trainer.logger.calls[-1]
    assert step == 17
    assert data["val/loss/rep_offsets"] == pytest.approx(0.9)
    assert data["val/metric/rep_offsets_l1_per_coord_m"] == pytest.approx(0.1)
    assert data["val/metric/rep_offsets_l1_per_coord_cm"] == pytest.approx(10.0)
