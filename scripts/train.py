import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from damo.trainer import Trainer
from damo.logger.base_logger import BaseLogger
from damo.dataset.amass_dataset import make_dataloader
import damo.utils as utils

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(utils.Paths.config), config_name="config")
def run(cfg: DictConfig):
    log.info("\n" + OmegaConf.to_yaml(cfg))

    utils.set_seed(cfg.seed)
    utils.save_config(cfg, "config_resolved.yaml")
    device = utils.get_device(use_cuda=cfg.use_cuda)


    import time
    from line_profiler import LineProfiler
    from damo.dataset.amass_dataset import AmassDataset
    from damo.dataset.mocap_noise_augmentor import MocapNoiseAugmentor, sample_tracks_mask
    from damo.dataset.marker_sampler import MarkerSampler

    ds_train, train_loader = make_dataloader(cfg.dataset, train=True)
    ds_val, val_loader = make_dataloader(cfg.dataset, train=False)

    model = instantiate(cfg.model)
    model.to(device)

    logger: BaseLogger = instantiate(cfg.logger)
    logger.log_config(cfg)

    trainer = Trainer(
        cfg=cfg.trainer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
    )
    trainer.fit()

if __name__ == "__main__":
    run()