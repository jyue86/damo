import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from damo.dataset.amass_dataset import make_dataloader
import damo.utils as utils
import smplx
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(utils.Paths.config), config_name="config")
def run(cfg: DictConfig):
    log.info("\n" + OmegaConf.to_yaml(cfg))

    utils.set_seed(cfg.seed)
    device = utils.get_device(use_cuda=cfg.use_cuda)

    ds_train, train_loader = make_dataloader(cfg.dataset, train=True)
    # ds_val, val_loader = make_dataloader(cfg.dataset, train=False)

    # model = instantiate(cfg.model)
    # model.to(device)

    pbar = utils.make_tqdm_pbar(train_loader, 1)

    for bix, batch in enumerate(pbar, 1):
        pass

if __name__ == "__main__":
    run()