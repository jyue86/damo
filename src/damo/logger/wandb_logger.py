from omegaconf import OmegaConf
from typing import Any, Mapping

from damo.logger.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
            self,
            project: str,
            run_name: str | None = None,
            mode: str = "online",
            dir: str | None = None,
            entity: str | None = None,
            group: str | None = None,
            tags: list[str] | None = None,
            **_: Any,
    ):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(
            project=project, name=run_name, mode=mode,
            dir=dir, entity=entity, group=group, tags=tags
        )

    def log_config(self, cfg: Any) -> None:
        self.wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.wandb.log(dict(metrics) if step is None else dict(metrics), step=step)

    def log_artifact(self, path: str, name: str | None = None, type: str | None = None) -> None:
        art = self.wandb.Artifact(name or path.split("/")[-1], type or "file")
        art.add_file(path)
        self.run.log_artifact(art)

    def info(self, msg: str) -> None:
        print(msg)

    def finish(self) -> None:
        self.run.finish()